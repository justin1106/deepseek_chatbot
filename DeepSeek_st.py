import re
import json
import requests
import streamlit as st
import time

# -----------------------------
# <think> 태그 제거용
# -----------------------------
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_thinking(text: str):
    return THINK_PATTERN.sub("", text)

# -----------------------------
# UI 설정
# -----------------------------
st.set_page_config(page_title="DeepSeek Chat", page_icon="🤖", layout="wide")

with st.sidebar:
    st.markdown("## ⚙️ 설정")
    model = st.selectbox("모델 선택", ["deepseek8q", "deepseek8", "deepseek7", "deepseek1.5"])
    language = st.selectbox("답변 언어", ["한국어", "영어", "중국어"])
    use_gpu = st.toggle("GPU 사용", value=True, help="끄면 CPU로만 실행 (CUDA_VISIBLE_DEVICES=-1)")
    st.markdown("---")
    if st.button("새 대화"):
        st.session_state.messages = []
        st.rerun()

st.title("DeepSeek Chat")

# 대화 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m.get("answer", m.get("content", "")))

# 사용자 입력
user_text = st.chat_input("메시지를 입력하세요…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        progress_ph = st.empty()
        answer_ph = st.empty()

        cuda_value = "0" if use_gpu else "-1"
        base_url = "http://localhost:11434"

        # API 요청
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": f"답변은 {language}로 해주세요."},
                *[
                    {"role": m["role"], "content": m.get("content", m.get("answer", ""))}
                    for m in st.session_state.messages if m["role"] in ("system", "user")
                ]
            ],
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=600,
                               headers={"CUDA_VISIBLE_DEVICES": cuda_value}) as r:
                r.raise_for_status()
                full_text = ""
                thinking_mode = False
                token_count = 0
                max_estimated = 300  # 토큰 예측값
                bar_percent = 0.0

                for line in r.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    if line.startswith("data:"):
                        line = line[5:].strip()

                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    if "message" in obj and obj["message"].get("role") == "assistant":
                        piece = obj["message"].get("content", "")

                        # <think> 시작/종료 감지
                        if "<think>" in piece:
                            thinking_mode = True
                            bar_percent = 0
                        if "</think>" in piece:
                            thinking_mode = False
                            # 생각 종료 시 빠른 애니메이션
                            for p in range(int(bar_percent*100), 100):
                                bar_percent = p / 100
                                progress_ph.progress(bar_percent)
                                time.sleep(0.005)
                            progress_ph.empty()
                            continue

                        if thinking_mode:
                            # 생각 중 진행바만
                            token_count += 1
                            bar_percent = min(token_count / max_estimated, 0.9)
                            progress_ph.progress(bar_percent, text="*Thinking...*")
                        else:
                            # 답변 출력
                            full_text += piece
                            answer_ph.markdown(strip_thinking(full_text))

                    if obj.get("done"):
                        break

            # 대화 저장
            st.session_state.messages.append({
                "role": "assistant",
                "answer": strip_thinking(full_text)
            })

        except requests.exceptions.ConnectionError:
            st.error("❌ Ollama 서버에 연결할 수 없습니다.")
        except requests.HTTPError as e:
            st.error(f"❌ HTTP 오류: {e.response.status_code} — {e.response.text[:300]}")
        except Exception as e:
            st.error(f"❌ 예기치 못한 오류: {e}")
