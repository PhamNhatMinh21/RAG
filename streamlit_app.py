import os

import streamlit as st

from chat import BASE_DIR, EMBEDDING_MODEL, GEMINI_MODEL, build_rag_chain, format_answer_citations


st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG chatbot")
st.caption("Đặt câu hỏi về các nội dung có trong tài liệu")

if "messages" not in st.session_state:
    st.session_state.messages = []

pdf_files = sorted((BASE_DIR / "papers").rglob("*.pdf"))

with st.sidebar:
    st.header("Cấu hình")
    model = st.text_input("Model Gemini", value=GEMINI_MODEL)
    env_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_api_key:
        api_key = env_api_key
        st.success("Đã đọc API key từ .env")
    else:
        api_key = st.text_input("Gemini API key", type="password")

    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption("Thiết bị: CPU")
    st.divider()
    st.subheader(f"Tài liệu ({len(pdf_files)})")
    for pdf_file in pdf_files:
        st.write(str(pdf_file.relative_to(BASE_DIR / "papers")))
    if not pdf_files:
        st.warning("Chưa có tài liệu trong thư mục papers.")

    st.divider()
    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("Tắt app", use_container_width=True):
        os._exit(0)
    st.caption("Lần đầu cần đọc file và tạo chỉ mục. Các lượt chat tiếp theo dùng lại RAG đã tạo.")


@st.cache_resource(show_spinner=False)
def get_rag_chain(api_key, model, pdf_signature, embedding_model):
    """Pipeline với nguồn rút gọn, gộp ở cuối câu trả lời."""
    return build_rag_chain(api_key=api_key, model=model)


rag_chain = None
if not pdf_files:
    st.info("Hãy thêm tài liệu vào thư mục papers.")
elif not api_key:
    st.info("Hãy nhập Gemini API key hoặc thêm API key vào file .env.")
elif not model.strip():
    st.info("Hãy nhập tên model Gemini.")
else:
    pdf_signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in pdf_files
    )
    try:
        with st.spinner("Đang chuẩn bị tài liệu..."):
            rag_chain = get_rag_chain(api_key, model.strip(), pdf_signature, EMBEDDING_MODEL)
        st.success("RAG đã sẵn sàng")
    except Exception as error:
        st.error(f"Không thể khởi tạo RAG: {error}")


def display_message(role, content):
    with st.chat_message(role):
        st.markdown(f"**{'Bạn' if role == 'user' else 'Trợ lý'}**")
        st.markdown(format_answer_citations(content) if role == "assistant" else content)


# Form dùng nút chữ thay cho biểu tượng gửi tin nhắn.
conversation = st.container()
with conversation:
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input(
        "Câu hỏi", placeholder="Nhập câu hỏi về tài liệu...", disabled=rag_chain is None,
    )
    submitted = st.form_submit_button("Gửi", disabled=rag_chain is None)

if submitted and question.strip() and rag_chain is not None:
    question = question.strip()
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": question})
    with conversation:
        display_message("user", question)
        try:
            with st.spinner("Đang tìm trong tài liệu và trả lời..."):
                answer = rag_chain.invoke({"question": question, "history": history})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            display_message("assistant", answer)
        except Exception as error:
            # Chỉ giữ các lượt đã trả lời trong ngữ cảnh gửi tới mô hình.
            st.session_state.messages.pop()
            st.error(f"Không thể trả lời: {error}. Bạn có thể gửi lại câu hỏi hoặc hỏi tiếp.")
