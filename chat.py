import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = "BAAI/bge-m3"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

MARKDOWN_SEPERATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

template = (
    "Bạn là một trợ lý nghiêm ngặt, tập trung vào việc trả lời dựa trên nguồn tài liệu trong kho tri thức riêng.\n"
    "QUY TẮC:\n"
    "1) CHỈ sử dụng ngữ cảnh được cung cấp để trả lời.\n"
    "2) Nếu câu trả lời không được nêu rõ trong ngữ cảnh, hãy trả lời: "
    "\"Tôi không biết dựa trên các tài liệu được cung cấp.\"\n"
    "3) KHÔNG sử dụng kiến thức bên ngoài, suy đoán hoặc thông tin từ web.\n"
    "4) Đánh dấu nguồn thực sự dùng để trả lời bằng [[SOURCE:tên_file.pdf:số_trang]], "
    "dùng đúng mã nguồn trong ngữ cảnh. Không ghi đường dẫn thư mục. "
    "Ứng dụng sẽ bỏ dấu nguồn trong nội dung và gom nguồn xuống cuối câu trả lời. "
    "Không tự viết mục nguồn; không liệt kê nguồn chưa sử dụng. "
    "Nếu không có đáp án trong tài liệu thì không đánh dấu nguồn.\n"
    "5) Dùng lịch sử hội thoại để hiểu câu hỏi nối tiếp. Lịch sử không phải nguồn tài liệu.\n"
    "6) Không dùng emoji hoặc biểu tượng trang trí trong câu trả lời.\n\n"
    "Ngữ cảnh:\n{context}"
)


def normalize_chat_input(value):
    """Giữ tương thích với invoke(question), đồng thời nhận lịch sử từng phiên."""
    if isinstance(value, str):
        return {"question": value, "history": []}
    # Chỉ giới hạn ngữ cảnh gửi tới LLM, không giới hạn số lượt hoặc lịch sử hiển thị.
    history = trim_messages(
        value.get("history", []), max_tokens=12000, token_counter="approximate",
        strategy="last", start_on="human", end_on="ai",
    )
    return {"question": value["question"], "history": history}


def document_source(doc):
    page = doc.metadata.get("page_number")
    if page is None:
        # PyPDFLoader đánh số trang từ 0, trích dẫn hiển thị từ trang 1.
        index = doc.metadata.get("page")
        page = index + 1 if isinstance(index, int) else "không rõ"
    filename = Path(str(doc.metadata.get("source", "không rõ")).replace("\\", "/")).name
    return f"{filename}:{page}"


def format_documents(documents):
    contexts = []
    for doc in documents:
        contexts.append(
            f"Nguồn: {document_source(doc)}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(contexts)


SOURCE_MARKER = re.compile(
    r"[ \t]*(?:\[\[SOURCE:(?P<marker>[^\]\n]+)\]\]|"
    r"\{(?P<legacy>[^{}\n]*\.pdf\s*(?:;\s*trang:\s*|:)\s*(?:\d+|không rõ))\})",
    re.IGNORECASE,
)


def format_answer_citations(answer, allowed_sources=None):
    """Chỉ gom các nguồn được trích trong đáp án, không liệt kê toàn bộ kết quả tìm kiếm."""
    sources = []

    def collect(match):
        reference = (match.group("marker") or match.group("legacy")).strip()
        parts = re.split(r";\s*trang:\s*|:(?=\s*(?:\d+|không rõ)\s*$)", reference)
        if len(parts) == 2:
            filename = Path(parts[0].strip().replace("\\", "/")).name
            source = f"{filename}:{parts[1].strip()}"
            if (allowed_sources is None or source in allowed_sources) and source not in sources:
                sources.append(source)
        return ""

    body = SOURCE_MARKER.sub(collect, answer).strip()
    if sources:
        body = re.sub(r"[ \t]+([.,;!?])", r"\1", body)
        return body + "\n\n**Nguồn đã sử dụng**\n\n" + "\n".join(f"- `{source}`" for source in sources)
    return body


def create_conversation_chain(retriever, llm):
    """Tìm tài liệu theo câu hỏi đầy đủ và trả lời với lịch sử của người dùng."""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Dựa vào lịch sử, viết lại câu hỏi mới thành một câu hỏi độc lập "
         "để tìm kiếm tài liệu. Giữ ngôn ngữ của người dùng. "
         "Chỉ trả về câu hỏi, không trả lời và không thêm thông tin mới."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    def retrieve(inputs):
        query = rewrite_chain.invoke(inputs) if inputs["history"] else inputs["question"]
        documents = retriever.invoke(query)
        return {
            "context": format_documents(documents),
            "sources": {document_source(doc) for doc in documents},
        }

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    return (
        RunnableLambda(normalize_chat_input)
        | RunnablePassthrough.assign(retrieval=RunnableLambda(retrieve))
        | RunnablePassthrough.assign(context=lambda inputs: inputs["retrieval"]["context"])
        | RunnablePassthrough.assign(answer=answer_prompt | llm | StrOutputParser())
        | RunnableLambda(lambda inputs: format_answer_citations(inputs["answer"], inputs["retrieval"]["sources"]))
    )


def build_rag_chain(api_key=None, model=GEMINI_MODEL):
    """Khởi tạo pipeline gốc để giao diện Streamlit có thể dùng lại."""
    # Các thư viện đọc PDF và embedding chỉ được nạp khi khởi tạo RAG.
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Hãy đặt GOOGLE_API_KEY hoặc GEMINI_API_KEY trong .env.")

    # 1. đọc tài liệu
    loader = DirectoryLoader(
        path=str(BASE_DIR / "papers"),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
        use_multithreading=True,
    )
    docs = loader.load()

    # 2. chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPERATORS,
    )
    splits = text_splitter.split_documents(docs)
    if not splits:
        raise ValueError("Không có nội dung để lập chỉ mục. Hãy thêm PDF có văn bản vào papers.")

    # 3. embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(BASE_DIR / ".cache" / "huggingface"),
    )

    # 4. vector database
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # 5. retrieval
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2},
    )

    # 6. LLM gemini
    llm = ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        vertexai=False,
        temperature=0,  # mức độ ngẫu nhiên của mô hình, càng lớn càng sáng tạo.
    )

    # 7. pipeline dùng chung cho terminal và Streamlit
    return create_conversation_chain(retriever, llm)


def run_terminal_chat(rag_chain):
    history = []
    print("Chat đã sẵn sàng. Nhấn Ctrl+C hoặc nhập /exit để tắt.")
    while True:
        try:
            question = input("Bạn: ").strip()
            if question.lower() == "/exit":
                break
            if not question:
                continue
            try:
                answer = rag_chain.invoke({"question": question, "history": history.copy()})
            except Exception as error:
                print(f"Không thể trả lời: {error}. Bạn có thể gửi câu hỏi tiếp theo.")
                continue
            print(f"Trợ lý: {answer}")
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    rag_chain = build_rag_chain()
    run_terminal_chat(rag_chain)
