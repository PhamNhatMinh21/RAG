# RAG Chatbot

Chatbot tiếng Việt hỏi đáp dựa trên PDF, dùng Gemini để trả lời, BAAI/bge-m3 để tạo embedding, FAISS để tìm tài liệu và Streamlit làm giao diện. Hỗ trợ hội thoại nhiều lượt và trích dẫn tên file, số trang.

## Chạy nhanh

1. Cài **Python 3.12 bản 64-bit**. Trên Windows, bật **Add python.exe to PATH** khi cài.
2. Clone repo hoặc chọn **Code → Download ZIP** trên GitHub rồi giải nén.
3. Mở terminal trong thư mục dự án. Trên Windows, chạy:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
   ```

   Trên macOS/Linux có Python 3.12:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python -m streamlit run streamlit_app.py
   ```

4. Mở địa chỉ Streamlit hiển thị trong terminal (thường là `http://localhost:8501`) nếu trình duyệt chưa tự mở.
5. Nhập **Gemini API key của bạn** ở thanh bên. Có thể lấy key tại [Google AI Studio](https://aistudio.google.com/apikey). Nhập câu hỏi sau khi thấy **RAG đã sẵn sàng**.

Hai PDF trong `papers/` được giữ trong repo để có tài liệu dùng ngay. Thêm hoặc thay PDF của bạn trong thư mục này rồi khởi động lại ứng dụng. PDF cần có văn bản trích xuất được; app chưa hỗ trợ OCR cho tài liệu chỉ chứa ảnh.

Lần đầu cần Internet để tải thư viện và model embedding BAAI/bge-m3. Model khá lớn, nên việc tải và lập chỉ mục có thể mất thời gian; cache được lưu tại `.cache/` và dùng lại ở những lần sau. Ứng dụng chạy embedding trên CPU. Các lượt hỏi đáp cần Internet để gọi Gemini và chịu hạn mức/chi phí theo tài khoản Google của bạn.

## Cấu hình API key bằng file

Nếu muốn lưu cấu hình trên máy, sao chép `.env.example` thành `.env`, rồi điền:

```dotenv
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Ứng dụng cũng chấp nhận biến `GEMINI_API_KEY`. Nếu model mặc định không khả dụng với API key của bạn, đổi **Model Gemini** ở thanh bên hoặc `GEMINI_MODEL` trong `.env` sang model bạn có quyền dùng. API key trong `.env` được ưu tiên hơn ô nhập trong giao diện.

Không đưa `.env` thật lên Git. Repo chỉ chứa `.env.example` với key trống; người tải về cần dùng key của riêng họ.

## Chạy lại và chế độ terminal

Sau khi đã cài thư viện, các lần sau chỉ cần mở terminal trong thư mục dự án và chạy:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Trên macOS/Linux, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python` trong các lệnh tương ứng. Nếu báo thiếu `venv` khi tạo môi trường, cài gói venv cho Python 3.12 bằng trình quản lý gói của hệ điều hành.

Chạy ở terminal thay cho giao diện web:

```powershell
.\.venv\Scripts\python.exe chat.py
```

Chế độ terminal cần API key trong `.env` hoặc biến môi trường. Nhập `/exit` để thoát. Với giao diện web, nhấn `Ctrl+C` trong terminal hoặc nút **Tắt app** để dừng.

## File đưa lên Git

- `chat.py`, `streamlit_app.py`: xử lý RAG và giao diện.
- `papers/`: tài liệu PDF đi kèm.
- `requirements.txt`: danh sách thư viện cần cài.
- `.env.example`, `.gitignore`, `.gitattributes`: cấu hình mẫu và quy tắc Git.

`.env`, `.venv*`, `.cache/`, `.tmp/`, `__pycache__/` và file sinh tự động được bỏ qua bởi `.gitignore`. Các thư mục này vẫn có thể giữ trên máy để dùng tiếp; người tải repo sẽ tạo môi trường và cache của riêng họ.
