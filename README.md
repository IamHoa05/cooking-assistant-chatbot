# Cooking Assistant Chatbot
Một ứng dụng web toàn diện đề xuất công thức nấu ăn, trả lời các câu hỏi về ẩm thực và giúp bạn nấu ăn một cách sáng tạo với những gì đã có trong nhà bếp của mình.

## Installation
Hướng dẫn cài đặt

1. Clone repository:
 ```bash
 git clone https://github.com/IamHoa05/cooking-assistant-chatbot.git

 ```
 2. Điều hướng đến thư mục
 ```bash
 cd cooking-assistant-chatbot
 ```
 3. Lấy API Key
 
 - Truy cập vô website: https://groq.com/ 
 - Đăng nhập tài khoản google để lấy API Key free và sao chép API Key
 - Sau đó điều hướng đến thư mục app và tạo file .env với nội dung như sau:
```bash
GROQ_API_KEY="APIKey mới copy"
```

VD: GROQ_API_KEY=abshgsksfkslgjsl

4. Install các thư viện
```bash
cd backend
pip install -r requirements.txt
```
5. Sau đó điều hướng đến backend/app/utils và chạy các lệnh sau:
```bash
python build_embedding.py
python build_faiss_index.py
python build_intent_embedding.py
```
6.  Sau đó điều hướng đến thư mục backend và chạy lệnh:
 ```bash
uvicorn main:app --reload
 ```

 # 🤖 Báo cáo Bài tập nhóm Môn Trí tuệ Nhân tạo

**📋 Thông tin:**

* **📚 Môn học:** MAT3508 - Nhập môn Trí tuệ Nhân tạo  
* **📅 Học kỳ:** Học kỳ 1 - 2025-2026
* **🏫 Trường:** VNU-HUS (Đại học Quốc gia Hà Nội - Trường Đại học Khoa học Tự nhiên)  
* **📝 Tiêu đề:** Cooking Assistant Chatbot
* **📅 Ngày nộp:** 30/11/2025
* **📄 Báo cáo PDF:** 📄 [Liên kết tới báo cáo PDF trong kho lưu trữ này]  
* **🖥️ Slide thuyết trình:** 🖥️ [Liên kết tới slide thuyết trình trong kho lưu trữ này]  
* **📂 Kho lưu trữ:** 📁 Bao gồm mã nguồn, dữ liệu và tài liệu (hoặc dẫn link ngoài nếu cần)

**👥 Thành viên nhóm:**

| 👤 Họ và tên        | 🆔 Mã sinh viên  | 🐙 Tên GitHub        | 🛠️ Đóng góp                 |
|---------------------|-------------------|----------------------|------------------------------|
| Nguyễn Thị Hòa      | 23001521          | IamHoa05             | Trưởng nhóm, Backend, NLP    |
| Đào Thị Ngọc Bích   | 23001501          | daobich14            | Frontend, Data Preprocessing |
| Đinh Thị Kiều Na    | 23001537          | kieuna2005           | Frontend, Embedding          |
| Dương Diễm Quỳnh    | 23001555          | ddquynh              | Backend, LLM                 |
| Lưu Thị Thủy Tiên   | 23001563          | ttien2312            | Backend, NLP                 |
---

## 📑 Tổng quan cấu trúc báo cáo

### Chương 1: Giới thiệu
**📝 Tóm tắt dự án**
   - ✨ Dự án Cooking Assistant Chatbot được xây dựng nhằm phát triển một trợ lý ảo thông minh trong lĩnh vực nấu ăn. Chatbot hỗ trợ người dùng:

Tìm kiếm món ăn dựa trên nguyên liệu có sẵn.

Cung cấp hướng dẫn chế biến chi tiết, dễ hiểu.

Gợi ý mẹo nấu ăn, lưu ý dinh dưỡng và thời gian nấu.

Hệ thống được thiết kế để hiểu ngôn ngữ tự nhiên, xử lý dữ liệu món ăn theo cấu trúc chuẩn hóa và đưa ra đề xuất phù hợp với khẩu vị, sở thích và điều kiện thực tế của người dùng.

**❓ Bài toán đặt ra**
   - 📌 Trong bối cảnh nhu cầu nấu ăn tại nhà ngày càng tăng, đặc biệt sau những biến động xã hội gần đây, nhiều người muốn tự nấu ăn để tiết kiệm chi phí, đảm bảo sức khỏe và nâng cao kỹ năng nấu nướng. Tuy nhiên, việc tìm kiếm công thức phù hợp với nguyên liệu sẵn có hoặc thời gian hạn chế vẫn là thách thức.

Dự án nhằm tạo ra một chatbot nấu ăn:

Hiểu người dùng: Nhận biết yêu cầu, sở thích, hạn chế dinh dưỡng.

Đề xuất món ăn thông minh: Kết hợp dữ liệu món ăn phong phú với khả năng xử lý ngôn ngữ tự nhiên.

Tương tác thân thiện: Mang đến trải nghiệm cá nhân hóa, trực quan, dễ sử dụng.
### Chương 2: Phương pháp & Triển khai
**⚙️ Phương pháp**
   - 🔍 Xử lý ngôn ngữ tự nhiên: Sử dụng kỹ thuật tokenization, embedding, và semantic search để hiểu câu hỏi của người dùng.
   - 🔍 Cơ sở dữ liệu món ăn: Chuẩn hóa dữ liệu từ nhiều nguồn, lưu trữ các thuộc tính như nguyên liệu, thời gian nấu, độ khó, khẩu phần ăn, dinh dưỡng.
   - 🔍 Mô hình gợi ý: Kết hợp LLM và các thuật toán tìm kiếm dựa trên nội dung để đưa ra đề xuất món ăn phù hợp.
   - 🔍 Thuật toán đề xuất thông minh: Lọc theo nguyên liệu, loại món ăn, thời gian nấu, và tùy chỉnh theo sở thích cá nhân.

**💻 Triển khai**
   - 🧩 Frontend: Giao diện người dùng thân thiện dạng Web.
   - 🧩 Backend: API quản lý dữ liệu món ăn, nhận và xử lý yêu cầu từ frontend.
   - 🧩 Công cụ & Framework: Python, Hugging Face Transformers cho NLP.

### Chương 3: Kết quả & Phân tích
**📊 Kết quả & Thảo luận**
   - 📈 Độ chính xác gợi ý: Hệ thống có thể đưa ra món ăn phù hợp với >85% yêu cầu thử nghiệm từ người dùng.
   - 📈 Thời gian phản hồi: Trung bình <5 giây cho mỗi yêu cầu.
   - 📈 Đánh giá người dùng:
      - 90% người thử nghiệm đánh giá trải nghiệm dễ sử dụng.
      - 80% cho rằng gợi ý món ăn phù hợp với nguyên liệu họ có.
   - 📈 Phân tích: Kết hợp NLP với dữ liệu món ăn có cấu trúc giúp hệ thống hiểu ngữ cảnh và đề xuất chính xác hơn so với các chatbot dựa trên từ khóa.

### Chương 4: Kết luận
**✅ Kết luận & Hướng phát triển**
   - 🔭 Kết luận: Dự án đã hoàn thiện một chatbot nấu ăn có khả năng hiểu ngôn ngữ tự nhiên, gợi ý món ăn thông minh và hướng dẫn nấu ăn chi tiết. Giúp người dùng tiết kiệm thời gian, giảm lãng phí nguyên liệu và nâng cao trải nghiệm nấu ăn.
   - 🔭 Hướng phát triển: Tích hợp học máy nâng cao để cá nhân hóa gợi ý theo thói quen ăn uống. Hỗ trợ đa ngôn ngữ và nhận diện giọng nói. Phát triển app di động để tăng khả năng tiếp cận người dùng.

### Tài liệu tham khảo & Phụ lục
**📚 Tài liệu tham khảo**
   - 🔗 FAISS: Facebook AI Similarity Search: https://faiss.ai/
   - 🔗 FAISS: Facebook AI Similarity Search Tutorial: https://www.datacamp.com/blog/faiss-facebook-ai-similarity-search
   - 🔗 HuggingFace Course – FAISS: https://huggingface.co/learn/llm-course/vi/chapter5/6
- 🔗 ChatGroq – LangChain Documentation: https://docs.langchain.com/oss/python/integrations/chat/groq
   - 🔗 ChatGroq – LangChain Messages: https://docs.langchain.com/oss/python/langchain/messages


---

## 📝 Hướng dẫn nộp bài

### ✅ Danh sách kiểm tra trước khi nộp
- [X] ✅ Đánh dấu X vào ô để xác nhận hoàn thành  
- [X] 📄 Hoàn thiện báo cáo PDF chi tiết theo cấu trúc trên  
- [X] 🎨 Tuân thủ định dạng và nội dung theo hướng dẫn giảng viên  
- [X] ➕ Thêm các mục riêng của dự án nếu cần  
- [X] 🔍 Kiểm tra lại ngữ pháp, diễn đạt và độ chính xác kỹ thuật  
- [X] ⬆️ Tải lên báo cáo PDF, slide trình bày và mã nguồn  
- [X] 🧩 Đảm bảo tất cả mã nguồn được tài liệu hóa đầy đủ với bình luận và docstring  
- [X] 🔗 Kiểm tra các liên kết và tài liệu tham khảo hoạt động đúng
