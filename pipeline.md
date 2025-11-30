# COOKING ASSISTANT CHATBOT PIPELINE

---

## BƯỚC 1: THU THẬP DỮ LIỆU
**Nguồn dữ liệu:**  
- Các trang web công thức nấu ăn (ví dụ: monngonmoingay)  

**Định dạng dữ liệu:**  
- JSON  

**Thông tin cần thu thập:**  
- Tên món ăn  
- Nguyên liệu  
- Hướng dẫn nấu  
- Thời gian nấu  
- Số lượng khẩu phần  
- Mức độ khó  

---

## BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU
**Các bước xử lý:**  
- Chuẩn hóa văn bản (loại bỏ ký tự đặc biệt, thống nhất chữ hoa/chữ thường)  
- Tokenization (phân tách câu và từ; sử dụng spaCy cho tiếng Việt)  
- Chuẩn hóa nguyên liệu  
- Loại bỏ dữ liệu trùng lặp và kiểm tra tính đầy đủ của công thức  

---

## BƯỚC 3: BIỂU DIỄN DỮ LIỆU (EMBEDDING)
**Phương pháp:**  
- Word embeddings / Sentence embeddings  
- Sử dụng HuggingFace Transformers  

**Mục đích:**  
- Chuyển nguyên liệu và công thức thành vector để tìm kiếm tương đồng  

---

## BƯỚC 4: CƠ SỞ DỮ LIỆU & TRUY VẤN
- Lưu embeddings trong FAISS vector index  
- Tìm kiếm ngữ nghĩa: tìm công thức tương tự dựa trên truy vấn người dùng  

---

## BƯỚC 5: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP)
- Phân loại intent (ví dụ: tìm công thức, hướng dẫn nấu ăn)  
- Nhận dạng thực thể (nguyên liệu, tên món ăn, thông tin dinh dưỡng)  
- Xử lý các truy vấn phức tạp của người dùng  

---

## BƯỚC 6: CHATBOT & GIAO DIỆN
**Backend:**  
- Logic sinh phản hồi  
- Kết nối với cơ sở dữ liệu công thức và vector search  
- Framework: FastAPI  

**Frontend:**  
- Giao diện chat trên web  
- Hiển thị công thức, hình ảnh, hướng dẫn từng bước  

---

## BƯỚC 7: SINH PHẢN HỒI
- Sử dụng Large Language Model (LLM) để tạo phản hồi tự nhiên  
- Gợi ý công thức từ kết quả truy vấn  

---

## BƯỚC 8: ĐÁNH GIÁ & PHẢN HỒI
- Đánh giá Precision / Recall cho truy vấn tìm kiếm tương đồng  
- Đánh giá chất lượng phản hồi chatbot (đúng nghĩa, đầy đủ, thân thiện)