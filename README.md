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
 5.  Sau đó điều hướng đến thư mục backend và chạy lệnh:
 ```bash
uvicorn main:app --reload
 ```