const menuItems = document.querySelectorAll('.tips-list li');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendButton = document.getElementById('sendButton');

let selectedMode = null;

// Placeholders cho từng chế độ
const placeholders = {
    ingredient: 'Nhập nguyên liệu, ví dụ: thịt bò, cà rốt...',
    guide: 'Nhập tên món ăn...',
    difficulty: 'Nhập độ khó: dễ, trung bình, khó',
    time: 'Nhập thời gian nấu (phút)',
    serving: 'Nhập số khẩu phần',
    category: 'Nhập thể loại món ăn'
};

// Click vào menu
menuItems.forEach(item => {
    item.addEventListener('click', () => {
        // Xóa selected cũ
        menuItems.forEach(i => i.classList.remove('selected'));
        item.classList.add('selected');

        selectedMode = item.getAttribute('data-mode') || 'ingredient';

        // Reset chat và hiển thị thông báo chế độ mới
        chatMessages.innerHTML = '';
        if(selectedMode === 'ingredient'){
            addMessage(`Bạn đã chuyển sang chế độ: ${item.textContent.trim()}`);
        } else {
            addMessage(`Chế độ "${item.textContent.trim()}" hiện đang phát triển.`);
        }

        // Update placeholder
        chatInput.placeholder = placeholders[selectedMode] || 'Gõ tin nhắn của bạn...';
    });
});

function addMessage(text, isUser=false) {
    const msg = document.createElement('div');
    msg.className = `message ${isUser ? 'user' : 'ai'}`;
    msg.innerHTML = `<div class="message-bubble">${text}</div>`;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hàm gửi message
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    chatInput.value = '';

    if(selectedMode !== 'ingredient'){
        addMessage('Chức năng này đang phát triển, vui lòng thử chế độ Nguyên liệu.');
        return;
    }

    const loadingMessage = "Đang tìm công thức...";
    addMessage(loadingMessage, false);

    // Tạo list ingredients từ chuỗi người dùng nhập
    const ingredientsArray = message
        .split(",")
        .map(i => i.trim())
        .filter(i => i !== "");

    try {
        const res = await fetch("/api/smart_recipes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ingredients: ingredientsArray })
        });

        const data = await res.json();

        // xóa loading
        const lastMsg = chatMessages.lastChild;
        if (lastMsg && lastMsg.querySelector('.message-bubble').textContent === loadingMessage) {
            lastMsg.remove();
        }

        if (data.description) {
            addMessage(data.description);
        } else {
            addMessage("Không tìm thấy món ăn phù hợp.");
        }

    } catch (err) {
        console.error(err);
        const lastMsg = chatMessages.lastChild;
        if (lastMsg && lastMsg.querySelector('.message-bubble').textContent === loadingMessage) {
            lastMsg.remove();
        }
        addMessage("Có lỗi xảy ra khi tìm kiếm công thức.");
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') sendMessage();
});
