// Data cho Quick Tips Modal
const tipsData = {
  ingredient: {
    title: "Gợi ý từ nguyên liệu",
    icon: "fas fa-check-circle",
    description:
      "Bạn có thể cung cấp các nguyên liệu có sẵn trong bếp của mình, và tôi sẽ gợi ý những món ăn ngon mà bạn có thể nấu.",
    example:
      '<strong>Ví dụ:</strong> "Tôi có thịt bò, cà rốt và hành tây. Hãy gợi ý món ăn dựa trên những nguyên liệu này."',
  },
  guide: {
    title: "Hướng dẫn nấu ăn",
    icon: "fas fa-sliders-h",
    description:
      "Bạn đưa ra một món ăn, tôi sẽ đưa ra hướng dẫn chi tiết cách làm món đó.",
    example:
      '<strong>Ví dụ:</strong> "Hướng dẫn cách nấu phở gà"; "Làm thế nào để nấu cơm chiên?"',
  },
  difficulty: {
    title: "Gợi ý theo độ khó",
    icon: "fas fa-exchange-alt",
    description:
      "Bạn có thể yêu cầu các món ăn dễ, vừa hoặc khó. Tôi sẽ gợi ý những món phù hợp với kỹ năng nấu của bạn.",
    example:
      '<strong>Ví dụ:</strong> "Gợi ý món ăn dễ làm."; "Tôi muốn nấu các món có độ khó trung bình."',
  },
  time: {
    title: "Gợi ý theo thời gian nấu",
    icon: "fas fa-stopwatch",
    description:
      "Chỉ cần đưa ra thời gian bạn giành ra việc nấu nướng, tôi sẽ gợi ý món phù hợp với bạn.",
    example:
      '<strong>Ví dụ:</strong> "Gợi ý các món nấu trong 30 phút?"; "Tôi có 1 tiếng thì nấu món gì?"',
  },
  serving: {
    title: "Gợi ý theo khẩu phần",
    icon: "fas fa-users",
    description:
      "Bạn có thể cho biết số lượng người ăn, và tôi sẽ gợi ý các món ăn và lượng nguyên liệu phù hợp.",
    example:
      '<strong>Ví dụ:</strong> "Nấu cho 4 người"; "Tôi cần gợi ý món ăn cho 6 người"',
  },
  category: {
    title: "Gợi ý theo thể loại",
    icon: "fas fa-list-alt",
    description:
      "Tôi có thể gợi ý các món ăn theo thể loại xào, chiên, hầm, nướng,..",
    example:
      '<strong>Ví dụ:</strong> "Tôi muốn nấu món xào"; "Gợi ý món nướng ngon."',
  },
  "mix-choice": {
    title: "Gợi ý kết hợp nhiều tiêu chí",
    icon: "fas fa-magic",
    description:
      "Bạn có thể kết hợp nhiều tiêu chí như: nguyên liệu + thời gian, độ khó + khẩu phần, hay thể loại + thời gian. Tôi sẽ gợi ý những món ăn phù hợp nhất cho bạn.",
    example:
      '<strong>Ví dụ:</strong> "Tôi muốn nấu trong 30 phút với các nguyên liệu: thịt bò, cà rốt, cần tây."; "Gợi món hầm dễ làm cho 6 người."',
  },
};

// ===========================
// Modal Functions
// ===========================
function initModal() {
  const menuItems = document.querySelectorAll(".menu-item");
  const modal = document.getElementById("modalOverlay");
  const closeBtn = document.getElementById("modalCloseBtn");

  // Click vào menu item để hiện modal
  menuItems.forEach((item) => {
    item.addEventListener("click", () => {
      const mode = item.dataset.mode;
      const data = tipsData[mode];

      document.getElementById("modalTitle").textContent = data.title;
      document.getElementById("modalContent").textContent = data.description;
      document.getElementById("modalExample").innerHTML = data.example;

      const iconEl = document.getElementById("modalIcon");
      iconEl.innerHTML = `<i class="${data.icon}"></i>`;

      modal.classList.add("active");
    });
  });

  // Đóng modal khi nhấn nút
  closeBtn.addEventListener("click", () => {
    modal.classList.remove("active");
  });

  // Đóng modal khi nhấn ra ngoài
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.remove("active");
    }
  });
}

// ===========================
// Chat Functions
// ===========================
function initChat() {
  const chatInput = document.getElementById("chatInput");
  const sendButton = document.getElementById("sendButton");
  const chatMessages = document.getElementById("chatMessages");

  function addMessage(text, isUser = false) {
    const message = document.createElement("div");
    message.className = `message ${isUser ? "user" : "ai"}`;
    message.innerHTML = `<div class="message-bubble">${text}</div>`;
    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function addLoading() {
    const msg = document.createElement("div");
    msg.className = "message ai loading";
    msg.innerHTML = `<div class="message-bubble">Đang xử lý...</div>`;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeLoading() {
    const last = chatMessages.lastChild;
    if (last && last.classList.contains("loading")) last.remove();
  }

  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage(text, true);
    chatInput.value = "";

    addLoading();

    try {
      const response = await fetch("http://localhost:8000/process_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      removeLoading();
      console.log("Response:", data);

      // Intent 1: suggest_dishes
      if (data.intent === "suggest_dishes") {
        if (!data.top_dishes || data.top_dishes.length === 0) {
          addMessage("Không tìm thấy món ăn phù hợp.");
          return;
        }

        let html = "";
        html += `<b>🎯 Gợi ý món ăn phù hợp:</b><br>`;
        html += data.top_dishes
          .slice(0, 10)
          .map((d) => `• ${d}`)
          .join("<br>");

        html += `<br><br><b>📘 Mô tả:</b><br>${data.description}`;

        addMessage(html);
        return;
      }

      // Intent 2: cooking_guide
      else if (data.intent === "cooking_guide") {
        if (data.error) {
          addMessage(data.error);
          return;
        }

        let html = `<b>🍽 Hướng dẫn nấu món: ${data.dish_name}</b><br><br>`;

        // Xử lý nguyên liệu
        html += `<b>🧂 Nguyên liệu:</b><br>`;

        if (Array.isArray(data.ingredients)) {
          html += data.ingredients.map((i) => `• ${i}`).join("<br>");
        } else {
          html += "Không có dữ liệu nguyên liệu.";
        }

        // Xử lý các bước thực hiện
        html += `<br><br><b>👨‍🍳 Các bước thực hiện:</b><br>`;

        let steps = [];

        if (Array.isArray(data.steps_smooth)) {
          steps = data.steps_smooth;
        } else if (typeof data.steps_smooth === "string") {
          steps = data.steps_smooth.split("\n");
        } else {
          html += "Không có hướng dẫn.";
          addMessage(html);
          return;
        }

        html += steps
          .filter((s) => s.trim().length > 0)
          .map((step, idx) => `${idx + 1}. ${step.trim()}`)
          .join("<br>");

        addMessage(html);
        return;
      }
      // Fallback
      else {
        addMessage(data.error || "Xin lỗi, tôi chưa hiểu yêu cầu của bạn.");
      }
    } catch (err) {
      removeLoading();
      console.error(err);
      addMessage("❌ Lỗi kết nối tới server.");
    }
  }

  sendButton.addEventListener("click", sendMessage);

  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });
}

// ===========================
// Initialize App
// ===========================
document.addEventListener("DOMContentLoaded", () => {
  initModal();
  initChat();
});
