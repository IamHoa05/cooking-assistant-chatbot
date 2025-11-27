// // const menuItems = document.querySelectorAll('.tips-list li');
// // const chatMessages = document.getElementById('chatMessages');
// // const chatInput = document.getElementById('chatInput');
// // const sendButton = document.getElementById('sendButton');

// // let selectedMode = null;

// // // Placeholders cho từng chế độ
// // const placeholders = {
// //     ingredient: 'Nhập nguyên liệu, ví dụ: thịt bò, cà rốt...',
// //     guide: 'Nhập tên món ăn...',
// //     difficulty: 'Nhập độ khó: dễ, trung bình, khó',
// //     time: 'Nhập thời gian nấu (phút)',
// //     serving: 'Nhập số khẩu phần',
// //     category: 'Nhập thể loại món ăn'
// // };

// // // Click vào menu
// // menuItems.forEach(item => {
// //     item.addEventListener('click', () => {
// //         // Xóa selected cũ
// //         menuItems.forEach(i => i.classList.remove('selected'));
// //         item.classList.add('selected');

// //         selectedMode = item.getAttribute('data-mode') || 'ingredient';

// //         // Reset chat và hiển thị thông báo chế độ mới
// //         chatMessages.innerHTML = '';
// //         if(selectedMode === 'ingredient'){
// //             addMessage(`Bạn đã chuyển sang chế độ: ${item.textContent.trim()}`);
// //         } else {
// //             addMessage(`Chế độ "${item.textContent.trim()}" hiện đang phát triển.`);
// //         }

// //         // Update placeholder
// //         chatInput.placeholder = placeholders[selectedMode] || 'Gõ tin nhắn của bạn...';
// //     });
// // });

// // function addMessage(text, isUser=false) {
// //     const msg = document.createElement('div');
// //     msg.className = `message ${isUser ? 'user' : 'ai'}`;
// //     msg.innerHTML = `<div class="message-bubble">${text}</div>`;
// //     chatMessages.appendChild(msg);
// //     chatMessages.scrollTop = chatMessages.scrollHeight;
// // }


// // async function sendMessage() {
// //     const message = chatInput.value.trim();
// //     if (!message) return;

// //     addMessage(message, true);
// //     chatInput.value = '';

// //     if (selectedMode !== 'ingredient') {
// //         addMessage('Chức năng này đang phát triển, vui lòng thử chế độ Nguyên liệu.');
// //         return;
// //     }

// //     const loadingMessage = "Đang tìm công thức...";
// //     addMessage(loadingMessage, false);

// //     // Tạo list ingredients từ chuỗi người dùng nhập
// //     const ingredientsArray = message
// //         .split(",")
// //         .map(i => i.trim())
// //         .filter(i => i !== "");

// //     try {
// //         const res = await fetch("/api/smart_recipes", {
// //             method: "POST",
// //             headers: { "Content-Type": "application/json" },
// //             body: JSON.stringify({ ingredients: ingredientsArray })
// //         });

// //         const data = await res.json();

// //         // xóa loading
// //         const lastMsg = chatMessages.lastChild;
// //         if (lastMsg && lastMsg.querySelector('.message-bubble').textContent === loadingMessage) {
// //             lastMsg.remove();
// //         }

// //         // Debug: log ra console
// //         console.log("DEBUG - smart_recipes result:", data);

// //         if (data.top_dishes && data.top_dishes.length > 0) {
// //             addMessage("Món tìm được: " + data.top_dishes.join(", "));
// //         }

// //         if (data.description) {
// //             addMessage(data.description);
// //         } else {
// //             addMessage("Không tìm thấy món ăn phù hợp.");
// //         }

// //     } catch (err) {
// //         console.error(err);
// //         const lastMsg = chatMessages.lastChild;
// //         if (lastMsg && lastMsg.querySelector('.message-bubble').textContent === loadingMessage) {
// //             lastMsg.remove();
// //         }
// //         addMessage("Có lỗi xảy ra khi tìm kiếm công thức.");
// //     }
// // }

// // // Event listeners
// // sendButton.addEventListener('click', sendMessage);
// // chatInput.addEventListener('keypress', e => {
// //     if (e.key === 'Enter') sendMessage();
// // });

// // =======================================
// // 1. DOM & STATE
// // =======================================
// const menuItems = document.querySelectorAll('.tips-list li');
// const chatMessages = document.getElementById('chatMessages');
// const chatInput = document.getElementById('chatInput');
// const sendButton = document.getElementById('sendButton');

// let selectedMode = null;

// // Placeholder cho từng chế độ
// const placeholders = {
//     ingredient: 'Nhập nguyên liệu, ví dụ: thịt bò, cà rốt...',
//     guide: 'Nhập tên món ăn...',
//     difficulty: 'Nhập độ khó: dễ, trung bình, khó',
//     time: 'Nhập thời gian nấu (phút)',
//     serving: 'Nhập số khẩu phần',
//     category: 'Nhập thể loại món ăn'
// };

// // =======================================
// // 2. UI FUNCTIONS
// // =======================================
// function addMessage(text, isUser = false) {
//     const msg = document.createElement('div');
//     msg.className = `message ${isUser ? 'user' : 'ai'}`;
//     msg.innerHTML = `<div class="message-bubble">${text}</div>`;
//     chatMessages.appendChild(msg);
//     chatMessages.scrollTop = chatMessages.scrollHeight;
// }

// function addLoading(text = "Đang xử lý...") {
//     const msg = document.createElement('div');
//     msg.className = "message ai loading";
//     msg.innerHTML = `<div class="message-bubble">${text}</div>`;
//     chatMessages.appendChild(msg);
//     chatMessages.scrollTop = chatMessages.scrollHeight;
// }

// function removeLastLoading() {
//     const lastMsg = chatMessages.lastChild;
//     if (lastMsg && lastMsg.classList.contains("loading")) {
//         lastMsg.remove();
//     }
// }

// // =======================================
// // 3. API LAYER
// // =======================================
// async function apiSmartRecipes(ingredientsArray) {
//     const res = await fetch("/api/smart_recipes", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ ingredients: ingredientsArray })
//     });
//     const data = await res.json();
//     console.log("DEBUG API smart_recipes:", data);
//     return data;
// }

// async function apiRecipeDetail(dishName) {
//     const res = await fetch("/api/recipe_detail_llm", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ dish: dishName })
//     });
//     const data = await res.json();
//     console.log("DEBUG API recipe_detail_llm:", data);
//     return data;
// }

// async function apiRecipesByDifficulty(difficulty) {
//     try {
//         const response = await fetch("/api/recipes_by_difficulty", {
//             method: "POST",
//             headers: {
//                 "Content-Type": "application/json"
//             },
//             body: JSON.stringify({ difficulty })
//         });
//         return await response.json();
//     } catch (err) {
//         console.error("Error calling recipes_by_difficulty API:", err);
//         return { top_dishes: [], description: "Lỗi khi gọi API." };
//     }
// }
// async function apiRecipesByTime(minutes) {
//     try {
//         const response = await fetch("/api/recipes_by_time", {
//             method: "POST",
//             headers: {
//                 "Content-Type": "application/json"
//             },
//             body: JSON.stringify({ minutes })
//         });
//         return await response.json();
//     } catch (err) {
//         console.error("Error calling recipes_by_time API:", err);
//         return { top_dishes: [], description: "Lỗi khi gọi API." };
//     }
// }

// // Gọi API
// async function apiRecipesByServings(servings) {
//     try {
//         const response = await fetch("/api/recipes_by_servings", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ servings })
//         });
//         return await response.json();
//     } catch (err) {
//         console.error("Error calling recipes_by_servings API:", err);
//         return { top_dishes: [], description: "Lỗi khi gọi API." };
//     }
// }

// async function apiRecipesByCategory(category) {
//     try {
//         const response = await fetch("/api/recipes_by_category", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ category })
//         });
//         return await response.json();
//     } catch (err) {
//         console.error("Error calling recipes_by_category API:", err);
//         return { top_dishes: [], description: "Lỗi khi gọi API." };
//     }
// }

// // =======================================
// // 4. HANDLERS
// // =======================================

// // ---- ingredient mode ----
// async function handleIngredientMode(message) {
//     const ingredientsArray = message
//         .split(",")
//         .map(i => i.trim())
//         .filter(i => i !== "");

//     addLoading("Đang tìm công thức...");

//     try {
//         const data = await apiSmartRecipes(ingredientsArray);
//         removeLastLoading();

//         if (data.top_dishes && data.top_dishes.length > 0) {
//             addMessage("Món tìm được: " + data.top_dishes.join(", "));
//         } else {
//             addMessage("Không tìm thấy món ăn phù hợp.");
//         }

//         if (data.description) {
//             addMessage(data.description);
//         }

//     } catch (err) {
//         console.error("Error handleIngredientMode:", err);
//         removeLastLoading();
//         addMessage("Có lỗi xảy ra khi tìm kiếm công thức.");
//     }
// }

// // ---- guide mode ----
// async function handleGuideMode(message) {
//     addLoading("Đang lấy công thức nấu ăn...");

//     try {
//         const data = await apiRecipeDetail(message);

        
//         removeLastLoading();

//         if (data.error) {
//             console.warn("Frontend received error:", data.error);
//             return addMessage(data.error);
//         }
//         // Debug: log ra console
//         console.log("DEBUG - frontend received:", data);
//         // Hiển thị món ăn
//         addMessage(`🍽️ <b>${data?.dish_name || "không xác định"}</b>`);


//         // Hiển thị nguyên liệu
//         if (data.ingredients && data.ingredients.length > 0) {
//             addMessage(
//                 "<b>Nguyên liệu:</b><br>" +
//                 data.ingredients.map(i => `- ${i}`).join("<br>")
//             );
//         }

//         // Hiển thị hướng dẫn mượt từ LLM
//         if (data.steps_smooth) {
//             addMessage(
//                 "<b>Cách làm tóm tắt:</b><br>" + data.steps_smooth
//             );
//         }

//         // Debug hướng dẫn gốc
//         if (data.instructions && data.instructions.length > 0) {
//             console.debug("Steps original:", data.instructions);
//         }

//     } catch (err) {
//         console.error("Error handleGuideMode:", err);
//         removeLastLoading();
//         addMessage("Lỗi khi lấy hướng dẫn món ăn.");
//     }
// }

// // Recipes by difficulty
// async function handleDifficultyMode(difficulty) {
//     addLoading(`Đang tìm món ăn độ khó '${difficulty}'...`);

//     const data = await apiRecipesByDifficulty(difficulty);
//     removeLastLoading();

//     console.log("DEBUG - frontend received:", data);

//     if (!data || !data.top_dishes.length) {
//         addMessage("Không tìm thấy món ăn phù hợp.");
//         return;
//     }

//     // Hiển thị top dishes
//     addMessage(`<b>Món ăn độ khó ${difficulty}:</b><br>` +
//         data.top_dishes.map(d => `- ${d}`).join("<br>")
//     );

//     // Hiển thị mô tả LLM
//     if (data.description) {
//         addMessage("<b>Mô tả hấp dẫn:</b><br>" + data.description);
//     }
// }

// // Recipes by time
// async function handleTimeMode(minutes) {
//     addLoading(`Đang tìm món ăn gần với ${minutes} phút...`);

//     const data = await apiRecipesByTime(minutes);
//     removeLastLoading();

//     console.log("DEBUG - frontend received:", data);

//     if (!data || !data.top_dishes.length) {
//         addMessage("Không tìm thấy món ăn phù hợp.");
//         return;
//     }

//     // Hiển thị top dishes
//     addMessage(`<b>Món ăn gần với ${minutes} phút:</b><br>` +
//         data.top_dishes.map(d => `- ${d}`).join("<br>")
//     );

//     // Hiển thị mô tả LLM
//     if (data.description) {
//         addMessage("<b>Mô tả hấp dẫn:</b><br>" + data.description);
//     }
// }

// // Recipes by servings
// async function handleServingsMode(servings) {
//     addLoading(`Đang tìm món ăn cho khoảng ${servings} người...`);

//     const data = await apiRecipesByServings(servings);
//     removeLastLoading();

//     console.log("DEBUG - frontend received:", data);

//     if (!data || !data.top_dishes.length) {
//         addMessage("Không tìm thấy món ăn phù hợp.");
//         return;
//     }

//     addMessage(`<b>Món ăn cho khoảng ${servings} người:</b><br>` +
//         data.top_dishes.map(d => `- ${d}`).join("<br>")
//     );

//     if (data.description) {
//         addMessage("<b>Mô tả hấp dẫn:</b><br>" + data.description);
//     }
// }

// async function handleCategoryMode(category) {
//     addLoading(`Đang tìm món ăn thể loại '${category}'...`);
//     const data = await apiRecipesByCategory(category);
//     removeLastLoading();

//     console.log("DEBUG - frontend received:", data);

//     if (!data || !data.top_dishes.length) {
//         addMessage("Không tìm thấy món ăn phù hợp.");
//         return;
//     }

//     addMessage(`<b>Món ăn thể loại ${category}:</b><br>` +
//         data.top_dishes.map(d => `- ${d}`).join("<br>")
//     );

//     if (data.description) {
//         addMessage("<b>Mô tả hấp dẫn:</b><br>" + data.description);
//     }
// }

// // =======================================
// // 5. MAIN LOGIC
// // =======================================

// // Chọn chế độ
// menuItems.forEach(item => {
//     item.addEventListener('click', () => {
//         menuItems.forEach(i => i.classList.remove('selected'));
//         item.classList.add('selected');

//         selectedMode = item.getAttribute('data-mode') || 'ingredient';
//         chatMessages.innerHTML = '';
//         addMessage(`Bạn đã chuyển sang chế độ: ${item.textContent.trim()}`);
//         chatInput.placeholder = placeholders[selectedMode] || "Nhập tin nhắn...";
//     });
// });

// // Gửi tin nhắn
// async function sendMessage() {
//     const message = chatInput.value.trim();
//     if (!message) return;

//     addMessage(message, true);
//     chatInput.value = "";

//     if (!selectedMode) {
//         return addMessage("Vui lòng chọn chế độ trước.");
//     }

//     switch (selectedMode) {
//         case "ingredient":
//             return handleIngredientMode(message);

//         case "guide":
//             return handleGuideMode(message);
//         case "difficulty":
//             return handleDifficultyMode(message);
//         case "time":
//             return handleTimeMode(message);
//         case "serving":
//             return handleServingsMode(message);
//         case "category":
//             return handleCategoryMode(message);
//         default:
//             return addMessage("Chế độ này đang phát triển.");
//     }
// }

// // Event listener
// sendButton.addEventListener("click", sendMessage);
// chatInput.addEventListener("keypress", e => {
//     if (e.key === "Enter") sendMessage();
// });

document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chatMessages");
    const chatInput = document.getElementById("chatInput");
    const sendButton = document.getElementById("sendButton");

    // -----------------------------
    // Add message to UI
    // -----------------------------
    function addMessage(text, isUser = false) {
        const msg = document.createElement("div");
        msg.className = `message ${isUser ? "user" : "ai"}`;
        msg.innerHTML = `<div class="message-bubble">${text}</div>`;
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Loading indicator
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

    // -----------------------------
    // Send message handler
    // -----------------------------
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
                body: JSON.stringify({ text })
            });

            const data = await response.json();
            removeLoading();
            console.log("Response:", data);

            // -----------------------------
            // Intent 1: suggest_dishes
            // -----------------------------
            if (data.intent === "suggest_dishes") {
                // Nếu không có món
                if (!data.top_dishes || data.top_dishes.length === 0) {
                    addMessage("Không tìm thấy món ăn phù hợp.");
                    return;
                }

                let html = "";
                html += `<b>🎯 Gợi ý món ăn phù hợp:</b><br>`;
                html += data.top_dishes.slice(0, 10).map(d => `• ${d}`).join("<br>");

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

                // ----------------------
                // Xử lý nguyên liệu
                // ----------------------
                html += `<b>🧂 Nguyên liệu:</b><br>`;

                if (Array.isArray(data.ingredients)) {
                    html += data.ingredients.map(i => `• ${i}`).join("<br>");
                } else {
                    html += "Không có dữ liệu nguyên liệu.";
                }

                // ----------------------
                // Xử lý steps_smooth: string hoặc list đều OK
                // ----------------------
                html += `<br><br><b>👨‍🍳 Các bước thực hiện:</b><br>`;

                let steps = [];

                if (Array.isArray(data.steps_smooth)) {
                    // Backend trả về dạng list
                    steps = data.steps_smooth;
                } else if (typeof data.steps_smooth === "string") {
                    // Backend trả về dạng string → split thành dòng
                    steps = data.steps_smooth.split("\n");
                } else {
                    html += "Không có hướng dẫn.";
                    addMessage(html);
                    return;
                }

                html += steps
                    .filter(s => s.trim().length > 0)
                    .map((step, idx) => `${idx + 1}. ${step.trim()}`)
                    .join("<br>");

                addMessage(html);
                return;
            }
            // -----------------------------
            // Fallback
            // -----------------------------
            else {
                addMessage(data.error || "Xin lỗi, tôi chưa hiểu yêu cầu của bạn.");
            }
        } catch (err) {
            removeLoading();
            console.error(err);
            addMessage("❌ Lỗi kết nối tới server.");
        }
    }

    // -----------------------------
    // Event listeners
    // -----------------------------
    sendButton.addEventListener("click", sendMessage);

    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
});
