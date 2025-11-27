from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from pyvi import ViTokenizer
import numpy as np
import re
import string
import json

# -------------------------
# Load dữ liệu
# -------------------------
with open("app/utils/data/ingredient_dict.json", encoding="utf-8") as f:
    ingredients_list = json.load(f)
with open("app/utils/data/category_dict.json", encoding="utf-8") as f:
    category_list = json.load(f)
with open("app/utils/data/dish_dict.json", encoding="utf-8") as f:
    dishes_list = json.load(f)

ingredients_set = set([i.lower().replace(" ", "_") for i in ingredients_list])
dishes_set = [d.lower().replace(" ", "_") for d in dishes_list]

STOPWORDS = {"còn","miếng","cái","từ","trước","làm","món","gì","vừa","ngon","dễ","thế","nào","nhé","ạ","tôi","muốn","nấu"}
DIFFICULTY_KEYWORDS = ["dễ","trung bình","khó"]

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# -------------------------
# Intent samples
# -------------------------
intent_samples = {
    "suggest_dishes": [
        "Tôi có thịt gà, khoai tây, cà rốt, làm món gì ngon?",
        "Nhà còn tôm, bông cải xanh, giúp tôi gợi ý món ăn",
        "Còn cá hồi, khoai lang, bông cải, tối nay nấu gì?",
        "Giúp mình với, còn thịt heo khoai môn bắp non, tối nay nấu gì?",
        "Tối nay tôi chỉ có thịt gà, sả, ớt, một ít thịt bò, nên nấu gì?",
        "Bạn gợi ý món nào nấu từ thịt gà, sả, ớt, thịt bò?",
        "Tôi muốn nấu món từ thịt gà, sả, ớt, thịt bò?",
        "Tôi muốn nấu món salad với cà chua bi, dưa leo, 4 phần ăn.",
        "giúp tôi làm món từ tôm , thịt bò, rau muống cho 3 người",
        "Tôi có thịt heo, cà rốt, khoai tây, làm món gì nhanh cho 2 người?",
        "tìm món đơn giản cho người mới bắt đầu",
        "tìm món nấu nhanh trong 30p",
    ],
    "cooking_guide": [
        "Hướng dẫn nấu món canh chua",
        "Công thức làm bánh mì chiên",
        "Làm sao để nấu món cà ri gà ngon?",
        "Bạn chỉ mình cách nấu canh mướp với tôm",
        "Mình muốn học nấu mì Ý bò bằm",
        "Hướng dẫn cách làm gà kho tộ",
        "Bạn có công thức nấu phở gà không?",
        "Làm món tráng miệng socola, hướng dẫn nhé",
        "Mình muốn học nấu canh rau củ nhanh",
        "Công thức nấu bánh pizza tại nhà",
        "Bạn hướng dẫn cách làm món xào thịt bò",
        "Mình muốn học nấu cháo hải sản",
        "Cách nấu món cà tím nhồi thịt",
        "Hướng dẫn làm món tôm rang muối",
        "Công thức nấu canh bí đỏ cho trẻ",
        "Làm món salad trộn, hướng dẫn chi tiết",
        "Bạn chỉ cách nấu gà sốt tiêu đen"
    ]
}

intent_embeddings = {k: model.encode(v, convert_to_tensor=True) for k,v in intent_samples.items()}

# -------------------------
# Intent detection
# -------------------------
def get_rule_weight(text):
    t = text.lower()
    weights = {}
    if re.search(r"(dạy tôi làm|hướng dẫn|cách làm|bước|chia sẻ|công thức|giới thiệu|cách nấu)", t):
        weights["cooking_guide"] = 0.05
    if re.search(r"(nấu món|có .* nấu gì|làm món|gợi ý món|thịt|cá|rau|trứng|tôm|cà rốt)", t):
        weights["suggest_dishes"] = 0.05
    return weights

def detect_intent(text):
    user_emb = model.encode([text], convert_to_tensor=True)[0]
    all_scores = {}
    for intent_name, embeddings in intent_embeddings.items():
        sim_scores = cosine_similarity(user_emb.reshape(1,-1), embeddings.cpu().numpy())[0]
        all_scores[intent_name] = float(np.max(sim_scores))
    for k,w in get_rule_weight(text).items():
        all_scores[k] += w
    final_intent = max(all_scores, key=all_scores.get)
    final_score = all_scores[final_intent]
    return final_intent, final_score, all_scores

# -------------------------
# Slot extraction
# -------------------------
def tokenize(text):
    tokens = ViTokenizer.tokenize(text).split()
    return [t.lower() for t in tokens if t not in STOPWORDS and t not in string.punctuation]

def generate_ngrams(tokens, max_n=6):
    ngrams = []
    for n in range(max_n,0,-1):
        for i in range(len(tokens)-n+1):
            ngrams.append((tokens[i:i+n], i, i+n))
    return ngrams

def extract_ingredients(text):
    tokens = tokenize(text)
    ngrams = generate_ngrams(tokens)
    detected = set()
    used_indices = set()
    for ng, start, end in ngrams:
        joined = "_".join(ng)
        if joined in ingredients_set and not any(i in used_indices for i in range(start,end)):
            detected.add(joined.replace("_"," "))
            used_indices.update(range(start,end))
    return list(detected)

def extract_time(text):
    text = text.lower()
    match_hour_min = re.search(r'(\d+)\s*(giờ|h|tiếng)\s*(\d+)?\s*(phút|p)?', text)
    if match_hour_min:
        hours = int(match_hour_min.group(1))
        minutes = int(match_hour_min.group(3)) if match_hour_min.group(3) else 0
        return [hours*60+minutes]
    match_only_min = re.findall(r'(\d+)\s*(phút|p)', text)
    if match_only_min:
        return [sum(int(m[0]) for m in match_only_min)]
    return None

def extract_serving(text):
    matches = re.findall(r'(\d+)\s*(người|phần)', text.lower())
    return [int(m[0]) for m in matches] if matches else None

def extract_difficulty(text):
    for kw in DIFFICULTY_KEYWORDS:
        if kw in text.lower():
            return kw
    return None

def extract_category(text):
    text_lower = " ".join(text.lower().split())
    sorted_keywords = sorted(category_list, key=lambda x:-len(x))
    for cat in sorted_keywords:
        if f"món {cat}" in text_lower or cat in text_lower:
            return cat
    return None

# def extract_dish_name(text, category=None, threshold=60, dishes_set=None):
#     """
#     Kết hợp category detection + fuzzy match danh sách dishes
#     Giống logic extract_best_dish
#     """
#     if dishes_set is None:
#         raise ValueError("Bạn phải truyền dishes_set vào")

#     text_lower = text.lower().strip()

#     # 1️⃣ Exact match với category trước (fallback)
#     if category and category.lower() in text_lower:
#         return category.lower()

#     # 2️⃣ Tokenize + n-grams + fuzzy match
#     tokens = tokenize(text_lower)
#     ngrams = generate_ngrams(tokens)
#     best_match = None
#     best_score = 0

#     for ng, start, end in ngrams:
#         joined = "_".join(ng)
#         for dish in dishes_set:
#             score = fuzz.partial_ratio(joined, dish)
#             if score > best_score and score >= threshold:
#                 best_score = score
#                 best_match = dish

#     if best_match:
#         return best_match.replace("_", " ")

#     return None

def extract_dish_name(text, category=None, threshold=60, dishes_set=None):
    """
    Extract tên món từ text bằng fuzzy match, theo logic giống extract_best_dish.
    - threshold: chỉ nhận match score >= threshold
    """
    if dishes_set is None:
        raise ValueError("Bạn phải truyền dishes_set vào")

    text_lower = text.lower().strip()
    tokens = tokenize(text_lower)
    ngrams = generate_ngrams(tokens)  # tạo tất cả n-gram từ 1 token tới length

    best_match = None
    best_score = 0

    for ng, start, end in ngrams:
        joined = "_".join(ng)
        for dish in dishes_set:
            score = fuzz.partial_ratio(joined, dish)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = dish
                # debug
                print(f"[DEBUG] New best match: {best_match.replace('_',' ')} with score {best_score}")

    if best_match:
        return best_match.replace("_", " ")

    return None
def extract_all_slots(text):
    slots = {}
    category = extract_category(text)
    if category:
        slots["category"] = category

    ingredients = extract_ingredients(text)
    if category and category in ingredients:
        ingredients.remove(category)
    if ingredients:
        slots["ingredient"] = ingredients

    time = extract_time(text)
    if time:
        slots["time"] = time

    difficulty = extract_difficulty(text)
    if difficulty:
        slots["difficulty"] = difficulty

    serving = extract_serving(text)
    if serving:
        slots["serving"] = serving

    dish_name = extract_dish_name(text, category, dishes_set=dishes_set)
    if dish_name:
        slots["dish_name"] = dish_name

    return slots

# -------------------------
# Format output theo intent
# -------------------------
def format_output_by_intent(intent, slots):
    if intent == "suggest_dishes":
        return {
            "ingredients": slots.get("ingredient"),
            "difficulty": slots.get("difficulty"),
            "time": slots.get("time"),
            "servings": slots.get("serving"),
            "category": slots.get("category")
        }
    elif intent == "cooking_guide":
        return {
            "dish_name": slots.get("dish_name")
        }
    else:
        return slots

# -------------------------
# Demo interactive
# -------------------------
if __name__ == "__main__":
    print("Nhập câu để detect intent + extract slots (gõ 'exit' để thoát):")
    while True:
        text = input("Bạn: ")
        if text.lower() in ["exit","thoát"]:
            break
        intent, score, _ = detect_intent(text)
        slots = extract_all_slots(text)
        output = format_output_by_intent(intent, slots)
        print(f"Intent: {intent}, Score: {score:.3f}")
        print("Output JSON:", json.dumps(output, ensure_ascii=False, indent=2))