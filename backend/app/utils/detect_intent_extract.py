"""
Detect Intent + Extract Slots
Refactor toàn bộ code cho sạch, rõ, dễ mở rộng.
"""

import re
import json
import pickle
import torch
import numpy as np
import string
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer

# ============================================
# 1. LOAD DATA
# ============================================

def load_json(path):
    """Load JSON file với UTF-8."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Danh sách ingredients, categories, dishes
ingredients_list = load_json("app/utils/data/ingredient_dict.json")
category_list = load_json("app/utils/data/category_dict.json")
dishes_list = load_json("app/utils/data/dish_dict.json")

# Chuẩn hoá dữ liệu về dạng set, lowercase, token dạng "ga_kho"
ingredients_set = set(i.lower().replace(" ", "_") for i in ingredients_list)
dishes_set = [d.lower().replace(" ", "_") for d in dishes_list]

# Stopwords loại bỏ khỏi token
STOPWORDS = {
    "còn","miếng","cái","từ","trước","làm","món","gì","vừa","ngon",
    "dễ","thế","nào","nhé","ạ","tôi","muốn","nấu","cho","bị","thấy",
    "có","để","ăn","lại","thêm","nữa","nha","ôi","ơi","à","ra"
}

DIFFICULTY_KEYWORDS = ["dễ", "trung bình", "khó"]


# ============================================
# 2. LOAD MODEL + INTENT EMBEDDINGS
# ============================================

MODEL_NAME = "BAAI/bge-m3"
model = SentenceTransformer(MODEL_NAME)

# intent_embeddings: dict[intent] = np.array([...])
with open("app/utils/data/intent_embeddings.pkl", "rb") as f:
    intent_embeddings = pickle.load(f)


# ============================================
# 3. TOKENIZE + NGRAMS
# ============================================

def tokenize(text: str):
    """Tokenize tiếng Việt + remove stopwords + punctuation."""
    tokens = ViTokenizer.tokenize(text).split()
    return [
        t.lower() for t in tokens
        if t not in STOPWORDS and t not in string.punctuation
    ]


def generate_ngrams(tokens, max_n=6):
    """Sinh n-grams từ token list, ưu tiên n lớn trước."""
    for n in range(max_n, 0, -1):
        for i in range(len(tokens) - n + 1):
            yield tokens[i:i+n], i, i+n


# ============================================
# 4. RULE-BASED INTENT BOOST
# ============================================

COOKING_KEYWORDS = [
    r"dạy", r"hướng dẫn", r"cách làm", r"công thức", r"bước", r"làm sao",
    r"cách nấu", r"học nấu", r"chỉ mình", r"chia sẻ cách", r"cách chế biến",
    r"hướng dẫn chi tiết", r"hướng dẫn làm"
]

SUGGEST_KEYWORDS = [
    r"nấu món", r"có .* nấu gì", r"gợi ý món", r"món gì",
    r"có nguyên liệu", r"tối nay nấu", r"nấu với", r"làm món gì",
    r"tìm món", r"món phù hợp", r"cần tìm"
]


def get_rule_weight(text):
    """
    Kiểm tra rule trong câu → cộng điểm phụ (boost) vào intent.
    """
    t = text.lower()
    weights = {}

    if any(re.search(k, t) for k in COOKING_KEYWORDS):
        weights["cooking_guide"] = 0.05

    if any(re.search(k, t) for k in SUGGEST_KEYWORDS):
        weights["suggest_dishes"] = 0.05

    return weights


# ============================================
# 5. INTENT DETECTION
# ============================================

def detect_intent(text: str, alpha=0.1, top_k=5):
    """
    Hybrid Intent Detection:
    - Dùng embedding để tính cosine similarity
    - Lấy top-k điểm similarity trung bình
    - Kết hợp rule-based boosting (alpha * weight)
    """
    text_emb = torch.tensor(model.encode([text])[0], dtype=torch.float32)
    rule_weights = get_rule_weight(text)

    best_intent = None
    best_score = -999
    scores = {}

    for intent, emb_list in intent_embeddings.items():
        emb_tensor = torch.tensor(emb_list, dtype=torch.float32)

        # cosine giữa input và all samples của intent
        sims = torch.nn.functional.cosine_similarity(
            text_emb.unsqueeze(0),
            emb_tensor,
            dim=1
        )

        # mean top-k similarity
        topk_mean = sims.topk(min(top_k, len(sims))).values.mean().item()

        # apply rule weight
        final_score = topk_mean + alpha * rule_weights.get(intent, 0.0)
        scores[intent] = final_score

        if final_score > best_score:
            best_intent = intent
            best_score = final_score

    return best_intent, best_score, scores


# ============================================
# 6. SLOT EXTRACTION
# ============================================

def extract_ingredients(text):
    """Trích xuất ingredient bằng n-grams + dictionary match."""
    tokens = tokenize(text)
    detected = set()
    used = set()

    for ng, s, e in generate_ngrams(tokens):
        key = "_".join(ng)
        if key in ingredients_set and not any(i in used for i in range(s, e)):
            detected.add(key.replace("_", " "))
            used.update(range(s, e))

    return list(detected)


def extract_time(text):
    """Trích xuất thời gian: phút, giờ."""
    t = text.lower()

    # Case: X giờ Y phút
    match_hour_min = re.search(r'(\d+)\s*(giờ|h|tiếng)\s*(\d+)?\s*(phút|p)?', t)
    if match_hour_min:
        hours = int(match_hour_min.group(1))
        mins = int(match_hour_min.group(3)) if match_hour_min.group(3) else 0
        return [hours * 60 + mins]

    # Case: chỉ có phút
    matches = re.findall(r'(\d+)\s*(phút|p)', t)
    if matches:
        return [sum(int(m[0]) for m in matches)]

    return None


def extract_serving(text):
    """Trích xuất số người ăn."""
    matches = re.findall(r'(\d+)\s*(người|phần)', text.lower())
    return [int(m[0]) for m in matches] if matches else None


def extract_difficulty(text):
    """Dễ – Trung bình – Khó."""
    t = text.lower()
    for d in DIFFICULTY_KEYWORDS:
        if d in t:
            return d
    return None


def extract_category(text):
    """
    Match category theo danh sách category_dict.
    Ưu tiên category dài hơn trước.
    """
    t = text.lower()
    sorted_cats = sorted(category_list, key=lambda x: -len(x))

    for cat in sorted_cats:
        c = cat.lower()

        # ưu tiên format "món + category"
        if re.search(rf"\bmón\s+{re.escape(c)}\b", t):
            return cat

        # match trực tiếp
        if re.search(rf"\b{re.escape(c)}\b", t):
            return cat

    return None


def extract_dish_name(text, threshold=60):
    tokens = tokenize(text.lower())
    best_dish = None
    best_score = 0

    for ng, _, _ in generate_ngrams(tokens, max_n=6):
        s = "_".join(ng)
        for dish in dishes_set:
            score = fuzz.ratio(s, dish)
            # Nếu trùng score, ưu tiên chứa nhiều token trùng với input
            if score > best_score and score >= threshold:
                best_dish = dish
                best_score = score

    # Nếu best_dish tồn tại, convert back
    if best_dish:
        return best_dish.replace("_", " ")

    # Fallback: kết hợp rule-based đơn giản
    ingredients = extract_ingredients(text)
    for dish in dishes_list:
        dish_lower = dish.lower()
        if all(ing.lower() in dish_lower for ing in ingredients):
            return dish

    return None


# ============================================
# 7. SLOT EXTRACTOR THEO INTENT
# ============================================

def extract_all_slots(text, intent=None):
    slots = {}

    # ---------- Intent Suggest Dishes ----------
    if intent == "suggest_dishes":
        if (cat := extract_category(text)):
            slots["category"] = cat

        if (ing := extract_ingredients(text)):
            if cat in ing:   # tránh nhầm category thành ingredient
                ing.remove(cat)
            if ing:
                slots["ingredient"] = ing

        if (t := extract_time(text)):
            slots["time"] = t

        if (diff := extract_difficulty(text)):
            slots["difficulty"] = diff

        if (serv := extract_serving(text)):
            slots["serving"] = serv

    # ---------- Intent Cooking Guide ----------
    elif intent == "cooking_guide":
        if (dish := extract_dish_name(text)):
            slots["dish_name"] = dish

    # ---------- Fallback: extract hết ----------
    else:
        for extractor, key in [
            (extract_category, "category"),
            (extract_ingredients, "ingredient"),
            (extract_time, "time"),
            (extract_difficulty, "difficulty"),
            (extract_serving, "serving"),
            (extract_dish_name, "dish_name")
        ]:
            if (v := extractor(text)):
                slots[key] = v

    return slots


# ============================================
# 8. FORMAT OUTPUT
# ============================================

def format_output_by_intent(intent, slots):
    """Chuẩn hoá output theo intent."""
    if intent == "suggest_dishes":
        return {
            "ingredients": slots.get("ingredient"),
            "difficulty": slots.get("difficulty"),
            "time": slots.get("time"),
            "servings": slots.get("serving"),
            "category": slots.get("category"),
        }

    if intent == "cooking_guide":
        return {"dish_name": slots.get("dish_name")}

    return slots
