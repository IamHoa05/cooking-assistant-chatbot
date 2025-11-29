from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from pyvi import ViTokenizer
import numpy as np
import re
import string
import json
import pickle
import torch
# -------------------------
# Load dictionaries
# -------------------------
with open("app/utils/data/ingredient_dict.json", encoding="utf-8") as f:
    ingredients_list = json.load(f)
with open("app/utils/data/category_dict.json", encoding="utf-8") as f:
    category_list = json.load(f)
with open("app/utils/data/dish_dict.json", encoding="utf-8") as f:
    dishes_list = json.load(f)

ingredients_set = set([i.lower().replace(" ", "_") for i in ingredients_list])
dishes_set = [d.lower().replace(" ", "_") for d in dishes_list]

STOPWORDS = {
    "còn","miếng","cái","từ","trước","làm","món","gì","vừa","ngon",
    "dễ","thế","nào","nhé","ạ","tôi","muốn","nấu","cho","bị","thấy",
    "có","để","ăn","lại","thêm","nữa","nha","ôi","ơi","à","ra"
}
DIFFICULTY_KEYWORDS = ["dễ","trung bình","khó"]

# -------------------------
# Load embedding model
# -------------------------
MODEL_NAME = "BAAI/bge-m3"
model = SentenceTransformer(MODEL_NAME)

# Load prebuilt embeddings
with open("app/utils/data/intent_embeddings.pkl", "rb") as f:
    intent_embeddings = pickle.load(f)

# -------------------------
# Intent detection
# -------------------------
def get_rule_weight(text):
    t = text.lower()
    weights = {}

    # keywords cho cooking_guide
    cooking_keywords = [
        r"dạy", r"hướng dẫn", r"cách làm", r"công thức", r"bước", r"làm sao", r"cách nấu",
        r"học nấu", r"chỉ mình", r"bài học", r"chia sẻ cách", r"cách chế biến",
        r"thực hiện món", r"hướng dẫn chi tiết", r"làm món", r"hướng dẫn làm"
    ]
    if any(re.search(k, t) for k in cooking_keywords):
        weights["cooking_guide"] = 0.05

    # keywords cho suggest_dishes
    suggest_keywords = [
        r"nấu món", r"có .* nấu gì", r"gợi ý món", r"món gì", r"thịt", r"tôm",
        r"rau", r"cá", r"trứng", r"có nguyên liệu", r"tối nay nấu", r"nấu cho",
        r"nấu nhanh", r"món ngon", r"công thức món", r"nấu với", r"làm món gì",
        r"làm món nhanh", r"tìm món", r"món phù hợp", r"cần tìm"
    ]
    if any(re.search(k, t) for k in suggest_keywords):
        weights["suggest_dishes"] = 0.05

    return weights

def detect_intent(text: str):
    # Encode đoạn text → numpy
    text_emb = model.encode([text])[0]

    # Convert sang tensor để tính cosine
    text_emb = torch.tensor(text_emb, dtype=torch.float32)

    scores = {}
    best_intent = None
    best_score = -1

    for intent, emb_list in intent_embeddings.items():  # <-- sửa tên biến

        # emb_list có thể là numpy → convert
        if not torch.is_tensor(emb_list):
            emb_list = torch.tensor(emb_list, dtype=torch.float32)

        # cosine similarity giữa text_emb và list embeddings của intent đó
        sim = torch.nn.functional.cosine_similarity(
            text_emb.unsqueeze(0),  # [1, 384]
            emb_list,               # [N, 384]
            dim=1
        ).max().item()

        scores[intent] = sim

        if sim > best_score:
            best_score = sim
            best_intent = intent

    # Trả ra:
    #   - intent tốt nhất
    #   - score tương ứng của intent đó
    #   - toàn bộ bảng scores
    return best_intent, scores[best_intent], scores

# -------------------------
# Slot extraction (GIỮ NGUYÊN TOÀN BỘ)
# -------------------------
def tokenize(text):
    tokens = ViTokenizer.tokenize(text).split()
    return [t.lower() for t in tokens if t not in STOPWORDS and t not in string.punctuation]

def generate_ngrams(tokens, max_n=6):
    ngrams = []
    for n in range(max_n, 0, -1):
        for i in range(len(tokens)-n+1):
            ngrams.append((tokens[i:i+n], i, i+n))
    return ngrams

def extract_ingredients(text):
    tokens = tokenize(text)
    ngrams = generate_ngrams(tokens)
    detected = set()
    used_idx = set()

    for ng, s, e in ngrams:
        key = "_".join(ng)
        if key in ingredients_set and not any(i in used_idx for i in range(s, e)):
            detected.add(key.replace("_", " "))
            used_idx.update(range(s, e))

    return list(detected)

def extract_time(text):
    text = text.lower()
    match_hour_min = re.search(r'(\d+)\s*(giờ|h|tiếng)\s*(\d+)?\s*(phút|p)?', text)
    if match_hour_min:
        hours = int(match_hour_min.group(1))
        mins = int(match_hour_min.group(3)) if match_hour_min.group(3) else 0
        return [hours*60 + mins]

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
    t = " ".join(text.lower().split())
    sorted_cats = sorted(category_list, key=lambda x: -len(x))
    for cat in sorted_cats:
        if f"món {cat}" in t or cat in t:
            return cat
    return None

def extract_dish_name(text, dishes_set=dishes_set, threshold=60):
    text_lower = text.lower()
    tokens = tokenize(text_lower)
    ngrams = generate_ngrams(tokens)

    best = None
    best_score = 0

    for ng, s, e in ngrams:
        joined = "_".join(ng)
        for dish in dishes_set:
            score = fuzz.ratio(joined, dish)
            if score >= threshold and score > best_score:
                best = dish
                best_score = score

    return best.replace("_", " ") if best else None

def extract_all_slots(text, intent=None):
    """
    Extract slots from text based on intent.
    - suggest_dishes: category, ingredient, time, difficulty, serving
    - cooking_guide: dish_name
    """
    slots = {}

    if intent == "suggest_dishes":
        # Category
        category = extract_category(text)
        if category:
            slots["category"] = category

        # Ingredients
        ingredients = extract_ingredients(text)
        if category and category in ingredients:
            ingredients.remove(category)
        if ingredients:
            slots["ingredient"] = ingredients

        # Time
        time = extract_time(text)
        if time:
            slots["time"] = time

        # Difficulty
        difficulty = extract_difficulty(text)
        if difficulty:
            slots["difficulty"] = difficulty

        # Serving
        serving = extract_serving(text)
        if serving:
            slots["serving"] = serving

    elif intent == "cooking_guide":
        # Dish name
        dish_name = extract_dish_name(text, dishes_set=dishes_set)
        if dish_name:
            slots["dish_name"] = dish_name

    else:
        # fallback: try all
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

        dish_name = extract_dish_name(text, dishes_set=dishes_set)
        if dish_name:
            slots["dish_name"] = dish_name

    return slots

def format_output_by_intent(intent, slots):
    if intent == "suggest_dishes":
        return {
            "ingredients": slots.get("ingredient"),
            "difficulty": slots.get("difficulty"),
            "time": slots.get("time"),
            "servings": slots.get("serving"),
            "category": slots.get("category")
        }

    if intent == "cooking_guide":
        return {
            "dish_name": slots.get("dish_name")
        }

    return slots
