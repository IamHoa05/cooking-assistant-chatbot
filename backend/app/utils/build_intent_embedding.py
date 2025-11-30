import json
import pickle
from sentence_transformers import SentenceTransformer

# ============================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================

INPUT = "./data/intent_samples.json"        # File chứa các câu mẫu intent
OUTPUT = "./data/intent_embeddings.pkl"     # File sẽ lưu embedding của intent

# ============================================
# 2. LOAD MÔ HÌNH BGE-M3
# ============================================

print("🔹 Loading model BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

# ============================================
# 3. ĐỌC FILE intent_samples.json
# ============================================

print("🔹 Loading intent samples...")
with open(INPUT, encoding="utf-8") as f:
    intent_samples = json.load(f)

# intent_samples có dạng:
# {
#     "cooking_guide": ["Cách nấu canh chua", "Chỉ tôi cách làm món bò kho", ...],
#     "ingredient_query": ["Rau này mua ở đâu", ...]
# }

# ============================================
# 4. TẠO EMBEDDING CHO MỖI INTENT
# ============================================

print("🔹 Encoding samples...")
intent_embeddings = {
    intent: model.encode(samples, convert_to_numpy=True)
    for intent, samples in intent_samples.items()
}

# intent_embeddings sẽ có dạng:
# {
#     "cooking_guide": np.array([...]),
#     "ingredient_query": np.array([...]),
# }

# ============================================
# 5. LƯU EMBEDDINGS RA FILE .pkl
# ============================================

print("🔹 Saving embeddings...")
with open(OUTPUT, "wb") as f:
    pickle.dump(intent_embeddings, f)

print("✅ Done! Saved →", OUTPUT)
