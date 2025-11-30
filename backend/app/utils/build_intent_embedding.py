import json
import pickle
from sentence_transformers import SentenceTransformer

INPUT = "./data/intent_samples.json"
OUTPUT = "./data/intent_embeddings.pkl"

print("Loading model BGE-M3...")
model = SentenceTransformer("BAAI/bge-m3")

print("Loading intent samples...")
with open(INPUT, encoding="utf-8") as f:
    intent_samples = json.load(f)

print("Encoding samples...")
intent_embeddings = {
    intent: model.encode(samples, convert_to_numpy=True)
    for intent, samples in intent_samples.items()
}

print("Saving embeddings...")
with open(OUTPUT, "wb") as f:
    pickle.dump(intent_embeddings, f)

print("✅ Done! Saved →", OUTPUT)