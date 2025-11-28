# new_embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
import yaml
import os

# ====== Load config ======
CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../config.yml")
)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EMBEDDING_CONFIG = config.get("embedding", {})
MODEL_NAME = EMBEDDING_CONFIG["model_name"]     
BATCH_SIZE = EMBEDDING_CONFIG["batch_size"]     


def load_embedding_model(model_name=None, device=None):
    """
    Load embedding model BGE-M3 (không dùng prefix).
    Nếu model_name không truyền, lấy từ config.
    """
    if model_name is None:
        model_name = MODEL_NAME

    print(f"🔹 Loading embedding model: {model_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name)
    model.to(device)

    print(f"👉 Using device: {device}")
    return model


def embed_texts(texts, model, batch_size=None):
    """
    Encode list text bằng BGE-M3 (không prefix).
    Output: numpy.float32 matrix
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    if isinstance(texts, str):
        texts = [texts]

    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        batch_vec = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        vectors.extend(batch_vec)

    return np.array(vectors, dtype="float32")
