# new_embedder.py
"""
Module hỗ trợ load model embedding (BGE-M3) và encode văn bản.
"""

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============================================
# 1. LOAD CONFIG
# ============================================

CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../config.yml")
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EMBEDDING_CONFIG = config.get("embedding", {})
MODEL_NAME = EMBEDDING_CONFIG.get("model_name", "BAAI/bge-m3")  # mặc định
BATCH_SIZE = EMBEDDING_CONFIG.get("batch_size", 32)             # mặc định


# ============================================
# 2. LOAD EMBEDDING MODEL
# ============================================

def load_embedding_model(model_name=None, device=None):
    """
    Load embedding model SentenceTransformer (BGE-M3).

    Args:
        model_name (str): tên model, nếu None dùng config.
        device (str): 'cpu' hoặc 'cuda', nếu None tự detect.

    Returns:
        model: SentenceTransformer đã load và chuyển device.
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


# ============================================
# 3. EMBED TEXTS
# ============================================

def embed_texts(texts, model, batch_size=None, text_type=None):
    """
    Encode list text thành vector embedding.

    Args:
        texts (str | list[str]): văn bản hoặc list văn bản.
        model: SentenceTransformer đã load.
        batch_size (int): kích thước batch khi encode, default từ config.
        text_type: placeholder, để mở rộng nếu cần xử lý loại text khác.

    Returns:
        np.ndarray: ma trận embedding (float32), shape=(len(texts), embedding_dim)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    if isinstance(texts, str):
        texts = [texts]

    vectors = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i:i + batch_size]

        batch_vec = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalize embeddings
        )

        vectors.extend(batch_vec)

    return np.array(vectors, dtype="float32")
