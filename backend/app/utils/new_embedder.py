# new_embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm


def load_embedding_model(model_name="BAAI/bge-m3", device=None):
    """
    Load embedding model BGE-M3 (không dùng prefix).
    """
    print(f"🔹 Loading embedding model: {model_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name)
    model.to(device)

    print(f"👉 Using device: {device}")
    return model


def embed_texts(texts, model, batch_size=32):
    """
    Encode list text bằng BGE-M3 (không prefix).
    Output: numpy.float32 matrix
    """

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
