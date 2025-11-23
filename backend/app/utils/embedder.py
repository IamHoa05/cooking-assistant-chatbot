# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import pandas as pd
# import torch


# def load_vietnamese_embedding_model(device: str = None):
#     """
#     Load embedding model with GPU support if available.
#     """
#     model_name = "paraphrase-multilingual-MiniLM-L12-v2"
#     print(f"🔹 Loading model: {model_name}")

#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = SentenceTransformer(model_name, device=device)
#     print(f"👉 Using device: {device}")

#     return model


# def embed_texts(texts, model: SentenceTransformer, batch_size=64):
#     """
#     Encode texts efficiently with GPU + normalization.
#     """
#     return model.encode(
#         texts,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_tensor=False,  # để lưu pickle dễ
#         normalize_embeddings=True # embedding chuẩn hóa → tốt cho kNN / cosine
#     )


# def generate_vietnamese_recipe_embeddings(df: pd.DataFrame):

#         model = load_vietnamese_embedding_model()
        
#         columns_to_embed = {
#             "ingredient_names": "ingredient_names_embedding",
#             "ingredient_quantities": "ingredient_quantities_embedding",
#             "dish_name": "dish_name_embedding"
#         }

#         for text_col, out_col in columns_to_embed.items():
#             print(f"Embedding column: {text_col}")
#             texts = df[text_col].fillna("").astype(str).tolist()

#             vectors = embed_texts(texts, model, batch_size=32)  # numpy array (n, 384)

#             # Fix chính xác:
#             vectors_list = [v.tolist() for v in vectors]  # mỗi row là list 384 float
#             df[out_col] = pd.Series(vectors_list, dtype=object)

#         return df



# if __name__ == "__main__":
#     df = pd.read_csv("./data/test_processed_recipes.csv")

#     df_emb = generate_vietnamese_recipe_embeddings(df)

#     output = "./data/recipes_embeddings.pkl"
#     df_emb.to_pickle(output)
#     print(f"✅ Saved embeddings to {output}")


from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import torch
from typing import List, Union


def load_vietnamese_embedding_model(device: str = None):
    """
    Load BGE embedding model with GPU support.
    """
    # 🔥 Model BGE cho tiếng Việt (nên dùng)
    model_name = "BAAI/bge-base-en-v1.5"   # hoặc "dinhnguyenhv/bge-vi-base" nếu bạn có bản VI-FT
    print(f"🔹 Loading BGE model: {model_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    print(f"👉 Using device: {device}")

    return model


# def embed_texts(texts, model: SentenceTransformer, batch_size=64):
#     """
#     Encode texts using BGE + normalization.
#     LƯU Ý: BGE khuyến nghị dùng normalize_embeddings=True cho cosine + FAISS.
#     """
#     # Với BGE, để kết quả đúng, phải thêm prefix "query: " hoặc "passage: " (tùy task)
#     # Nhưng ở đây bạn đang encode dữ liệu (passages) → dùng prefix "passage: "
#     prefixed = [f"passage: {t}" for t in texts]

#     return model.encode(
#         prefixed,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_tensor=False,
#         normalize_embeddings=True
#     )

def embed_text(text: str, model: SentenceTransformer, text_type: str = "passage") -> List[float]:
    """
    Tạo embedding cho một chuỗi duy nhất sử dụng mô hình BGE.
    Tự động thêm prefix ('passage' hoặc 'query') và chuẩn hóa embedding.
    
    Args:
        text: Chuỗi cần embed.
        model: Mô hình SentenceTransformer (BGE).
        text_type: Loại text, 'passage' (mặc định) hoặc 'query'.
    
    Returns:
        List[float]: Vector embedding đã chuẩn hóa.
    """
    # Thêm prefix theo loại text
    prefix = f"{text_type}: "
    
    # Encode text, normalize embedding để dùng cho cosine similarity hoặc FAISS
    embedding = model.encode(
        [prefix + text], 
        show_progress_bar=False, 
        normalize_embeddings=True
    )
    
    # Trả về vector dạng list
    return embedding[0].tolist()


def embed_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 32, text_type: str = "passage") -> List[List[float]]:
    """
    Tạo embedding cho danh sách nhiều chuỗi sử dụng mô hình BGE theo batch.
    Tự động thêm prefix ('passage' hoặc 'query') và chuẩn hóa embedding.
    
    Args:
        texts: Danh sách các chuỗi cần embed.
        model: Mô hình SentenceTransformer (BGE).
        batch_size: Số lượng text mỗi batch (mặc định 32).
        text_type: Loại text, 'passage' (mặc định) hoặc 'query'.
    
    Returns:
        List[List[float]]: Danh sách các vector embedding đã chuẩn hóa.
    """
    # Thêm prefix cho tất cả các text
    prefix = f"{text_type}: "
    prefixed_texts = [prefix + t for t in texts]
    
    embeddings = []
    
    # Encode theo từng batch
    for i in tqdm(range(0, len(prefixed_texts), batch_size)):
        batch = prefixed_texts[i:i + batch_size]
        
        # Encode batch, normalize embeddings
        batch_embeddings = model.encode(
            batch, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        
        # Lưu kết quả
        embeddings.extend(batch_embeddings)
    
    # Chuyển sang list of list
    return [emb.tolist() for emb in embeddings]

def generate_vietnamese_recipe_embeddings(df: pd.DataFrame):
    model = load_vietnamese_embedding_model()
    
    columns_to_embed = {
        "ingredient_names": "ingredient_names_embedding",
        "ingredient_quantities": "ingredient_quantities_embedding",
        "dish_name": "dish_name_embedding"
    }

    for text_col, out_col in columns_to_embed.items():
        print(f"Embedding column: {text_col}")
        texts = df[text_col].fillna("").astype(str).tolist()

        # Gọi embed_texts 1 lần, trả về list of list
        vectors = embed_texts(texts, model, batch_size=32)  # List[List[float]]

        # Trực tiếp lưu list vào DataFrame, không cần .tolist()
        df[out_col] = pd.Series(vectors, dtype=object)

    return df



if __name__ == "__main__":
    df = pd.read_csv("./data/test_501_1000_recipes.csv")

    df_emb = generate_vietnamese_recipe_embeddings(df)

    output = "./data/recipes_embeddings.pkl"
    df_emb.to_pickle(output)
    print(f"✅ Saved embeddings to {output}")
