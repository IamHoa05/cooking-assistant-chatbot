# faiss_handler.py
import os
import faiss
import numpy as np

class FAISSHandler:
    def __init__(self, df, embedding_columns, index_dir="./faiss_indexes"):
        """
        Class để load FAISS index từ file và search.
        
        Args:
            df: DataFrame chứa dữ liệu (metadata)
            embedding_columns: dict {key: column_name_in_df}
            index_dir: thư mục chứa index đã build
        """
        self.df = df
        self.embedding_columns = embedding_columns
        self.index_dir = index_dir
        self.indexes = {}

        # Load tất cả index từ file
        self._load_indexes()

    def _load_indexes(self):
        for key, col in self.embedding_columns.items():
            index_path = os.path.join(self.index_dir, f"{col}.index")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            self.indexes[col] = faiss.read_index(index_path)
            print(f"✅ Loaded FAISS index for column '{col}' from '{index_path}'")

    def search(self, query_vector: np.ndarray, column_key: str, top_k: int = 5):
        """
        Tìm top_k kết quả gần nhất cho query_vector trên cột embedding column_key.
        Không dùng cosine similarity.
        
        Args:
            query_vector: np.ndarray (1D) vector query
            column_key: key trong embedding_columns
            top_k: số lượng kết quả trả về
        Returns:
            List of dict (row metadata + distance)
        """
        if column_key not in self.embedding_columns:
            raise ValueError(f"Invalid column_key '{column_key}'")

        index = self.indexes[self.embedding_columns[column_key]]
        query_vector = np.array([query_vector]).astype("float32")

        # Search top_k
        distances, indices = index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = self.df.iloc[idx].to_dict()
            row["_distance"] = dist
            results.append(row)
        return results
