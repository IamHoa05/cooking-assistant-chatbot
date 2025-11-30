"""
FAISSHandler
-------------
Module quản lý FAISS index:
- Load index từ file
- Tìm kiếm embeddings với top-k
- Hỗ trợ cả dish_name và ingredient_names (flattened)
"""

import os
import faiss
import numpy as np


class FAISSHandler:
    def __init__(self, df, index_dir="./faiss_indexes"):
        """
        Args:
            df (pd.DataFrame): dataframe chứa embeddings và dữ liệu gốc
            index_dir (str): thư mục chứa FAISS index và row_id mapping
        """
        self.df = df
        self.index_dir = index_dir

        # mapping index name → column trong df
        self.cols = {
            "dish": "dish_name_embedding",
            "names": "ingredient_names_embedding"
        }

        self.indexes = {}  # lưu các FAISS index
        self.row_map = {}  # lưu row mapping cho ingredient flattened

        self._load_all()

    # =========================================
    # Load tất cả FAISS index
    # =========================================
    def _load_all(self):
        """Load tất cả FAISS indexes từ thư mục index_dir."""
        # dish_name
        self.indexes["dish"] = faiss.read_index(
            f"{self.index_dir}/dish_name_embedding.index"
        )

        # ingredient names (flattened)
        self.indexes["names"] = faiss.read_index(
            f"{self.index_dir}/ingredient_names_embedding.index"
        )
        self.row_map["names"] = np.load(
            f"{self.index_dir}/ingredient_names_row_ids.npy"
        )

        print("✅ All FAISS indexes loaded successfully.")

    # =========================================
    # Search
    # =========================================
    def search(self, query_vec: np.ndarray, column_key: str, top_k: int = 10):
        """
        Tìm kiếm top-k kết quả trong FAISS index.

        Args:
            query_vec (np.ndarray): embedding vector để search
            column_key (str): 'dish' hoặc 'names'
            top_k (int): số lượng kết quả trả về

        Returns:
            list[dict]: danh sách dict kết quả, mỗi dict chứa:
                - dữ liệu gốc
                - '_distance': cosine similarity
                - '_rowid': id trong dataframe
                - '_match_count' (chỉ ingredient_names)
        """
        if column_key not in self.indexes:
            raise ValueError(f"Invalid column key: {column_key}")

        index = self.indexes[column_key]
        q = np.array(query_vec, dtype="float32").reshape(1, -1)
        distances, indices = index.search(q, top_k)

        # =====================================
        # Ingredient names → group by row_id
        # =====================================
        if column_key == "names":
            result_dict = {}
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                row_id = int(self.row_map["names"][idx])

                if row_id not in result_dict:
                    row_dict = self.df.iloc[row_id].to_dict()
                    row_dict["_distance"] = float(dist)
                    row_dict["_rowid"] = row_id
                    row_dict["_match_count"] = 1
                    result_dict[row_id] = row_dict
                else:
                    # tăng số nguyên liệu match
                    result_dict[row_id]["_match_count"] += 1
                    # giữ distance tốt nhất
                    if dist > result_dict[row_id]["_distance"]:
                        result_dict[row_id]["_distance"] = float(dist)

            # convert dict → list, sort theo match_count + distance
            results = list(result_dict.values())
            results.sort(key=lambda x: (-x["_match_count"], x["_distance"]))
            return results

        # =====================================
        # Dish → regular search
        # =====================================
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            row = self.df.iloc[idx].to_dict()
            row["_distance"] = float(dist)
            row["_rowid"] = idx
            results.append(row)

        return results
