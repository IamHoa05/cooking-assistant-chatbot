# faiss_handler.py
import os
import faiss
import numpy as np


class FAISSHandler:

    def __init__(self, df, index_dir="./faiss_indexes"):
        self.df = df
        self.index_dir = index_dir

        # index name → df column
        self.cols = {
            "dish": "dish_name_embedding",
            "names": "ingredient_names_embedding"
        }

        self.indexes = {}
        self.row_map = {}

        self._load_all()

    def _load_all(self):
        """Load tất cả FAISS indexes."""
        # dish_name
        self.indexes["dish"] = faiss.read_index(f"{self.index_dir}/dish_name_embedding.index")

        # ingredient names (flatten)
        self.indexes["names"] = faiss.read_index(f"{self.index_dir}/ingredient_names_embedding.index")
        self.row_map["names"] = np.load(f"{self.index_dir}/ingredient_names_row_ids.npy")

        print("✅ All FAISS indexes loaded successfully.")

    def search(self, query_vecs: np.ndarray, column_key: str, top_k: int = 10):
        if column_key not in self.indexes:
            raise ValueError(f"Invalid column key: {column_key}")

        index = self.indexes[column_key]
        query_arr = np.array(query_vecs, dtype="float32")
        
        distances, indices = index.search(query_arr, top_k)
        
        # ingredient names → group by row_id, flat dict
        if column_key == "names":
            result_dict = {}
            for q_idx in range(len(query_vecs)):
                for dist, idx in zip(distances[q_idx], indices[q_idx]):
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
                        # giữ distance CAO nhất (tương đồng tốt nhất) - ĐÃ SỬA
                        if dist > result_dict[row_id]["_distance"]:
                            result_dict[row_id]["_distance"] = float(dist)

            # convert dict → list, sort theo match_count + distance
            results = list(result_dict.values())
            results.sort(key=lambda x: (-x["_match_count"], -x["_distance"]))
            return results

        # dish → regular search
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            row = self.df.iloc[idx].to_dict()
            row["_distance"] = float(dist)
            row["_rowid"] = int(idx)
            results.append(row)
        
        results.sort(key=lambda x: -x["_distance"])
        return results