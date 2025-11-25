# new_faiss_handler.py
import os
import faiss
import numpy as np


class FAISSHandler:

    def __init__(self, df, index_dir="./new_faiss_indexes"):

        self.df = df
        self.index_dir = index_dir

        # index name → df column
        self.cols = {
            "dish": "dish_name_embedding",
            "qty": "ingredient_quantities_embedding",
            "names": "ingredient_names_embedding"
        }

        self.indexes = {}
        self.row_map = {}

        self._load_all()

    def _load_all(self):

        # dish_name
        self.indexes["dish"] = faiss.read_index(f"{self.index_dir}/dish_name_embedding.index")

        # ingredient_quantities
        self.indexes["qty"] = faiss.read_index(f"{self.index_dir}/ingredient_quantities_embedding.index")

        # ingredient_names (flatten)
        self.indexes["names"] = faiss.read_index(f"{self.index_dir}/ingredient_names_embedding.index")
        self.row_map["names"] = np.load(f"{self.index_dir}/ingredient_names_row_ids.npy")

        print("✅ All FAISS indexes loaded.")

    def search(self, query_vec: np.ndarray, column_key: str, top_k: int = 10):

        if column_key not in self.indexes:
            raise ValueError(f"Invalid column key: {column_key}")

        index = self.indexes[column_key]

        # reshape into (1, dim)
        q = np.array(query_vec, dtype="float32").reshape(1, -1)

        distances, indices = index.search(q, top_k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):

            # map ingredient vector → recipe row
            if column_key == "names":
                row_id = self.row_map["names"][idx]
            else:
                row_id = idx

            row = self.df.iloc[row_id].to_dict()
            row["_distance"] = float(dist)
            row["_rowid"] = int(row_id)

            results.append(row)

        return results
