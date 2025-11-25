# faiss_handler.py
import os
import faiss
import numpy as np

class FAISSHandler:
    def __init__(self, df, embedding_columns, index_dir="./faiss_indexes"):
        """
        df: DataFrame chứa metadata
        embedding_columns: dict {key: embedding_column_name}
        index_dir: thư mục chứa FAISS index
        """
        self.df = df
        self.embedding_columns = embedding_columns
        self.index_dir = index_dir
        self.indexes = {}
        self.row_indices = {}

        self._load_indexes()

    # ---------------------------------------
    # Load FAISS indexes
    # ---------------------------------------
    def _load_indexes(self):
        for key, col in self.embedding_columns.items():

            index_path = os.path.join(self.index_dir, f"{col}.index")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index not found: {index_path}")

            index = faiss.read_index(index_path)
            self.indexes[col] = index

            # if ingredient_names embedding → load row mapping
            if col == "ingredient_names_embedding":
                map_path = os.path.join(self.index_dir, f"{col}_row_indices.npy")
                if not os.path.exists(map_path):
                    raise FileNotFoundError(f"Row index mapping not found: {map_path}")
                self.row_indices[col] = np.load(map_path)

            print(f"✅ Loaded index '{col}' from: {index_path}")

    # ---------------------------------------
    # Search FAISS
    # ---------------------------------------
    def search(self, query_vector: np.ndarray, column_key: str, top_k: int = 10):
        if column_key not in self.embedding_columns:
            raise ValueError(f"Invalid column_key '{column_key}'")

        col_name = self.embedding_columns[column_key]
        index = self.indexes[col_name]

        # reshape query vector
        query_vector = np.array(query_vector, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # dimension check
        d_query = query_vector.shape[1]
        d_index = index.d

        if d_query != d_index:
            raise ValueError(
                f"[DimensionMismatch] Query dim = {d_query}, but index dim = {d_index}.\n"
                f"→ Bạn đang encode query bằng model khác model đã build index!"
            )

        # search
        distances, indices = index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # map idx → row index
            if column_key == "names":
                row_idx = self.row_indices[col_name][idx]
            else:
                row_idx = idx

            row = self.df.iloc[row_idx].to_dict()
            row["_distance"] = dist
            row["_rowid__"] = row_idx
            results.append(row)

        return results




# faiss_handler.py
# import os
# import faiss
# import numpy as np

# class FAISSHandler:
#     def __init__(self, df, embedding_columns, index_dir="./faiss_indexes"):
#         """
#         Class để load FAISS index từ file và search.
        
#         Args:
#             df: DataFrame chứa dữ liệu (metadata)
#             embedding_columns: dict {key: column_name_in_df}
#             index_dir: thư mục chứa index đã build
#         """
#         self.df = df
#         self.embedding_columns = embedding_columns
#         self.index_dir = index_dir
#         self.indexes = {}

#         # Load tất cả index từ file
#         self._load_indexes()

#     def _load_indexes(self):
#         for key, col in self.embedding_columns.items():
#             index_path = os.path.join(self.index_dir, f"{col}.index")
#             if not os.path.exists(index_path):
#                 raise FileNotFoundError(f"Index file not found: {index_path}")
#             self.indexes[col] = faiss.read_index(index_path)
#             print(f"✅ Loaded FAISS index for column '{col}' from '{index_path}'")

#     def search(self, query_vector: np.ndarray, column_key: str, top_k: int = 5):
#         """
#         Tìm top_k kết quả gần nhất cho query_vector trên cột embedding column_key.
#         Không dùng cosine similarity.
        
#         Args:
#             query_vector: np.ndarray (1D) vector query
#             column_key: key trong embedding_columns
#             top_k: số lượng kết quả trả về
#         Returns:
#             List of dict (row metadata + distance)
#         """
#         if column_key not in self.embedding_columns:
#             raise ValueError(f"Invalid column_key '{column_key}'")

#         index = self.indexes[self.embedding_columns[column_key]]
#         query_vector = np.array([query_vector]).astype("float32")

#         # Search top_k
#         distances, indices = index.search(query_vector, top_k)

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             row = self.df.iloc[idx].to_dict()
#             row["_distance"] = dist
#             row["__rowid__"] = idx   # 🔥 Quan trọng: lưu lại chỉ số dòng thật
#             results.append(row)
#         return results


# faiss_handler.py
# import os
# import faiss
# import numpy as np

# class FAISSHandler:
#     def __init__(self, df, embedding_columns, index_dir="./faiss_indexes"):
#         """
#         Load FAISS index + row mapping.
#         df: DataFrame chứa metadata.
#         embedding_columns: dict {key: column_name_in_df}
#         index_dir: thư mục chứa index đã build.
#         """
#         self.df = df
#         self.embedding_columns = embedding_columns
#         self.index_dir = index_dir
#         self.indexes = {}
#         self.row_indices = {}  # chỉ dùng cho flattened ingredient_names

#         self._load_indexes()

#     def _load_indexes(self):
#         for key, col in self.embedding_columns.items():
#             index_path = os.path.join(self.index_dir, f"{col}.index")
#             if not os.path.exists(index_path):
#                 raise FileNotFoundError(f"Index file not found: {index_path}")

#             index = faiss.read_index(index_path)
#             self.indexes[col] = index

#             # Nếu là ingredient_names_embedding, load row_indices
#             if col == "ingredient_names_embedding":
#                 row_idx_path = os.path.join(self.index_dir, f"{col}_row_indices.npy")
#                 if not os.path.exists(row_idx_path):
#                     raise FileNotFoundError(f"Row indices file not found: {row_idx_path}")
#                 self.row_indices[col] = np.load(row_idx_path)

#             print(f"✅ Loaded FAISS index for '{col}' from '{index_path}'")

#     def search(self, query_vector: np.ndarray, column_key: str, top_k: int = 10):
#         """
#         Tìm top_k kết quả gần nhất cho query_vector.
#         """
#         if column_key not in self.embedding_columns:
#             raise ValueError(f"Invalid column_key '{column_key}'")

#         index = self.indexes[self.embedding_columns[column_key]]
#         query_vector = np.array(query_vector, dtype='float32')
#         if query_vector.ndim == 1:
#             query_vector = query_vector.reshape(1, -1)
        
#         d_index = index.d
#         d_query = query_vector.shape[1]
        
#         distances, indices = index.search(query_vector, top_k)
#         results = []

#         for dist, idx in zip(distances[0], indices[0]):
#             row_dict = {}

#             # Nếu flattened ingredient_names → map về row gốc
#             if column_key == "names":
#                 row_idx = self.row_indices[self.embedding_columns[column_key]][idx]
#             else:
#                 row_idx = idx

#             row_dict.update(self.df.iloc[row_idx].to_dict())
#             row_dict["_distance"] = dist
#             row_dict["_rowid__"] = row_idx  # chỉ số món gốc
#             results.append(row_dict)

#         return results
