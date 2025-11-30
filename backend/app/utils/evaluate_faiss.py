"""
FAISSEvaluator
---------------
Module đánh giá hiệu suất và độ chính xác của FAISS index.
Bao gồm:
- Đánh giá tốc độ (latency / QPS)
- Đánh giá độ chính xác (precision/recall)
- Thông tin index
- Xuất báo cáo JSON
"""

import time
import json
import numpy as np
import pandas as pd
from faiss_handler import FAISSHandler


class FAISSEvaluator:
    def __init__(self, faiss_handler: FAISSHandler):
        """
        Args:
            faiss_handler (FAISSHandler): instance quản lý các FAISS index
        """
        self.faiss_handler = faiss_handler

    # =========================================
    # 1. EVALUATE SPEED
    # =========================================
    def evaluate_speed(self, num_queries=100, top_k=5):
        """
        Đánh giá tốc độ truy vấn FAISS.

        Args:
            num_queries (int): số lượng truy vấn random
            top_k (int): số kết quả trả về mỗi truy vấn

        Returns:
            dict: avg_time_ms, qps cho mỗi index
        """
        # Lấy dimension của embedding dish
        dimension = self.faiss_handler.indexes['dish'].d
        test_queries = [
            np.random.random(dimension).astype('float32')
            for _ in range(num_queries)
        ]

        results = {}
        for column_key in ['dish', 'names']:
            times = []
            for query in test_queries:
                start = time.time()
                self.faiss_handler.search(query, column_key, top_k)
                times.append((time.time() - start) * 1000)  # convert sang ms

            results[column_key] = {
                'avg_time_ms': np.mean(times),
                'qps': 1000 / np.mean(times)  # queries per second
            }

        return results

    # =========================================
    # 2. EVALUATE ACCURACY
    # =========================================
    def evaluate_accuracy(self, top_k=5, num_cases=100):
        """
        Đánh giá accuracy (precision/recall) trên các test case.

        Args:
            top_k (int): số kết quả trả về mỗi truy vấn
            num_cases (int): số lượng test case random

        Returns:
            dict: precision@k, recall@k, số test case
        """
        df = self.faiss_handler.df
        test_cases = []

        # Lấy sample ngẫu nhiên từ dataframe
        sample_indices = np.random.choice(
            len(df), min(num_cases, len(df)), replace=False
        )
        for idx in sample_indices:
            embedding = df.iloc[idx]['dish_name_embedding']
            if embedding is not None:
                test_cases.append({'query_vector': embedding, 'true_id': idx})

        results = {}
        for column_key in ['dish', 'names']:
            accuracies = []
            for case in test_cases:
                search_results = self.faiss_handler.search(
                    case['query_vector'], column_key, top_k
                )
                found_ids = [r['_rowid'] for r in search_results]
                accuracies.append(1 if case['true_id'] in found_ids else 0)

            accuracy = np.mean(accuracies)
            results[column_key] = {
                f'precision@{top_k}': round(accuracy * 100, 1),
                f'recall@{top_k}': round(accuracy * 100, 1),
                'test_cases': len(test_cases)
            }

        return results

    # =========================================
    # 3. INDEX INFO
    # =========================================
    def get_index_info(self):
        """
        Lấy thông tin các FAISS index.

        Returns:
            dict: số vectors và dimension cho mỗi index
        """
        info = {}
        for column_key in ['dish', 'names']:
            index = self.faiss_handler.indexes[column_key]
            info[column_key] = {'vectors': index.ntotal, 'dimension': index.d}
        return info

    # =========================================
    # 4. EVALUATE ALL & SAVE REPORT
    # =========================================
    def evaluate(self):
        """
        Thực hiện đánh giá speed + accuracy + thông tin index.
        Lưu report ra file JSON: 'faiss_report.json'

        Returns:
            dict: report đầy đủ
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'index_info': self.get_index_info(),
            'speed_results': self.evaluate_speed(),
            'accuracy_results': self.evaluate_accuracy()
        }

        with open('faiss_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    # Load dataframe embeddings
    df = pd.read_pickle('./data/recipes_embeddings.pkl')

    # Khởi tạo FAISS handler
    handler = FAISSHandler(df)

    # Khởi tạo evaluator và đánh giá
    evaluator = FAISSEvaluator(handler)
    report = evaluator.evaluate()
    print("✅ FAISS evaluation completed. Report saved to 'faiss_report.json'")
