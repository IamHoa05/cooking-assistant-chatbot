import time
import pandas as pd
import numpy as np
import json
from faiss_handler import FAISSHandler

class FAISSEvaluator:
    def __init__(self, faiss_handler):
        self.faiss_handler = faiss_handler
    
    def evaluate_speed(self, num_queries=100, top_k=5):
        dimension = self.faiss_handler.indexes['dish'].d
        test_queries = [np.random.random(dimension).astype('float32') for _ in range(num_queries)]
        
        results = {}
        for column_key in ['dish', 'names']:
            times = []
            for query in test_queries:
                start = time.time()
                self.faiss_handler.search(query, column_key, top_k)
                times.append((time.time() - start) * 1000)
            
            results[column_key] = {
                'avg_time_ms': np.mean(times),
                'qps': 1000 / np.mean(times)
            }
        return results
    
    def evaluate_accuracy(self, top_k=5, num_cases=100):
        test_cases = []
        df = self.faiss_handler.df
        
        sample_indices = np.random.choice(len(df), min(num_cases, len(df)), replace=False)
        for idx in sample_indices:
            embedding = df.iloc[idx]['dish_name_embedding']
            if embedding is not None:
                test_cases.append({'query_vector': embedding, 'true_id': idx})
        
        results = {}    
        for column_key in ['dish', 'names']:
            accuracies = []
            for case in test_cases:
                search_results = self.faiss_handler.search(case['query_vector'], column_key, top_k)
                found_ids = [r['_rowid'] for r in search_results]
                accuracies.append(1 if case['true_id'] in found_ids else 0)
            
            accuracy = np.mean(accuracies)
            results[column_key] = {
                f'precision@{top_k}': round(accuracy * 100, 1),
                f'recall@{top_k}': round(accuracy * 100, 1),
                'test_cases': len(test_cases)
            }
        return results
    
    def get_index_info(self):
        info = {}
        for column_key in ['dish', 'names']:
            index = self.faiss_handler.indexes[column_key]
            info[column_key] = {'vectors': index.ntotal, 'dimension': index.d}
        return info
    
    def evaluate(self):
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'index_info': self.get_index_info(),
            'speed_results': self.evaluate_speed(),
            'accuracy_results': self.evaluate_accuracy()
        }
        
        with open('faiss_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

if __name__ == "__main__":
    df = pd.read_pickle('./data/recipes_embeddings.pkl')
    handler = FAISSHandler(df)
    evaluator = FAISSEvaluator(handler)
    report = evaluator.evaluate()