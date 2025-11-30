# bm25_handler.py

from rank_bm25 import BM25Okapi
import numpy as np

class BM25Handler:
    
    def __init__(self, corpus_tokens, raw_corpus=None):
        """
        corpus_tokens: list[list[str]] - token đã chuẩn hóa
        raw_corpus: list[str] - để trả về text gốc
        """
        self.corpus_tokens = corpus_tokens
        self.raw_corpus = raw_corpus if raw_corpus else [" ".join(doc) for doc in corpus_tokens]

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query_tokens, top_k=100, min_score=None):
        """
        query_tokens: list[str] - token đã chuẩn hóa
        """
        if not isinstance(query_tokens, list):
            raise ValueError("Query phải là list[str] - danh sách token đã chuẩn hóa")
        
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            score = float(scores[i])
            if min_score is not None and score < min_score:
                continue
            results.append({
                "doc_id": i,
                "text": self.raw_corpus[i],
                "score": score
            })
        
        return results