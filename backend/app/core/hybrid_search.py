import numpy as np
from bm25_model import VietnameseBM25
from faiss_manager import FaissManager

class HybridFoodSearch:
    """
    Hybrid search combining BM25 and FAISS semantic search
    """
    
    def __init__(self, bm25_k1=1.5, bm25_b=0.75, 
                 embedding_model='keepitreal/vietnamese-sbert'):
        self.bm25 = VietnameseBM25(k1=bm25_k1, b=bm25_b)
        self.faiss_manager = FaissManager(model_name=embedding_model)
        self.documents = []
        
    def build_index(self, documents):
        """
        Build both BM25 and FAISS indexes
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.documents = documents
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25.fit(documents)
        
        # Build FAISS index
        print("Building FAISS index...")
        self.faiss_manager.build_index(documents)
        
        print("Hybrid search index built successfully!")
    
    def hybrid_search(self, query, top_k=10, method='reciprocal_rank', 
                     bm25_weight=0.4, semantic_weight=0.6):
        """
        Perform hybrid search using both BM25 and semantic search
        """
        # Get results from both methods
        bm25_results = self.bm25.search(query, top_k * 3)
        semantic_results = self.faiss_manager.search(query, top_k * 3)
        
        if method == 'reciprocal_rank':
            return self._reciprocal_rank_fusion(bm25_results, semantic_results, top_k)
        elif method == 'linear':
            return self._linear_combination(bm25_results, semantic_results, top_k, 
                                          bm25_weight, semantic_weight)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _reciprocal_rank_fusion(self, bm25_results, semantic_results, top_k):
        """Reciprocal Rank Fusion method"""
        # Create rank dictionaries
        bm25_ranks = {self.documents[idx]: rank for rank, (idx, score) in enumerate(bm25_results)}
        semantic_ranks = {result['document']: rank for rank, result in enumerate(semantic_results)}
        
        all_documents = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
        
        fused_scores = []
        for doc in all_documents:
            bm25_rank = bm25_ranks.get(doc, top_k * 10)
            semantic_rank = semantic_ranks.get(doc, top_k * 10)
            
            # RRF formula
            rrf_score = (1 / (60 + bm25_rank)) + (1 / (60 + semantic_rank))
            
            fused_scores.append({
                'document': doc,
                'score': rrf_score,
                'bm25_rank': bm25_rank,
                'semantic_rank': semantic_rank
            })
        
        # Sort by score descending
        fused_scores.sort(key=lambda x: x['score'], reverse=True)
        return fused_scores[:top_k]
    
    def _linear_combination(self, bm25_results, semantic_results, top_k, 
                          bm25_weight, semantic_weight):
        """Linear combination of scores"""
        # Convert to dictionaries for easy lookup
        bm25_scores = {self.documents[idx]: score for idx, score in bm25_results}
        semantic_scores = {result['document']: result['score'] for result in semantic_results}
        
        # Normalize scores
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        max_semantic = max(semantic_scores.values()) if semantic_scores else 1
        
        all_documents = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        combined_scores = []
        for doc in all_documents:
            bm25_score = bm25_scores.get(doc, 0) / max_bm25
            semantic_score = semantic_scores.get(doc, 0) / max_semantic
            
            combined_score = (bm25_weight * bm25_score + 
                            semantic_weight * semantic_score)
            
            combined_scores.append({
                'document': doc,
                'score': combined_score,
                'bm25_score': bm25_score,
                'semantic_score': semantic_score
            })
        
        combined_scores.sort(key=lambda x: x['score'], reverse=True)
        return combined_scores[:top_k]