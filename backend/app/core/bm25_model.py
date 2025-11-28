import math
from collections import defaultdict
import numpy as np
import jieba

class VietnameseBM25:
    """
    BM25 implementation optimized for Vietnamese text
    """
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avgdl = 0
        self.idf = {}
        self.fitted = False
    
    def vietnamese_tokenize(self, text):
        """
        Tokenize Vietnamese text using jieba
        """
        if not text or not isinstance(text, str):
            return []
        return list(jieba.cut(text.lower()))
    
    def fit(self, documents):
        """
        Train BM25 on a collection of documents
        
        Args:
            documents: List of text documents
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.corpus = documents
        self.doc_lengths = []
        total_doc_length = 0
        
        print("Training BM25 model...")
        
        # Calculate document statistics
        for doc in documents:
            tokens = self.vietnamese_tokenize(doc)
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            total_doc_length += doc_length
            
            # Calculate document frequency (each word counted once per document)
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        # Calculate average document length
        self.avgdl = total_doc_length / len(documents)
        
        # Calculate IDF for each word
        N = len(documents)
        for word, freq in self.doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[word] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        
        self.fitted = True
        print(f"BM25 training completed. Vocabulary size: {len(self.idf)}")
        return self
    
    def get_scores(self, query):
        """
        Calculate BM25 scores for query against all documents
        
        Args:
            query: Search query string
            
        Returns:
            numpy.array: BM25 scores for all documents
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        query_tokens = self.vietnamese_tokenize(query)
        scores = np.zeros(len(self.corpus))
        
        for i, doc in enumerate(self.corpus):
            doc_tokens = self.vietnamese_tokenize(doc)
            doc_length = self.doc_lengths[i]
            
            score = 0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # Term frequency in document
                tf = doc_tokens.count(token)
                
                # Calculate BM25 component
                numerator = self.idf[token] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                
                if denominator > 0:
                    score += numerator / denominator
            
            scores[i] = score
        
        return scores
    
    def search(self, query, top_k=10, return_scores=True):
        """
        Search for top_k documents matching the query
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            return_scores: Whether to return scores
            
        Returns:
            list: List of (document_index, score) or document_index
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        if return_scores:
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        else:
            return [int(idx) for idx in top_indices if scores[idx] > 0]
    
    def batch_search(self, queries, top_k=10):
        """
        Search for multiple queries
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            list: List of search results for each query
        """
        all_results = []
        for query in queries:
            results = self.search(query, top_k=top_k)
            all_results.append(results)
        return all_results
    
    def get_document_count(self):
        """Get number of documents in the index"""
        return len(self.corpus)
    
    def get_vocabulary_size(self):
        """Get vocabulary size"""
        return len(self.idf)
    
    def save_index(self, filepath):
        """
        Save BM25 index to file
        """
        import pickle
        
        index_data = {
            'corpus': self.corpus,
            'doc_freqs': dict(self.doc_freqs),
            'doc_lengths': self.doc_lengths,
            'avgdl': self.avgdl,
            'idf': self.idf,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, filepath):
        """
        Load BM25 index from file
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.corpus = index_data['corpus']
        self.doc_freqs = defaultdict(int, index_data['doc_freqs'])
        self.doc_lengths = index_data['doc_lengths']
        self.avgdl = index_data['avgdl']
        self.idf = index_data['idf']
        self.k1 = index_data.get('k1', 1.5)
        self.b = index_data.get('b', 0.75)
        self.fitted = True