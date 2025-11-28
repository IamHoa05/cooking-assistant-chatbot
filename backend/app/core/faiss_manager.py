import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

class FaissManager:
    """
    FAISS-based semantic search manager
    """
    
    def __init__(self, model_name='keepitreal/vietnamese-sbert', device='cpu'):
        """
        Initialize FAISS manager with embedding model
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.embedding_model = None
        self.index = None
        self.embeddings = None
        self.documents = []
        self.dimension = 0
        self.index_loaded = False
        
    def load_model(self):
        """
        Load the embedding model
        """
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def build_index(self, documents, normalize=True):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of text documents
            normalize: Whether to normalize vectors for cosine similarity
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        self.load_model()
        self.documents = documents
        
        print("Generating embeddings...")
        # Generate embeddings
        self.embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.embeddings = self.embeddings.astype('float32')
        
        print("Building FAISS index...")
        # Create FAISS index
        self.dimension = self.embeddings.shape[1]
        
        # Use Inner Product (dot product) index for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if normalize:
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(self.embeddings)
        
        # Add vectors to index
        self.index.add(self.embeddings)
        self.index_loaded = True
        
        print(f"FAISS index built. Documents: {len(documents)}, Index size: {self.index.ntotal}")
    
    def search(self, query, top_k=10, return_scores=True):
        """
        Semantic search using FAISS
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            list: List of search results
        """
        if not self.index_loaded:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        self.load_model()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        # Normalize query for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    'index': int(idx),
                    'document': self.documents[idx],
                    'score': float(score)
                }
                if not return_scores:
                    result.pop('score', None)
                results.append(result)
        
        return results
    
    def batch_search(self, queries, top_k=10):
        """
        Search for multiple queries
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            
        Returns:
            list: List of search results for each query
        """
        if not self.index_loaded:
            raise ValueError("FAISS index not built. Call build_index() first.")
        
        self.load_model()
        
        # Encode all queries
        query_embeddings = self.embedding_model.encode(queries)
        query_embeddings = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        all_results = []
        for query_idx, query in enumerate(queries):
            query_results = []
            for i, (score, doc_idx) in enumerate(zip(scores[query_idx], indices[query_idx])):
                if doc_idx < len(self.documents):
                    query_results.append({
                        'index': int(doc_idx),
                        'document': self.documents[doc_idx],
                        'score': float(score)
                    })
            all_results.append(query_results)
        
        return all_results
    
    def save_index(self, filepath):
        """
        Save FAISS index and documents to file
        """
        if not self.index_loaded:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save documents and metadata
        metadata = {
            'documents': self.documents,
            'model_name': self.model_name,
            'dimension': self.dimension
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"FAISS index saved to {filepath}.faiss")
    
    def load_index(self, filepath):
        """
        Load FAISS index and documents from file
        """
        if not os.path.exists(f"{filepath}.faiss"):
            raise FileNotFoundError(f"FAISS index file not found: {filepath}.faiss")
        
        if not os.path.exists(f"{filepath}.pkl"):
            raise FileNotFoundError(f"Metadata file not found: {filepath}.pkl")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        self.index_loaded = True
        
        print(f"FAISS index loaded. Documents: {len(self.documents)}, Index size: {self.index.ntotal}")
    
    def get_index_size(self):
        """Get number of documents in the index"""
        return self.index.ntotal if self.index_loaded else 0
    
    def get_embedding_dimension(self):
        """Get embedding dimension"""
        return self.dimension