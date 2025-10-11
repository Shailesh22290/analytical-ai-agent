"""
FAISS-based vector database for storing and retrieving embeddings
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from config.settings import settings
from src.utils.models import VectorMetadata


class VectorStore:
    """FAISS vector store with metadata"""
    
    def __init__(self, dimension: int = settings.VECTOR_DIMENSION):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.metadata: List[VectorMetadata] = []
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        metadata: List[VectorMetadata]
    ) -> None:
        """
        Add vectors with metadata to the store
        
        Args:
            vectors: Array of vectors (n_vectors x dimension)
            metadata: List of metadata objects
        """
        if vectors.shape[0] != len(metadata):
            raise ValueError("Number of vectors must match metadata length")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match {self.dimension}")
        
        # Normalize vectors for better search
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadata)
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 5,
        file_id: Optional[str] = None
    ) -> List[Tuple[VectorMetadata, float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            file_id: Optional file_id to filter results
            
        Returns:
            List of (metadata, distance) tuples
        """
        # Normalize query
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search more results if filtering by file_id
        search_k = k * 10 if file_id else k
        
        distances, indices = self.index.search(query_vector, min(search_k, len(self.metadata)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                
                # Filter by file_id if specified
                if file_id and meta.file_id != file_id:
                    continue
                
                results.append((meta, float(dist)))
                
                if len(results) >= k:
                    break
        
        return results
    
    def get_vectors_by_file(self, file_id: str) -> List[Tuple[int, VectorMetadata]]:
        """
        Get all vector indices and metadata for a specific file
        
        Args:
            file_id: File identifier
            
        Returns:
            List of (index, metadata) tuples
        """
        results = []
        for idx, meta in enumerate(self.metadata):
            if meta.file_id == file_id:
                results.append((idx, meta))
        return results
    
    def save(self, file_id: str) -> None:
        """
        Save index and metadata to disk
        
        Args:
            file_id: Identifier for the saved files
        """
        index_path = settings.get_vector_db_path(file_id)
        metadata_path = settings.get_metadata_path(file_id)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    @classmethod
    def load(cls, file_id: str) -> 'VectorStore':
        """
        Load index and metadata from disk
        
        Args:
            file_id: Identifier for the saved files
            
        Returns:
            Loaded VectorStore instance
        """
        index_path = settings.get_vector_db_path(file_id)
        metadata_path = settings.get_metadata_path(file_id)
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Vector store for {file_id} not found")
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Create store instance
        store = cls(dimension=index.d)
        store.index = index
        store.metadata = metadata
        
        return store
    
    def size(self) -> int:
        """Get number of vectors in store"""
        return self.index.ntotal


class VectorStoreManager:
    """Manages multiple vector stores for different files"""
    
    def __init__(self):
        """Initialize manager"""
        self.stores: Dict[str, VectorStore] = {}
    
    def create_store(self, file_id: str) -> VectorStore:
        """
        Create a new vector store
        
        Args:
            file_id: File identifier
            
        Returns:
            New VectorStore instance
        """
        store = VectorStore()
        self.stores[file_id] = store
        return store
    
    def get_store(self, file_id: str) -> Optional[VectorStore]:
        """
        Get vector store for a file
        
        Args:
            file_id: File identifier
            
        Returns:
            VectorStore instance or None
        """
        if file_id not in self.stores:
            try:
                self.stores[file_id] = VectorStore.load(file_id)
            except FileNotFoundError:
                return None
        
        return self.stores[file_id]
    
    def save_store(self, file_id: str) -> None:
        """
        Save vector store to disk
        
        Args:
            file_id: File identifier
        """
        if file_id in self.stores:
            self.stores[file_id].save(file_id)
    
    def list_stores(self) -> List[str]:
        """
        List all available store file_ids
        
        Returns:
            List of file_ids
        """
        # Check both in-memory and on-disk stores
        file_ids = set(self.stores.keys())
        
        # Add saved stores from disk
        for path in settings.VECTOR_DIR.glob("*.faiss"):
            file_id = path.stem
            file_ids.add(file_id)
        
        return sorted(list(file_ids))


# Global manager instance
vector_store_manager = VectorStoreManager()