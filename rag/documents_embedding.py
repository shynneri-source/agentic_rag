import json
import os
import sys
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import get_embedding_model, generate_embeddings


class DocumentEmbeddingManager:
    """
    Manages document embeddings using GPU-accelerated embedding models
    and stores them in Qdrant vector database.
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "document_embeddings",
        embedding_dim: int = 1024,  
        chunks_file: str = "documents_chunks.json",
        use_gpu: bool = True
    ):
        """
        Initialize the embedding manager.
        
        Args:
            qdrant_host (str): Qdrant server host
            qdrant_port (int): Qdrant server port
            collection_name (str): Name of the Qdrant collection
            embedding_dim (int): Dimension of embeddings
            chunks_file (str): Path to document chunks JSON file
            use_gpu (bool): Whether to use GPU for embeddings
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.chunks_file = chunks_file
        self.use_gpu = use_gpu
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port
        )
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model()
        
        # Configure GPU usage if available and requested
        if self.use_gpu and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Move model to GPU if using sentence-transformers
            if hasattr(self.embedding_model, 'to'):
                self.embedding_model.to('cuda')
        else:
            print("Using CPU for embeddings")
            if self.use_gpu:
                print("Warning: GPU requested but not available")
    
    def load_document_chunks(self) -> List[Dict[str, Any]]:
        """
        Load document chunks from JSON file.
        
        Returns:
            List[Dict[str, Any]]: List of document chunks
        """
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")
        
        print(f"Loading document chunks from {self.chunks_file}")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        print(f"Loaded {len(chunks)} chunks from {data['summary']['total_documents']} documents")
        
        return chunks
    
    def create_qdrant_collection(self, recreate: bool = False) -> None:
        """
        Create Qdrant collection for storing embeddings.
        
        Args:
            recreate (bool): Whether to recreate collection if it exists
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists and recreate:
                print(f"Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                print(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection_name}' created successfully")
            else:
                print(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise
    
    def generate_chunk_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for document chunks using GPU acceleration.
        
        Args:
            chunks (List[Dict[str, Any]]): List of document chunks
            batch_size (int): Batch size for processing
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        print(f"Generating embeddings for {len(chunks)} chunks using GPU: {self.use_gpu and torch.cuda.is_available()}")
        
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Process in batches to manage memory
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = generate_embeddings(batch_texts)
                
                # Convert to numpy array if not already
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                elif not isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = np.array(batch_embeddings)
                
                # Add individual embeddings to list
                for j in range(len(batch_texts)):
                    embeddings.append(batch_embeddings[j])
                    
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                # Create zero embeddings as fallback
                for j in range(len(batch_texts)):
                    embeddings.append(np.zeros(self.embedding_dim))
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def store_embeddings_in_qdrant(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[np.ndarray],
        batch_size: int = 100
    ) -> None:
        """
        Store embeddings and metadata in Qdrant.
        
        Args:
            chunks (List[Dict[str, Any]]): Document chunks
            embeddings (List[np.ndarray]): Corresponding embeddings
            batch_size (int): Batch size for uploading
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
        
        print(f"Storing {len(chunks)} embeddings in Qdrant collection '{self.collection_name}'")
        
        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create point with embedding and metadata
            point = PointStruct(
                id=i,  # Use index as ID
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_index": chunk["document_index"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "content_length": chunk["content_length"],
                    "word_count": chunk["word_count"],
                    "filename": chunk["metadata"]["filename"],
                    "source": chunk["metadata"]["source"],
                    "file_size": chunk["metadata"]["file_size"],
                    "file_extension": chunk["metadata"]["file_extension"]
                }
            )
            points.append(point)
        
        # Upload in batches
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch_points = points[i:i + batch_size]
            
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
                raise
        
        print(f"Successfully stored {len(points)} embeddings in Qdrant")
    
    def process_and_store_embeddings(
        self, 
        recreate_collection: bool = False,
        embedding_batch_size: int = 32,
        storage_batch_size: int = 100
    ) -> None:
        """
        Complete pipeline: load chunks, generate embeddings, store in Qdrant.
        
        Args:
            recreate_collection (bool): Whether to recreate the Qdrant collection
            embedding_batch_size (int): Batch size for embedding generation
            storage_batch_size (int): Batch size for storage
        """
        print("Starting embedding processing pipeline...")
        
        try:
            # 1. Create Qdrant collection
            self.create_qdrant_collection(recreate=recreate_collection)
            
            # 2. Load document chunks
            chunks = self.load_document_chunks()
            
            # 3. Generate embeddings
            embeddings = self.generate_chunk_embeddings(
                chunks, 
                batch_size=embedding_batch_size
            )
            
            # 4. Store in Qdrant
            self.store_embeddings_in_qdrant(
                chunks, 
                embeddings, 
                batch_size=storage_batch_size
            )
            
            print("✅ Embedding processing pipeline completed successfully!")
            
            # Print collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection info: {collection_info.points_count} points stored")
            
        except Exception as e:
            print(f"❌ Error in embedding pipeline: {str(e)}")
            raise
    
    def search_similar_chunks(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            score_threshold (float): Minimum similarity score
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with scores
        """
        print(f"Searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = generate_embeddings([query])[0]
        
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "score": result.score,
                "chunk_id": result.payload["chunk_id"],
                "content": result.payload["content"],
                "filename": result.payload["filename"],
                "content_preview": result.payload["content"][:200] + "..." if len(result.payload["content"]) > 200 else result.payload["content"]
            })
        
        print(f"Found {len(results)} similar chunks")
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status.status if collection_info.optimizer_status else None,
            }
        except Exception as e:
            return {"error": f"Failed to get collection stats: {str(e)}"}


def main():
    """
    Example usage of DocumentEmbeddingManager.
    """
    # Initialize embedding manager
    embedding_manager = DocumentEmbeddingManager(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="document_embeddings",
        chunks_file="documents_chunks.json",
        use_gpu=True  # Enable GPU usage
    )
    
    # Process and store embeddings
    embedding_manager.process_and_store_embeddings(
        recreate_collection=True,  # Recreate collection for fresh start
        embedding_batch_size=16,   # Smaller batch size for GPU memory
        storage_batch_size=100
    )
    
    # Get collection statistics
    stats = embedding_manager.get_collection_stats()
    print("\nCollection Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example search
    print("\n" + "="*50)
    print("Example Search:")
    results = embedding_manager.search_similar_chunks(
        query="Bắc Giang thanh niên tình nguyện",
        limit=3,
        score_threshold=0.5
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   File: {result['filename']}")
        print(f"   Preview: {result['content_preview']}")


if __name__ == "__main__":
    main()
