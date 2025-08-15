from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.schema import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
import os
import numpy as np
from typing import List, Dict, Any, Optional


class ModelManager:
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.qdrant_client = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM and embedding models"""
        # Initialize LLM with LangChain OpenAI for LMStudio
        self.llm = ChatOpenAI(
            model="qwen/qwen3-30b-a3b",
            base_url="http://localhost:1234/v1",  # Default LMStudio local server
            api_key="lm-studio",  # LMStudio doesn't require a real API key
            temperature=0.7,
        )
        # Initialize embedding model with Sentence Transformers (force GPU)
        self.embedding_model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device="cuda"  # Force GPU
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host="localhost",
            port=6333
        )
    
    def get_llm(self):
        """Get the initialized LLM model"""
        return self.llm
    
    def get_embedding_model(self):
        """Get the initialized embedding model"""
        return self.embedding_model
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts (always use GPU)"""
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts, device="cuda")
    
    def search_similar_documents(
        self, 
        query: str, 
        collection_name: str = "document_embeddings",
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Qdrant based on query
        
        Args:
            query (str): User's question/query
            collection_name (str): Qdrant collection name
            limit (int): Maximum number of results to retrieve
            score_threshold (float): Minimum similarity score threshold
            
        Returns:
            List[Dict[str, Any]]: List of relevant document chunks
        """
        # Generate embedding for the query
        query_embedding = self.generate_embeddings([query])[0]
        
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu().numpy()
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "score": result.score,
                "content": result.payload["content"],
                "filename": result.payload["filename"],
                "chunk_id": result.payload["chunk_id"],
                "source": result.payload["source"]
            })
        
        return results
    
    def generate_rag_response(
        self,
        question: str,
        collection_name: str = "document_embeddings",
        max_contexts: int = 3,
        score_threshold: float = 0.5,
        language: str = "vietnamese"
    ) -> Dict[str, Any]:
        """
        Generate response using RAG (Retrieval-Augmented Generation)
        
        Args:
            question (str): User's question
            collection_name (str): Qdrant collection name
            max_contexts (int): Maximum number of context chunks to use
            score_threshold (float): Minimum similarity score for retrieval
            language (str): Response language preference
            
        Returns:
            Dict[str, Any]: Response with answer and sources
        """
        try:
            # Step 1: Retrieve relevant documents and prepare context in parallel
            relevant_docs = self.search_similar_documents(
                query=question,
                collection_name=collection_name,
                limit=max_contexts,
                score_threshold=score_threshold
            )
            
            if not relevant_docs:
                return {
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Efficient parallel processing of context and sources
            context_parts = []
            sources = []
            
            # Process documents once for both context and sources
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"[Nguồn {i+1}]: {doc['content']}")
                sources.append({
                    "filename": doc["filename"],
                    "source": doc["source"],
                    "confidence": doc["score"],
                    "chunk_id": doc["chunk_id"]
                })
            
            # Join contexts with minimal formatting
            context = "\n".join(context_parts)
            
            # Step 3: Create optimized prompt for LLM
            if language.lower() == "vietnamese":
                system_prompt = """/nothink Bạn là AI assistant. Trả lời câu hỏi dựa trên thông tin được cung cấp.
Quy tắc:
1. Chỉ dùng thông tin từ nguồn
2. Trả lời ngắn gọn, đủ ý
3. Dùng [Nguồn X] để dẫn nguồn"""
                
                user_prompt = f"""Dựa trên thông tin sau đây:

{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi một cách chính xác và chi tiết."""
            else:
                system_prompt = """/nothink You are an AI assistant. Answer questions based on provided information.
Rules:
1. Use only source information
2. Be concise but complete
3. Use [Source X] for citations"""
                
                user_prompt = f"""Context: {context}
Question: {question}
Answer concisely using the context."""
            
            # Step 4: Generate response using LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Calculate average confidence from sources
            avg_confidence = sum(doc["score"] for doc in relevant_docs) / len(relevant_docs)
            
            return {
                "answer": response.content,
                "sources": sources,
                "confidence": avg_confidence,
                "context_used": len(relevant_docs)
            }
            
        except Exception as e:
            return {
                "answer": f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }


# Global instance
model_manager = ModelManager()

# Convenience functions
def get_llm():
    """Get the LLM model instance"""
    return model_manager.get_llm()

def get_embedding_model():
    """Get the embedding model instance"""
    return model_manager.get_embedding_model()

def generate_embeddings(texts):
    """Generate embeddings for texts"""
    return model_manager.generate_embeddings(texts)


def ask_question(question: str, **kwargs) -> Dict[str, Any]:
    """
    Ask a question using RAG system
    
    Args:
        question (str): User's question
        **kwargs: Additional parameters for generate_rag_response
        
    Returns:
        Dict[str, Any]: Response with answer and sources
    """
    return model_manager.generate_rag_response(question, **kwargs)


def search_documents(query: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Search for similar documents
    
    Args:
        query (str): Search query
        **kwargs: Additional parameters for search_similar_documents
        
    Returns:
        List[Dict[str, Any]]: List of relevant documents
    """
    return model_manager.search_similar_documents(query, **kwargs)
