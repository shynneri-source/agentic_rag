"""
Qdrant-backed vector store for long-term conversation memories.
Memories are extracted facts from conversations, stored as vectors
for semantic retrieval across sessions.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np

from core.model import generate_embeddings


MEMORY_COLLECTION = "conversation_memories"
EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B


class MemoryStore:
    """Qdrant-backed long-term memory store for conversation memories."""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = MEMORY_COLLECTION,
    ):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the memory collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    @staticmethod
    def deduplicate_by_content(memories: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Deduplicate a list of memories by content similarity (exact match + substring)."""
        seen = set()
        deduped = []
        for m in memories:
            normalized = m["content"].strip().lower()
            if normalized in seen:
                continue
            if any(normalized in s or s in normalized for s in seen):
                continue
            seen.add(normalized)
            deduped.append(m)
            if len(deduped) >= limit:
                break
        return deduped

    def _embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        embedding = generate_embeddings([text])[0]
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().numpy()
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

    # --- CRUD ---

    def store_memory(
        self,
        session_id: str,
        conversation_id: str,
        content: str,
        memory_type: str = "fact",
        source_message_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Store a memory in Qdrant."""
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        vector = self._embed(content)

        point = PointStruct(
            id=memory_id,
            vector=vector,
            payload={
                "memory_id": memory_id,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "content": content,
                "memory_type": memory_type,
                "source_message_ids": json.dumps(source_message_ids or []),
                "created_at": now,
            },
        )

        self.qdrant_client.upsert(collection_name=self.collection_name, points=[point])
        return {
            "memory_id": memory_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "content": content,
            "memory_type": memory_type,
            "source_message_ids": source_message_ids or [],
            "created_at": now,
        }

    def find_similar_memories(
        self,
        content: str,
        session_id: Optional[str] = None,
        score_threshold: float = 0.92,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find memories with semantically similar content. Used to detect duplicates before storing."""
        return self.search_memories(
            query=content,
            session_id=session_id,
            limit=limit,
            score_threshold=score_threshold,
        )

    def search_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity."""
        vector = self._embed(query)

        search_filter = None
        if session_id:
            search_filter = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        ).points

        memories = []
        for r in results:
            payload = r.payload
            try:
                source_ids = json.loads(payload.get("source_message_ids", "[]"))
            except (json.JSONDecodeError, TypeError):
                source_ids = []
            memories.append({
                "memory_id": payload["memory_id"],
                "session_id": payload["session_id"],
                "conversation_id": payload["conversation_id"],
                "content": payload["content"],
                "memory_type": payload["memory_type"],
                "source_message_ids": source_ids,
                "created_at": payload["created_at"],
                "score": r.score,
            })

        return memories

    def list_memories(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List memories for a session (most recent first). Uses scroll with offset pagination to get the latest entries."""
        memory_filter = None
        if session_id:
            memory_filter = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )

        # Scroll forward to capture more entries, then sort by created_at
        all_points = []
        offset = None
        while len(all_points) < limit * 2:  # fetch up to 2x limit for better coverage
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=memory_filter,
                limit=min(limit * 2, 100),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            batch, offset = results
            if not batch:
                break
            all_points.extend(batch)

        memories = []
        for point in all_points:
            payload = point.payload
            try:
                source_ids = json.loads(payload.get("source_message_ids", "[]"))
            except (json.JSONDecodeError, TypeError):
                source_ids = []
            memories.append({
                "memory_id": payload["memory_id"],
                "session_id": payload["session_id"],
                "conversation_id": payload["conversation_id"],
                "content": payload["content"],
                "memory_type": payload["memory_type"],
                "source_message_ids": source_ids,
                "created_at": payload["created_at"],
            })

        # Sort by created_at descending, then deduplicate by content
        memories.sort(key=lambda m: m["created_at"], reverse=True)
        return self.deduplicate_by_content(memories, limit=limit)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id],
            )
            return True
        except Exception:
            return False

    def delete_conversation_memories(self, conversation_id: str) -> int:
        """Delete all memories associated with a conversation."""
        results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
            ),
            limit=9999,
            with_payload=False,
            with_vectors=False,
        )[0]

        ids = [p.id for p in results]
        if ids:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
            )
        return len(ids)

    def delete_session_memories(self, session_id: str) -> int:
        """Delete all memories for a session."""
        results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=9999,
            with_payload=False,
            with_vectors=False,
        )[0]

        ids = [p.id for p in results]
        if ids:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
            )
        return len(ids)

    def clear_all(self) -> None:
        """Delete all memories (recreates collection)."""
        self.qdrant_client.delete_collection(self.collection_name)
        self._ensure_collection()


# Singleton
_store_instance: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create the singleton memory store."""
    global _store_instance
    if _store_instance is None:
        _store_instance = MemoryStore()
    return _store_instance
