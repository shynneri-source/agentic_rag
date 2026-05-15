"""
SQLite-backed persistent storage for conversations and messages.
Provides CRUD operations for sessions, conversations, and message history.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "conversations.db")


class ConversationStore:
    """Persistent SQLite store for conversations and messages."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        # Only create directory for file-based paths, not :memory:
        if db_path != ':memory:':
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        # For :memory: databases, cache and reuse the same connection
        # so tables survive across calls. File-based connections are
        # opened fresh each time (with WAL mode, this is fine).
        if self.db_path == ':memory:':
            if self._conn is None:
                self._conn = sqlite3.connect(':memory:')
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA foreign_keys=ON")
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    title TEXT DEFAULT 'New conversation',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    stats TEXT DEFAULT '{}',
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id);

                CREATE INDEX IF NOT EXISTS idx_conversations_session_id
                    ON conversations(session_id);

                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                    ON conversations(updated_at DESC);
            """)

    # --- Session methods ---

    def get_or_create_session_id(self, session_id: Optional[str] = None) -> str:
        """Return given session_id or generate a new one."""
        return session_id or str(uuid4())

    # --- Conversation methods ---

    def create_conversation(
        self,
        session_id: str,
        title: str = "New conversation",
        first_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation and return it."""
        conv_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        if first_message:
            title = first_message[:80] + ("..." if len(first_message) > 80 else "")

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, session_id, title, now, now),
            )

        return self.get_conversation(conv_id)

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def list_conversations(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List all conversations for a session, most recent first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE session_id = ? ORDER BY updated_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return cursor.rowcount > 0

    def delete_all_conversations(self, session_id: str) -> int:
        """Delete all conversations for a session. Returns number deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount

    # --- Message methods ---

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to a conversation. Updates conversation's updated_at."""
        now = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        stats_json = json.dumps(stats or {}, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO messages (conversation_id, role, content, sources, stats, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (conversation_id, role, content, sources_json, stats_json, now),
            )
            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            message_id = cursor.lastrowid

        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "stats": stats or {},
            "timestamp": now,
        }

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation, oldest first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ? OFFSET ?",
                (conversation_id, limit, offset),
            ).fetchall()

        messages = []
        for row in rows:
            msg = dict(row)
            try:
                msg["sources"] = json.loads(msg["sources"])
            except (json.JSONDecodeError, TypeError):
                msg["sources"] = []
            try:
                msg["stats"] = json.loads(msg["stats"])
            except (json.JSONDecodeError, TypeError):
                msg["stats"] = {}
            messages.append(msg)
        return messages

    def get_recent_messages(
        self,
        conversation_id: str,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the most recent N messages for a conversation, oldest first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM messages
                   WHERE conversation_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (conversation_id, count),
            ).fetchall()

        messages = []
        for row in reversed(rows):
            msg = dict(row)
            try:
                msg["sources"] = json.loads(msg["sources"])
            except (json.JSONDecodeError, TypeError):
                msg["sources"] = []
            try:
                msg["stats"] = json.loads(msg["stats"])
            except (json.JSONDecodeError, TypeError):
                msg["stats"] = {}
            messages.append(msg)
        return messages

    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            return row["count"] if row else 0

    def clear_all(self) -> None:
        """Delete all conversations and messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")


# Singleton instance
_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create the singleton conversation store."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ConversationStore()
    return _store_instance
