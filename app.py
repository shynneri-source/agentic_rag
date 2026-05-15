"""
FastAPI Web Interface for Agentic RAG Agent
Provides a modern web interface for interacting with the RAG agent
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent.agent import graph
from agent.config import Configuration
from store.conversation_store import get_conversation_store
from store.memory_store import get_memory_store, MemoryStore
from store.memory_extractor import get_memory_extractor

app = FastAPI(
    title="Shyn AI",
    description="Shyn AI — Agentic RAG Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    config: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    session_id: str
    sources: List[str] = []
    stats: Dict[str, Any] = {}
    timestamp: str

class ConversationSummary(BaseModel):
    id: str
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

class MessageOut(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    sources: List[str] = []
    stats: Dict[str, Any] = {}
    timestamp: str

# --- Helper functions ---

def _get_default_config() -> dict:
    return {
        "query_generator_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
        "reflection_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
        "rag_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
        "answer_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
        "max_rag_loops": int(os.getenv("MAX_RAG_LOOPS", "3")),
        "number_of_initial_queries": int(os.getenv("NUMBER_OF_INITIAL_QUERIES", "2")),
    }


def _build_initial_state(
    message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    memories: str = "",
) -> dict:
    """Build the initial LangGraph state, optionally including conversation history and relevant memories."""
    config = {
        "configurable": {**_get_default_config(), **(config_overrides or {})}
    }

    # Build user_messages from conversation history + new message
    user_messages = []
    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                user_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                user_messages.append(AIMessage(content=msg["content"]))
    user_messages.append(HumanMessage(content=message))

    return {
        "user_messages": user_messages,
        "rag_query": [],
        "rag_query_result": [],
        "source_gathered": [],
        "initial_rag_query_count": config["configurable"]["number_of_initial_queries"],
        "max_rag_loops": config["configurable"]["max_rag_loops"],
        "rag_loop_count": 0,
        "reasoning_model": config["configurable"]["rag_model"],
        "intent": "",
        "router_reason": "",
        "memories": memories,
    }, config


def _store_messages(
    store,
    conversation_id: str,
    request_message: str,
    response_content: str,
    sources: List[str],
    stats: Dict[str, Any],
) -> None:
    """Store user message and assistant response in the database."""
    store.add_message(conversation_id, "user", request_message)
    store.add_message(conversation_id, "assistant", response_content, sources=sources, stats=stats)


def _load_relevant_memories(
    memory_store,
    message: str,
    session_id: str,
    limit: int = 5,
) -> str:
    """Load relevant memories from past conversations for context injection.
    Deduplicates by content similarity before formatting.
    """
    try:
        memories = memory_store.search_memories(
            query=message,
            session_id=session_id,
            limit=limit * 3,  # fetch more to have room for dedup
            score_threshold=0.3,
        )
        if not memories:
            return ""

        # Deduplicate by content using shared helper
        deduped = MemoryStore.deduplicate_by_content(memories, limit=limit)

        if not deduped:
            return ""

        parts = ["Here are relevant memories from your past conversations:"]
        for m in deduped:
            parts.append(f"- {m['content']}")
        return "\n".join(parts)
    except Exception as e:
        print(f"Memory loading error: {e}")
        return ""


async def _extract_and_store_memories(
    memory_store,
    extractor,
    session_id: str,
    conversation_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """Background task: extract memories from a conversation turn and store them in Qdrant.
    Skips storing if a semantically similar memory already exists for this session.
    """
    stored_count = 0
    try:
        memories = extractor.extract(user_message, assistant_response)
        for memory in memories:
            # Check if a similar memory already exists before storing
            similar = memory_store.find_similar_memories(
                content=memory.content,
                session_id=session_id,
                score_threshold=0.92,
                limit=1,
            )
            if similar:
                print(f"Skipping duplicate memory: {memory.content[:60]}...")
                continue
            memory_store.store_memory(
                session_id=session_id,
                conversation_id=conversation_id,
                content=memory.content,
                memory_type=memory.memory_type,
            )
            stored_count += 1
        if stored_count:
            print(f"Stored {stored_count} new memories from conversation turn")
    except Exception as e:
        print(f"Memory extraction error: {e}")


def _extract_result(response: dict) -> tuple:
    """Extract response content, sources, and stats from the agent result."""
    response_content = ""
    if "user_messages" in response and response["user_messages"]:
        final_message = response["user_messages"][-1]
        if hasattr(final_message, 'content'):
            response_content = final_message.content
        else:
            response_content = str(final_message)
    else:
        response_content = "Sorry, no response was received from the agent."

    stats = {
        "rag_loop_count": response.get("rag_loop_count", 0),
        "total_queries": len(response.get("rag_query_result", [])),
        "unique_sources": len(set(response.get("source_gathered", []))),
        "processing_time": "N/A",
    }
    sources = list(set(response.get("source_gathered", [])))
    return response_content, sources, stats


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Synchronous chat endpoint with session persistence."""
    store = get_conversation_store()
    session_id = store.get_or_create_session_id(request.session_id)

    try:
        # Get or create conversation
        conversation_id = request.conversation_id
        if conversation_id:
            conv = store.get_conversation(conversation_id)
            if not conv or conv["session_id"] != session_id:
                conv = None
        else:
            conv = None

        if not conv:
            conv = store.create_conversation(
                session_id=session_id,
                first_message=request.message,
            )
            conversation_id = conv["id"]

        # Load recent conversation history (last 10 messages) for context
        recent_messages = store.get_recent_messages(conversation_id, count=10)

        # Load relevant long-term memories
        memory_store = get_memory_store()
        memories = _load_relevant_memories(memory_store, request.message, session_id)

        # Build initial state with history and memories
        initial_state, config = _build_initial_state(
            message=request.message,
            conversation_history=recent_messages,
            config_overrides=request.config,
            memories=memories,
        )

        # Run the agent
        result = await graph.ainvoke(initial_state, config=config)

        # Extract result
        response_content, sources, stats = _extract_result(result)

        # Store messages in database
        _store_messages(store, conversation_id, request.message, response_content, sources, stats)

        # Extract and store long-term memories in the background
        extractor = get_memory_extractor()
        asyncio.create_task(_extract_and_store_memories(
            memory_store, extractor,
            session_id, conversation_id,
            request.message, response_content,
        ))

        return ChatResponse(
            response=response_content,
            conversation_id=conversation_id,
            session_id=session_id,
            sources=sources,
            stats=stats,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/chat/stream")
async def chat_with_agent_stream(request: ChatRequest):
    """Streaming chat endpoint with session persistence."""
    store = get_conversation_store()
    session_id = store.get_or_create_session_id(request.session_id)

    # Get or create conversation upfront
    conversation_id = request.conversation_id
    if conversation_id:
        conv = store.get_conversation(conversation_id)
        if not conv or conv["session_id"] != session_id:
            conv = None
    else:
        conv = None

    if not conv:
        conv = store.create_conversation(
            session_id=session_id,
            first_message=request.message,
        )
        conversation_id = conv["id"]

    # Store the user message upfront so it's persisted regardless of routing path
    store.add_message(conversation_id, "user", request.message)

    # Get singleton instances for use inside the generator
    memory_store = get_memory_store()
    extractor = get_memory_extractor()

    async def stream_response() -> AsyncGenerator[str, None]:
        nonlocal conversation_id
        try:
            # Load recent conversation history (last 10 messages) for context
            recent_messages = store.get_recent_messages(conversation_id, count=10)

            # Load relevant long-term memories (memory_store and extractor from closure)
            memories = _load_relevant_memories(memory_store, request.message, session_id)

            # Build initial state with history and memories
            initial_state, config = _build_initial_state(
                message=request.message,
                conversation_history=recent_messages,
                config_overrides=request.config,
                memories=memories,
            )

            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Initializing agent and analyzing question...', 'data': {'conversation_id': conversation_id, 'session_id': session_id}, 'timestamp': datetime.now().isoformat()})}\n\n"

            step_count = 0
            all_node_data = {}
            seen_rag_queries = set()

            async for event in graph.astream(initial_state, config=config):
                if not event:
                    continue
                step_count += 1

                for node_name, node_data in event.items():
                    if not node_data:
                        continue
                    if not isinstance(node_data, dict):
                        continue

                    if node_name not in all_node_data:
                        all_node_data[node_name] = []
                    all_node_data[node_name].append(node_data)

                    if node_name == "generate_query":
                        queries = node_data.get("rag_query", [])
                        yield f"data: {json.dumps({'type': 'query_generation', 'content': f'Generated {len(queries)} search queries', 'data': {'queries': queries}, 'timestamp': datetime.now().isoformat()})}\n\n"

                    elif node_name == "rag_research":
                        queries = node_data.get("rag_query", [])
                        query_str = queries[0] if queries else "No query"

                        if "rag_query_result" in node_data:
                            query_key = f"{query_str}_{step_count}"

                            if query_key not in seen_rag_queries:
                                seen_rag_queries.add(query_key)
                                query_result = node_data.get("rag_query_result", [])
                                results_count = len(query_result) if isinstance(query_result, list) else 1
                                content_msg = f'Searching: "{query_str}" — found {results_count} result(s)'
                                yield f"data: {json.dumps({'type': 'rag_search', 'content': content_msg, 'data': {'query': query_str, 'results_count': results_count}, 'timestamp': datetime.now().isoformat()})}\n\n"

                    elif node_name == "reflection":
                        is_sufficient = node_data.get("is_sufficient", False)
                        rag_loop_count = node_data.get("rag_loop_count", 0)
                        knowledge_gap = node_data.get("knowledge_gap", "")

                        if is_sufficient:
                            content = f"Information sufficient to answer (RAG loop: {rag_loop_count})"
                            yield f"data: {json.dumps({'type': 'reflection', 'content': content, 'data': {'is_sufficient': True, 'rag_loop_count': rag_loop_count}, 'timestamp': datetime.now().isoformat()})}\n\n"
                        else:
                            content = f"Need more information — {knowledge_gap} (RAG loop: {rag_loop_count})"

                            follow_up_queries = node_data.get("follow_up_queries", [])
                            if follow_up_queries:
                                content += f" — will generate {len(follow_up_queries)} follow-up queries"

                            yield f"data: {json.dumps({'type': 'reflection', 'content': content, 'data': {'is_sufficient': False, 'rag_loop_count': rag_loop_count, 'knowledge_gap': knowledge_gap, 'follow_up_queries': follow_up_queries}, 'timestamp': datetime.now().isoformat()})}\n\n"

                    elif node_name == "chat":
                        if "user_messages" in node_data:
                            final_messages = node_data["user_messages"]
                            if final_messages:
                                final_message = final_messages[-1]
                                raw_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                                response_content = raw_content.strip()

                                # Store assistant response for chat path
                                store.add_message(conversation_id, "assistant", response_content, sources=[], stats={})

                                # Extract and store long-term memories in the background
                                asyncio.create_task(_extract_and_store_memories(
                                    memory_store, extractor,
                                    session_id, conversation_id,
                                    request.message, response_content,
                                ))

                                yield f"data: {json.dumps({'type': 'answer', 'content': response_content, 'data': {'sources': [], 'stats': {}, 'conversation_id': conversation_id}, 'timestamp': datetime.now().isoformat()})}\n\n"

                    elif node_name == "finalize_answer":
                        if "user_messages" in node_data:
                            final_messages = node_data["user_messages"]
                            if final_messages:
                                final_message = final_messages[-1]
                                raw_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

                                response_content = raw_content.strip()

                                all_rag_results = []
                                all_sources = []
                                max_rag_loop_count = 0

                                for node_type, node_list in all_node_data.items():
                                    for data in node_list:
                                        if "rag_query_result" in data:
                                            all_rag_results.extend(data["rag_query_result"])
                                        if "source_gathered" in data:
                                            all_sources.extend(data["source_gathered"])
                                        if "rag_loop_count" in data:
                                            max_rag_loop_count = max(max_rag_loop_count, data["rag_loop_count"])

                                stats = {
                                    "rag_loop_count": max_rag_loop_count,
                                    "total_queries": len(all_rag_results),
                                    "unique_sources": len(set(all_sources)),
                                    "processing_steps": step_count
                                }

                                sources = list(set(all_sources))

                                # Store assistant response for RAG path (user msg already stored upstream)
                                store.add_message(conversation_id, "assistant", response_content, sources=sources, stats=stats)

                                # Extract and store long-term memories in the background
                                asyncio.create_task(_extract_and_store_memories(
                                    memory_store, extractor,
                                    session_id, conversation_id,
                                    request.message, response_content,
                                ))

                                yield f"data: {json.dumps({'type': 'answer', 'content': response_content, 'data': {'sources': sources, 'stats': stats, 'conversation_id': conversation_id}, 'timestamp': datetime.now().isoformat()})}\n\n"

            final_stats = {
                "total_steps": step_count,
                "processed_nodes": len(all_node_data)
            }
            yield f"data: {json.dumps({'type': 'complete', 'content': 'Processing complete', 'data': {'final_stats': final_stats}, 'timestamp': datetime.now().isoformat()})}\n\n"

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"Stream error: {error_msg}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg, 'data': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")


# --- Session & History management endpoints ---

@app.get("/api/chat/sessions")
async def list_sessions(session_id: str = Query(..., description="Session identifier")):
    """List all conversations for a session."""
    store = get_conversation_store()
    conversations = store.list_conversations(session_id)

    result = []
    for conv in conversations:
        count = store.get_message_count(conv["id"])
        result.append(ConversationSummary(
            id=conv["id"],
            session_id=conv["session_id"],
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            message_count=count,
        ))

    return {"conversations": result}


@app.get("/api/chat/history")
async def get_chat_history(
    conversation_id: str = Query(..., description="Conversation ID"),
):
    """Get all messages for a conversation."""
    store = get_conversation_store()
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = store.get_messages(conversation_id)
    return {
        "conversation": ConversationSummary(
            id=conv["id"],
            session_id=conv["session_id"],
            title=conv["title"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
            message_count=len(messages),
        ),
        "messages": [MessageOut(**m) for m in messages],
    }


@app.delete("/api/chat/sessions/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation and its associated memories."""
    store = get_conversation_store()
    if not store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Clean up associated memories from Qdrant
    memory_store = get_memory_store()
    deleted_memories = memory_store.delete_conversation_memories(conversation_id)
    return {"message": "Conversation deleted", "memories_cleaned": deleted_memories}


@app.delete("/api/chat/history")
async def clear_chat_history(session_id: str = Query(..., description="Session identifier")):
    """Delete all conversations for a session."""
    store = get_conversation_store()
    count = store.delete_all_conversations(session_id)
    return {"message": f"Deleted {count} conversations"}


# --- Memory Management Endpoints ---

@app.get("/api/memories")
async def list_memories(session_id: str = Query(..., description="Session identifier")):
    """List all long-term memories for a session."""
    memory_store = get_memory_store()
    memories = memory_store.list_memories(session_id=session_id)
    return {"memories": memories}


@app.get("/api/memories/search")
async def search_memories(
    session_id: str = Query(..., description="Session identifier"),
    query: str = Query(..., description="Search query"),
):
    """Search long-term memories by semantic similarity."""
    memory_store = get_memory_store()
    memories = memory_store.search_memories(
        query=query,
        session_id=session_id,
        limit=20,  # fetch more to allow for dedup
        score_threshold=0.3,
    )

    # Deduplicate by content using shared helper
    deduped = MemoryStore.deduplicate_by_content(memories, limit=10)
    return {"memories": deduped}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory."""
    memory_store = get_memory_store()
    if memory_store.delete_memory(memory_id):
        return {"message": "Memory deleted"}
    raise HTTPException(status_code=404, detail="Memory not found")


@app.delete("/api/memories")
async def clear_memories(session_id: str = Query(..., description="Session identifier")):
    """Delete all memories for a session."""
    memory_store = get_memory_store()
    count = memory_store.delete_session_memories(session_id)
    return {"message": f"Deleted {count} memories"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Agentic RAG Agent is running"
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Agentic RAG Web Interface...")
    print("Access: http://localhost:3000")
    print("API Docs: http://localhost:3000/docs")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )
