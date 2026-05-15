"""
FastAPI Web Interface for Agentic RAG Agent
Provides a modern web interface for interacting with the RAG agent
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent.agent import graph
from agent.config import Configuration

app = FastAPI(
    title="Agentic RAG Agent",
    description="A modern web interface for the Agentic RAG Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str = None
    sources: List[str] = []
    stats: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    config: Dict[str, Any] = {}

class StreamMessage(BaseModel):
    type: str
    content: str
    data: Dict[str, Any] = {}
    timestamp: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    stats: Dict[str, Any] = {}
    timestamp: str

chat_history: List[ChatMessage] = []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    print(f"Request received: {request.message}")
    try:
        default_config = {
            "query_generator_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "reflection_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "rag_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "answer_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "max_rag_loops": 3,
            "number_of_initial_queries": 2
        }

        config = {
            "configurable": {**default_config, **request.config}
        }

        initial_state = {
            "user_messages": [HumanMessage(content=request.message)],
            "rag_query": [],
            "rag_query_result": [],
            "source_gathered": [],
            "initial_rag_query_count": config["configurable"]["number_of_initial_queries"],
            "max_rag_loops": config["configurable"]["max_rag_loops"],
            "rag_loop_count": 0,
            "reasoning_model": config["configurable"]["rag_model"],
            "intent": "",
            "router_reason": "",
        }

        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now().isoformat()
        )
        chat_history.append(user_message)

        result = await graph.ainvoke(initial_state, config=config)

        response_content = ""
        if "user_messages" in result and result["user_messages"]:
            final_message = result["user_messages"][-1]
            if hasattr(final_message, 'content'):
                response_content = final_message.content
            else:
                response_content = str(final_message)
        else:
            response_content = "Sorry, no response was received from the agent."

        stats = {
            "rag_loop_count": result.get('rag_loop_count', 0),
            "total_queries": len(result.get('rag_query_result', [])),
            "unique_sources": len(set(result.get('source_gathered', []))),
            "processing_time": "N/A"
        }

        sources = list(set(result.get('source_gathered', [])))

        chat_response = ChatResponse(
            response=response_content,
            sources=sources,
            stats=stats,
            timestamp=datetime.now().isoformat()
        )

        agent_message = ChatMessage(
            role="assistant",
            content=response_content,
            timestamp=datetime.now().isoformat(),
            sources=sources,
            stats=stats
        )
        chat_history.append(agent_message)

        return chat_response

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/chat/stream")
async def chat_with_agent_stream(request: ChatRequest):
    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            default_config = {
                "query_generator_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
                "reflection_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
                "rag_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
                "answer_model": os.getenv("LLM_MODEL", "Qwen3.5-4B-Q4_K_M.gguf"),
                "max_rag_loops": int(os.getenv("MAX_RAG_LOOPS", "3")),
                "number_of_initial_queries": int(os.getenv("NUMBER_OF_INITIAL_QUERIES", "2"))
            }

            config = {
                "configurable": {**default_config, **request.config}
            }

            initial_state = {
                "user_messages": [HumanMessage(content=request.message)],
                "rag_query": [],
                "rag_query_result": [],
                "source_gathered": [],
                "initial_rag_query_count": config["configurable"]["number_of_initial_queries"],
                "max_rag_loops": config["configurable"]["max_rag_loops"],
                "rag_loop_count": 0,
                "reasoning_model": config["configurable"]["rag_model"],
                "intent": "",
                "router_reason": "",
            }

            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Initializing agent and analyzing question...', 'data': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

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
                        yield f"data: {json.dumps({'type': 'query_generation', 'content': f'Generated {len(queries)} search queries (1 EN + 1 VI)', 'data': {'queries': queries}, 'timestamp': datetime.now().isoformat()})}\n\n"

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
                                yield f"data: {json.dumps({'type': 'answer', 'content': response_content, 'data': {'sources': [], 'stats': {}}, 'timestamp': datetime.now().isoformat()})}\n\n"

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

                                yield f"data: {json.dumps({'type': 'answer', 'content': response_content, 'data': {'sources': sources, 'stats': stats}, 'timestamp': datetime.now().isoformat()})}\n\n"

            final_stats = {
                "total_steps": step_count,
                "processed_nodes": len(all_node_data)
            }
            yield f"data: {json.dumps({'type': 'complete', 'content': 'Processing complete', 'data': {'final_stats': final_stats}, 'timestamp': datetime.now().isoformat()})}\n\n"

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg, 'data': {}, 'timestamp': datetime.now().isoformat()})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")

@app.get("/api/chat/history")
async def get_chat_history():
    return {"messages": chat_history}

@app.delete("/api/chat/history")
async def clear_chat_history():
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared"}

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
