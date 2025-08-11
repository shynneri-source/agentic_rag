"""
FastAPI Web Interface for Agentic RAG Agent
Provides a modern web interface for interacting with the RAG agent
"""

import asyncio
import json
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

# Khởi tạo FastAPI app
app = FastAPI(
    title="Agentic RAG Agent",
    description="A modern web interface for the Agentic RAG Agent",
    version="1.0.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models cho request/response
class ChatMessage(BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str
    timestamp: str = None
    sources: List[str] = []
    stats: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    config: Dict[str, Any] = {}

class StreamMessage(BaseModel):
    type: str  # "thinking", "query_generation", "rag_search", "reflection", "answer", "complete"
    content: str
    data: Dict[str, Any] = {}
    timestamp: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    stats: Dict[str, Any] = {}
    timestamp: str

# Lưu trữ lịch sử chat
chat_history: List[ChatMessage] = []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Trả về trang chủ HTML"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Endpoint để chat với agent
    """
    print(f"📨 Nhận được request: {request.message}")  # Debug log
    
    try:
        # Cấu hình mặc định
        default_config = {
            "query_generator_model": "qwen/qwen3-4b",
            "reflection_model": "qwen/qwen3-4b", 
            "rag_model": "qwen/qwen3-4b",
            "answer_model": "qwen/qwen3-4b",
            "max_rag_loops": 3,
            "number_of_initial_queries": 2
        }
        
        # Merge với config từ request
        config = {
            "configurable": {**default_config, **request.config}
        }
        
        print("⚙️ Config:", config)  # Debug log
        
        # Khởi tạo state ban đầu
        initial_state = {
            "user_messages": [HumanMessage(content=request.message)],
            "rag_query": [],
            "rag_query_result": [],
            "source_gathered": [],
            "initial_rag_query_count": config["configurable"]["number_of_initial_queries"],
            "max_rag_loops": config["configurable"]["max_rag_loops"],
            "rag_loop_count": 0,
            "reasoning_model": config["configurable"]["rag_model"]
        }
        
        print("🔄 Khởi tạo state thành công")  # Debug log
        
        # Thêm message của user vào lịch sử
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now().isoformat()
        )
        chat_history.append(user_message)
        
        print("🤖 Đang chạy agent...")  # Debug log
        
        # Chạy agent
        result = await graph.ainvoke(initial_state, config=config)
        
        print("✅ Agent hoàn thành, kết quả:", list(result.keys()))
        
        # Lấy response từ kết quả
        response_content = ""
        if "user_messages" in result and result["user_messages"]:
            final_message = result["user_messages"][-1]
            if hasattr(final_message, 'content'):
                response_content = final_message.content
            else:
                response_content = str(final_message)
        else:
            response_content = "Xin lỗi, không nhận được phản hồi từ agent."
        
        print(f"📤 Response: {response_content[:100]}...")  # Debug log
        
        # Tạo thống kê
        stats = {
            "rag_loop_count": result.get('rag_loop_count', 0),
            "total_queries": len(result.get('rag_query_result', [])),
            "unique_sources": len(set(result.get('source_gathered', []))),
            "processing_time": "N/A"
        }
        
        sources = list(set(result.get('source_gathered', [])))
        
        print(f"📊 Stats: {stats}")  # Debug log
        
        # Tạo response
        chat_response = ChatResponse(
            response=response_content,
            sources=sources,
            stats=stats,
            timestamp=datetime.now().isoformat()
        )
        
        # Thêm response của agent vào lịch sử
        agent_message = ChatMessage(
            role="assistant",
            content=response_content,
            timestamp=datetime.now().isoformat(),
            sources=sources,
            stats=stats
        )
        chat_history.append(agent_message)
        
        print("✅ Hoàn thành request")  # Debug log
        
        return chat_response
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.post("/api/chat/stream")
async def chat_with_agent_stream(request: ChatRequest):
    """
    Endpoint để chat với agent với streaming
    """
    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            # Cấu hình mặc định
            default_config = {
                "query_generator_model": "qwen/qwen3-4b",
                "reflection_model": "qwen/qwen3-4b", 
                "rag_model": "qwen/qwen3-4b",
                "answer_model": "qwen/qwen3-4b",
                "max_rag_loops": 3,
                "number_of_initial_queries": 2
            }
            
            # Merge với config từ request
            config = {
                "configurable": {**default_config, **request.config}
            }
            
            # Khởi tạo state ban đầu
            initial_state = {
                "user_messages": [HumanMessage(content=request.message)],
                "rag_query": [],
                "rag_query_result": [],
                "source_gathered": [],
                "initial_rag_query_count": config["configurable"]["number_of_initial_queries"],
                "max_rag_loops": config["configurable"]["max_rag_loops"],
                "rag_loop_count": 0,
                "reasoning_model": config["configurable"]["rag_model"]
            }
            
            # Stream thinking
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Đang khởi tạo agent và phân tích câu hỏi...', 'data': {}, 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Chạy agent với streaming
            step_count = 0
            all_node_data = {}  # Lưu trữ tất cả data từ các nodes
            seen_rag_queries = set()  # Theo dõi các RAG queries đã hiển thị
            
            async for event in graph.astream(initial_state, config=config):
                step_count += 1
                print(f"🔄 Step {step_count}: {list(event.keys())}")  # Debug log
                
                # Xử lý từng node event
                for node_name, node_data in event.items():
                    print(f"📊 Node {node_name}: {list(node_data.keys())}")  # Debug log
                    
                    # Lưu trữ data từ node này
                    if node_name not in all_node_data:
                        all_node_data[node_name] = []
                    all_node_data[node_name].append(node_data)
                    
                    if node_name == "generate_query":
                        queries = node_data.get("rag_query", [])
                        yield f"data: {json.dumps({'type': 'query_generation', 'content': f'Đã tạo {len(queries)} câu truy vấn để tìm kiếm thông tin', 'data': {'queries': queries}, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    elif node_name == "rag_research":
                        # Lấy query từ state hoặc từ Send message
                        query = node_data.get("rag_query", "Đang tìm kiếm...")
                        
                        # Xử lý query - có thể là string hoặc object từ Send message
                        query_str = "Đang tìm kiếm..."
                        if isinstance(query, str):
                            query_str = query
                        elif hasattr(query, 'query'):
                            query_str = query.query
                        elif isinstance(query, dict):
                            if "query" in query:
                                query_str = query["query"]
                            elif "rag_query" in query:
                                query_str = str(query["rag_query"])
                        else:
                            query_str = str(query)
                            
                        if "rag_query_result" in node_data:
                            # Tạo unique key để tránh duplicate
                            query_key = f"{query_str}_{step_count}"
                            
                            # Chỉ hiển thị nếu chưa thấy query này
                            if query_key not in seen_rag_queries:
                                seen_rag_queries.add(query_key)
                                query_result = node_data.get("rag_query_result", [])
                                results_count = len(query_result) if isinstance(query_result, list) else 1
                                content_msg = f'🔎 Tìm kiếm: "{query_str}" - Tìm thấy {results_count} kết quả'
                                yield f"data: {json.dumps({'type': 'rag_search', 'content': content_msg, 'data': {'query': query_str, 'results_count': results_count}, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    elif node_name == "reflection":
                        is_sufficient = node_data.get("is_sufficient", False)
                        rag_loop_count = node_data.get("rag_loop_count", 0)
                        knowledge_gap = node_data.get("knowledge_gap", "")
                        
                        if is_sufficient:
                            content = f"✅ Đánh giá: Thông tin đã đủ để trả lời câu hỏi (Vòng RAG: {rag_loop_count})"
                            yield f"data: {json.dumps({'type': 'reflection', 'content': content, 'data': {'is_sufficient': True, 'rag_loop_count': rag_loop_count}, 'timestamp': datetime.now().isoformat()})}\n\n"
                        else:
                            content = f"🔄 Đánh giá: Cần tìm thêm thông tin - {knowledge_gap} (Vòng RAG: {rag_loop_count})"
                            
                            # Hiển thị follow-up queries nếu có
                            follow_up_queries = node_data.get("follow_up_queries", [])
                            if follow_up_queries:
                                content += f" - Sẽ tìm thêm {len(follow_up_queries)} câu truy vấn"
                                
                            yield f"data: {json.dumps({'type': 'reflection', 'content': content, 'data': {'is_sufficient': False, 'rag_loop_count': rag_loop_count, 'knowledge_gap': knowledge_gap, 'follow_up_queries': follow_up_queries}, 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    elif node_name == "finalize_answer":
                        if "user_messages" in node_data:
                            final_messages = node_data["user_messages"]
                            if final_messages:
                                final_message = final_messages[-1]
                                raw_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                                
                                # Clean up content: loại bỏ \n đầu và cuối, trim whitespace
                                response_content = raw_content.strip()
                                
                                # Lấy thống kê từ tất cả data đã thu thập
                                all_rag_results = []
                                all_sources = []
                                max_rag_loop_count = 0
                                
                                # Tổng hợp data từ tất cả nodes
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
            
            # Stream completion với tổng hợp cuối cùng
            final_stats = {
                "total_steps": step_count,
                "processed_nodes": len(all_node_data)
            }
            yield f"data: {json.dumps({'type': 'complete', 'content': 'Hoàn thành xử lý', 'data': {'final_stats': final_stats}, 'timestamp': datetime.now().isoformat()})}\n\n"
            
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg, 'data': {}, 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")

@app.get("/api/chat/history")
async def get_chat_history():
    """Lấy lịch sử chat"""
    return {"messages": chat_history}

@app.delete("/api/chat/history")
async def clear_chat_history():
    """Xóa lịch sử chat"""
    global chat_history
    chat_history = []
    return {"message": "Đã xóa lịch sử chat"}

@app.get("/api/health")
async def health_check():
    """Kiểm tra tình trạng hệ thống"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Agentic RAG Agent đang hoạt động bình thường"
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động Agentic RAG Web Interface...")
    print("📱 Truy cập: http://localhost:3000")
    print("📖 API Docs: http://localhost:3000/docs")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=3000, 
        reload=True,
        log_level="info"
    )
