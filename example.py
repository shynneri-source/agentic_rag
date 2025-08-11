"""
Example script để chạy agentic RAG agent
Người dùng có thể nhập câu hỏi và agent sẽ tự động tìm kiếm và trả lời
"""

from langchain_core.messages import HumanMessage
from agent.agent import graph
from agent.config import Configuration


def run_agent(question: str):
    """
    Chạy agent với câu hỏi đầu vào
    
    Args:
        question: Câu hỏi cần trả lời
    
    Returns:
        Kết quả từ agent
    """
    
    # Cấu hình cho agent
    config = {
        "configurable": {
            "query_generator_model": "qwen/qwen3-4b",
            "reflection_model": "qwen/qwen3-4b", 
            "rag_model": "qwen/qwen3-4b",
            "answer_model": "qwen/qwen3-4b",
            "max_rag_loops": 3,
            "number_of_initial_queries": 2
        }
    }
    
    # Khởi tạo state ban đầu
    initial_state = {
        "user_messages": [HumanMessage(content=question)],
        "rag_query": [],
        "rag_query_result": [],
        "source_gathered": [],
        "initial_rag_query_count": 2,
        "max_rag_loops": 3,
        "rag_loop_count": 0,
        "reasoning_model": "qwen/qwen3-4b"
    }
    
    print(f"🤖 Agent đang xử lý câu hỏi: {question}")
    print("=" * 50)
    
    try:
        # Chạy agent
        print("🔍 Bắt đầu chạy agent...")
        result = graph.invoke(initial_state, config=config)
        
        print(f"🔍 Kết quả từ agent có type: {type(result)}")
        if isinstance(result, dict):
            print(f"🔍 Keys trong result: {list(result.keys())}")
        
        # Lấy câu trả lời cuối cùng
        if "user_messages" in result and result["user_messages"]:
            answer = result["user_messages"][-1].content
            print(f"✅ Câu trả lời:")
            print(answer)
            print("=" * 50)
            
            # Hiển thị thông tin thêm
            if "source_gathered" in result and result["source_gathered"]:
                print(f"📚 Nguồn tham khảo: {', '.join(set(result['source_gathered']))}")
            
            if "rag_loop_count" in result:
                print(f"🔄 Số vòng lặp RAG: {result['rag_loop_count']}")
                
            return answer
        else:
            print("❌ Không thể tạo câu trả lời")
            print(f"🔍 user_messages có trong result? {'user_messages' in result if isinstance(result, dict) else 'result is not dict'}")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi khi chạy agent: {str(e)}")
        print(f"🔍 Loại lỗi: {type(e)}")
        import traceback
        print(f"🔍 Traceback:")
        traceback.print_exc()
        return None


def interactive_mode():
    """
    Chế độ tương tác - người dùng có thể nhập nhiều câu hỏi
    """
    print("🚀 Chào mừng đến với Agentic RAG Agent!")
    print("Nhập câu hỏi của bạn (nhập 'quit' để thoát):")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n💭 Câu hỏi của bạn: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Tạm biệt!")
                break
                
            if not question:
                print("⚠️ Vui lòng nhập câu hỏi!")
                continue
                
            run_agent(question)
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")


def main():
    """
    Hàm main - có thể chạy interactive mode hoặc test với câu hỏi mẫu
    """
    import sys
    
    if len(sys.argv) > 1:
        # Nếu có argument, sử dụng như câu hỏi
        question = " ".join(sys.argv[1:])
        run_agent(question)
    else:
        # Chế độ tương tác
        interactive_mode()


if __name__ == "__main__":
    main()