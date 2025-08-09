"""
Example usage of RAG system for question answering
"""
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import ask_question, search_documents


def main():
    """
    Example usage of the RAG system
    """
    print("🤖 RAG Question Answering System")
    print("=" * 50)
    
    # Example questions (in Vietnamese)
    example_questions = [
        "Bắc Giang có gì đặc biệt về thanh niên tình nguyện?",
        "Thông tin về hoạt động tình nguyện ở Bắc Giang như thế nào?",
        "Có những chương trình gì dành cho thanh niên?",
    ]
    
    print("Ví dụ câu hỏi:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "=" * 50)
    
    # Interactive mode
    while True:
        print("\nNhập câu hỏi của bạn (hoặc 'quit' để thoát):")
        user_question = input("❓ Câu hỏi: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'thoát']:
            print("👋 Tạm biệt!")
            break
        
        if not user_question:
            continue
        
        print(f"\n🔍 Đang tìm kiếm thông tin liên quan...")
        
        try:
            # Generate response using RAG
            response = ask_question(
                question=user_question,
                max_contexts=3,
                score_threshold=0.4,
                language="vietnamese"
            )
            
            print(f"\n📋 Kết quả:")
            print("-" * 30)
            
            # Display answer
            print(f"💬 Câu trả lời:")
            print(f"{response['answer']}")
            
            # Display confidence
            print(f"\n🎯 Độ tin cậy: {response['confidence']:.2f}")
            print(f"📚 Số nguồn tham khảo: {response.get('context_used', 0)}")
            
            # Display sources
            if response['sources']:
                print(f"\n📖 Nguồn tham khảo:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"{i}. File: {source['filename']}")
                    print(f"   Độ liên quan: {source['confidence']:.2f}")
                    print(f"   Chunk ID: {source['chunk_id']}")
            else:
                print(f"\n⚠️ Không tìm thấy nguồn tham khảo phù hợp")
                
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")
    

def test_search_function():
    """
    Test the document search function
    """
    print("\n🔍 Test tìm kiếm tài liệu:")
    print("=" * 30)
    
    test_query = "thanh niên tình nguyện"
    print(f"Truy vấn: '{test_query}'")
    
    try:
        results = search_documents(
            query=test_query,
            limit=3,
            score_threshold=0.3
        )
        
        print(f"\nTìm thấy {len(results)} kết quả:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result['filename']}")
            print(f"   Content preview: {result['content'][:150]}...")
            
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm: {str(e)}")


if __name__ == "__main__":
    print("Chọn chế độ:")
    print("1. Hỏi đáp tương tác")
    print("2. Test tìm kiếm")
    print("3. Cả hai")
    
    choice = input("Lựa chọn (1/2/3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_search_function()
    elif choice == "3":
        test_search_function()
        main()
    else:
        print("Lựa chọn không hợp lệ. Chạy chế độ mặc định...")
        main()
