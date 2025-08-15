import json
import time
from langchain_core.messages import HumanMessage
from agent.agent import graph
from agent.config import Configuration

def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['cau_hoi'] for item in data]

def process_questions(questions):
    results = []
    total = len(questions)
    
    # Cấu hình mặc định cho agent
    default_config = {
        "query_generator_model": "qwen/qwen3-30b-a3b",
        "reflection_model": "qwen/qwen3-30b-a3b", 
        "rag_model": "qwen/qwen3-30b-a3b",
        "answer_model": "qwen/qwen3-30b-a3b",
        "max_rag_loops": 3,
        "number_of_initial_queries": 2
    }
    
    config = {"configurable": default_config}
    
    for idx, question in enumerate(questions, 1):
        print(f"\nProcessing question {idx}/{total}")
        print(f"Question: {question}")
        
        try:
            # Khởi tạo state ban đầu cho câu hỏi
            initial_state = {
                "user_messages": [HumanMessage(content=question)],
                "rag_query": [],
                "rag_query_result": [],
                "source_gathered": [],
                "initial_rag_query_count": config["configurable"]["number_of_initial_queries"],
                "max_rag_loops": config["configurable"]["max_rag_loops"]
            }
            
            # Chạy agent graph với state ban đầu
            final_state = graph.invoke(initial_state, config=config)
            
            # Lấy câu trả lời từ user_messages cuối cùng
            response_content = "Không có câu trả lời"
            if "user_messages" in final_state and final_state["user_messages"]:
                final_message = final_state["user_messages"][-1]
                if hasattr(final_message, 'content'):
                    response_content = final_message.content
                else:
                    response_content = str(final_message)
            
            result = {
                "cau_hoi": question,
                "ket_qua_agent": response_content
            }
            results.append(result)
            
            # Save intermediate results after each question
            save_results(results, "agent_results_intermediate.json")
            
            print(f"Answer: {response_content}\n")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            result = {
                "cau_hoi": question,
                "ket_qua_agent": f"Error: {str(e)}"
            }
            results.append(result)
            
        # Add a small delay between questions to prevent overload
        time.sleep(2)
    
    return results

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "datatest/test.json"
    output_file = "agent_results.json"
    
    print("Loading questions from test.json...")
    questions = load_questions(input_file)
    print(f"Loaded {len(questions)} questions")
    
    print("\nStarting to process questions...")
    results = process_questions(questions)
    
    print("\nSaving final results...")
    save_results(results, output_file)
    
    print(f"\nComplete! Results saved to {output_file}")
