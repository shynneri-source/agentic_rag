"""
Example script to run the agentic RAG agent
Users can input questions and the agent will search and answer
"""

from langchain_core.messages import HumanMessage
from agent.agent import graph
from agent.config import Configuration


def run_agent(question: str):
    """
    Run the agent with an input question

    Args:
        question: The question to answer

    Returns:
        Agent result
    """

    config = {
        "configurable": {
            "query_generator_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "reflection_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "rag_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "answer_model": "Qwen3.5-4B-Q4_K_M.gguf",
            "max_rag_loops": 3,
            "number_of_initial_queries": 2
        }
    }

    initial_state = {
        "user_messages": [HumanMessage(content=question)],
        "rag_query": [],
        "rag_query_result": [],
        "source_gathered": [],
        "initial_rag_query_count": 2,
        "max_rag_loops": 3,
        "rag_loop_count": 0,
        "reasoning_model": "Qwen3.5-4B-Q4_K_M.gguf",
        "intent": "",
        "router_reason": "",
    }

    print(f"Agent processing question: {question}")
    print("=" * 50)

    try:
        result = graph.invoke(initial_state, config=config)

        if isinstance(result, dict) and "user_messages" in result and result["user_messages"]:
            answer = result["user_messages"][-1].content
            print(f"Answer:")
            print(answer)
            print("=" * 50)

            if "source_gathered" in result and result["source_gathered"]:
                print(f"Sources: {', '.join(set(result['source_gathered']))}")

            if "rag_loop_count" in result:
                print(f"RAG loops: {result['rag_loop_count']}")

            return answer
        else:
            print("Could not generate an answer")
            return None

    except Exception as e:
        print(f"Error running agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode():
    print("Welcome to Agentic RAG Agent!")
    print("Enter your question (type 'quit' to exit):")
    print("=" * 50)

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not question:
                print("Please enter a question!")
                continue

            run_agent(question)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_agent(question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
