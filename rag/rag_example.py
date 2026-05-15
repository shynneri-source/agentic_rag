"""
Example usage of RAG system for question answering
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import ask_question, search_documents


def main():
    print("RAG Question Answering System")
    print("=" * 50)

    example_questions = [
        "What information is available in the knowledge base?",
        "Can you summarize the key topics?",
        "What are the main findings?",
    ]

    print("Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")

    print("\n" + "=" * 50)

    while True:
        print("\nEnter your question (or 'quit' to exit):")
        user_question = input("Question: ").strip()

        if user_question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_question:
            continue

        print("\nSearching for relevant information...")

        try:
            response = ask_question(
                question=user_question,
                max_contexts=3,
                score_threshold=0.4,
            )

            print(f"\nResults:")
            print("-" * 30)
            print(f"Answer: {response['answer']}")
            print(f"\nConfidence: {response['confidence']:.2f}")
            print(f"Sources used: {response.get('context_used', 0)}")

            if response['sources']:
                print(f"\nReferences:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"{i}. File: {source['filename']}")
                    print(f"   Relevance: {source['confidence']:.2f}")
            else:
                print("\nNo relevant sources found")

        except Exception as e:
            print(f"Error: {str(e)}")


def test_search_function():
    print("\nTest Document Search:")
    print("=" * 30)

    test_query = "search query"
    print(f"Query: '{test_query}'")

    try:
        results = search_documents(
            query=test_query,
            limit=3,
            score_threshold=0.3
        )

        print(f"\nFound {len(results)} results:")

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   File: {result['filename']}")
            print(f"   Content preview: {result['content'][:150]}...")

    except Exception as e:
        print(f"Search error: {str(e)}")


if __name__ == "__main__":
    print("Select mode:")
    print("1. Interactive Q&A")
    print("2. Test search")
    print("3. Both")

    choice = input("Choice (1/2/3): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        test_search_function()
    elif choice == "3":
        test_search_function()
        main()
    else:
        print("Invalid choice. Running default mode...")
        main()
