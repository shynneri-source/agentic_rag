import re
from IPython.display import Image, display

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.states import OverallState, ReflectionState, QueryGenerationState, rag_query_state, Query

from agent.config import Configuration

from agent.prompt import query_writer_instructions, reflection_instructions, answer_instructions, router_instructions, chat_instructions

from agent.schema import Reflection, rag_query_list, FinalAnswer

from agent.utils import get_research_topic

from core.model import get_llm, model_manager
from typing import Literal



def router_node(state: OverallState, config: RunnableConfig) -> dict:
    llm = get_llm()
    question = get_research_topic(state["user_messages"])
    memories = state.get("memories", "")
    response = llm.invoke(router_instructions.format(question=question, memories=memories))
    content = response.content.strip().lower()
    intent = "rag" if "rag" in content else "chat"
    return {"intent": intent, "router_reason": content[:200]}

def route_decision(state: OverallState) -> Literal["chat", "generate_query"]:
    if state.get("intent") == "rag":
        return "generate_query"
    return "chat"

def chat_node(state: OverallState, config: RunnableConfig) -> OverallState:
    llm = get_llm()
    question = get_research_topic(state["user_messages"])
    memories = state.get("memories", "")
    response = llm.invoke(chat_instructions.format(question=question, memories=memories))
    return {"user_messages": [AIMessage(content=response.content)]}

def generate_query(state: OverallState, config: RunnableConfig) ->  QueryGenerationState:

    config_runnable = Configuration.from_runnable_config(config)
    if state.get("initial_rag_query_count"):
        config_runnable.number_of_initial_queries = state["initial_rag_query_count"]

    llm = get_llm()
    structured_llm = llm.with_structured_output(rag_query_list)

    memories = state.get("memories", "")
    fomatted_prompt =  query_writer_instructions.format(
        research_topic=get_research_topic(state["user_messages"]),
        rag_loop_count=state.get("rag_loop_count", 0),
        memories=memories,
    )
    
    result = structured_llm.invoke(fomatted_prompt)
    
    return {"rag_query": result.query}

def continue_rag_process(state: QueryGenerationState):
    queries = state.get("rag_query", [])
    if not queries:
        return "chat"
    return[
        Send("rag_research", {"rag_query": rag_query, "id": int(idx)})
        for idx, rag_query in enumerate(queries)
    ] 


def _is_relevant(query: str, doc_content: str) -> bool:
    """Check if document contains keywords from the query.
    Prevents semantic search from retrieving irrelevant documents.
    """
    query_lower = query.lower()
    doc_lower = doc_content.lower()

    stop_words = {
        'la', 'cua', 'va', 'co', 'duoc', 'cac', 'nhung', 'voi', 'tren', 've',
        'cho', 'de', 'tu', 'trong', 'khi', 'se', 'nay', 'mot', 'viec', 'khong',
        'tai', 'vao', 'ra', 'hay', 'hoac', 'theo', 'sau', 'truoc', 'qua', 'lai',
        'da', 'dang', 'gi', 'ai', 'bao', 'nhieu', 'o', 'thi', 'vi', 'nen',
        'rat', 'nhu', 'cung', 'deu', 'ma',
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'some', 'them',
        'than', 'what', 'when', 'which', 'their', 'about', 'would', 'could',
        'there', 'only', 'other', 'into', 'more', 'also', 'how', 'its',
    }

    query_words = set(re.findall(r'\w+', query_lower))
    query_words = {w for w in query_words if len(w) >= 3 and w not in stop_words}

    if not query_words:
        return True

    for word in query_words:
        if word in doc_lower:
            return True

    return False


def rag_research(state: rag_query_state, config: RunnableConfig) -> OverallState:
    """
    RAG research node: Use the RAG system to search and return results
    """
    query = state["rag_query"]

    docs = model_manager.search_similar_documents(
        query=query,
        collection_name="document_embeddings",
        limit=1,
        score_threshold=0.35
    )

    if not docs:
        return {
            "rag_query": [query],
            "rag_query_result": ["No relevant documents found."],
            "source_gathered": [],
        }

    relevant_docs = [doc for doc in docs if _is_relevant(query, doc["content"])]

    if not relevant_docs:
        return {
            "rag_query": [query],
            "rag_query_result": ["No relevant documents found."],
            "source_gathered": [],
        }

    raw_contents = []
    sources = []
    for i, doc in enumerate(relevant_docs):
        content = doc["content"]
        content = re.sub(
            r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})',
            r'Date: \1 Time: \2',
            content
        )
        raw_contents.append(f"[Source {i+1}]: {content}")
        sources.append(doc["filename"])
    
    return {
        "rag_query": [query],
        "rag_query_result": raw_contents,
        "source_gathered": sources,
    }

def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:


    config_runnable = Configuration.from_runnable_config(config)

    state["rag_loop_count"] = state.get("rag_loop_count", 0) + 1

    reasoning_model = state.get("reasoning_model", config_runnable.reflection_model)

    fomatted_prompt = reflection_instructions.format(
        research_topic=get_research_topic(state["user_messages"]),
        summaries="\n\n---\n\n".join(state["rag_query_result"]),
        rag_loop_count=state["rag_loop_count"],
        max_rag_loops=state.get("max_rag_loops", config_runnable.max_rag_loops)
    )

    llm = get_llm()
    result = llm.with_structured_output(Reflection).invoke(fomatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "rag_loop_count": state["rag_loop_count"],
        "number_of_rag_queries": len(state["rag_query_result"]),
    }

def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:

    configurable = Configuration.from_runnable_config(config)
    max_rag_loops = (
        state.get("max_rag_loops")
        if state.get("max_rag_loops") is not None
        else configurable.max_rag_loops
    )
    if state["is_sufficient"] or state["rag_loop_count"] >= max_rag_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "rag_research",
                {
                    "rag_query": follow_up_query,
                    "id": state["number_of_rag_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]
def finalize_answer(state: OverallState, config: RunnableConfig):
    config_runnable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model", config_runnable.answer_model)
    memories = state.get("memories", "")
    formatted_prompt = answer_instructions.format(
        research_topic=get_research_topic(state["user_messages"]),
        summaries="\n\n---\n\n".join(state["rag_query_result"]),
        rag_loop_count=state.get("rag_loop_count", 0),
        memories=memories,
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(FinalAnswer)
    result = structured_llm.invoke(formatted_prompt)

    return {
        "user_messages": [AIMessage(content=result.content)],
        "summary": result.summary,
        "sources": state.get("source_gathered", []),
        "rag_loop_count": state.get("rag_loop_count", 0)
    }

def get_graph_visualization(self, image_path: str = "./workflow.png") -> None:
    """Generate an .png image of the current graph workflow

    Args:
        image_path (str, optional): The path to save the image. Defaults to "./workflow.png".
    """
    try:
        png_bytes = self.graph.get_graph().draw_mermaid_png()

        # Display the image
        display(Image(png_bytes))

        # Save the image to a file
        with open(image_path, "wb") as f:
            f.write(png_bytes)

    except Exception as e:
        print(f"An error occurred: {e}")
builder = StateGraph(OverallState, config_schema=Configuration)


builder.add_node("router", router_node)
builder.add_node("chat", chat_node)
builder.add_node("generate_query", generate_query)
builder.add_node("rag_research", rag_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)


builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_decision, {"chat": "chat", "generate_query": "generate_query"})
builder.add_edge("chat", END)

builder.add_conditional_edges("generate_query", continue_rag_process, {"rag_research": "rag_research", "chat": "chat"})

builder.add_edge("rag_research", "reflection")

builder.add_conditional_edges("reflection", evaluate_research, ["finalize_answer", "rag_research"])

builder.add_edge("finalize_answer", END)

graph = builder.compile(name="agentic-rag")

# Generate and save the workflow visualization
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    
    # Save the image to a file
    with open("workflow.png", "wb") as f:
        f.write(png_bytes)
    print("Workflow visualization has been saved to workflow.png")
except Exception as e:
    print(f"An error occurred while generating workflow visualization: {e}")