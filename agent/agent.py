import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.states import OverallState, ReflectionState, QueryGenerationState, rag_query_state, RagStateOutput, Query

from agent.config import Configuration

from agent.prompt import query_writer_instructions, reflection_instructions, answer_instructions

from agent.schema import Reflection, rag_query_list

from agent.utils import get_research_topic

from core.model import ask_question



def generate_query(state: OverallState, config: RunnableConfig) ->  QueryGenerationState:

    config_runnable = Configuration.from_runnable_config(config)
    if state.get("initial_rag_query_count"):
        config_runnable.number_of_initial_queries = state["initial_rag_query_count"]

    llm = ChatOpenAI(
        model=config_runnable.query_generator_model,
        base_url="http://localhost:1234/v1",  # Default LMStudio local server
        api_key="lm-studio",  # LMStudio doesn't require a real API key
        temperature=0.7,        
    )

    structured_llm = llm.with_structured_output(rag_query_list)

    fomatted_prompt =  query_writer_instructions.format(
        research_topic=get_research_topic(state["user_messages"]),
        rag_loop_count=state.get("rag_loop_count", 0)
    )
    
    result = structured_llm.invoke(fomatted_prompt)
    
    return {"rag_query": result.query}

def continue_rag_process(state: QueryGenerationState):

    return[
        Send("rag_research", {"rag_query": rag_query, "id": int(idx)})
        for idx, rag_query in enumerate(state["rag_query"])
    ] 

def rag_research(state: rag_query_state, config: RunnableConfig) -> OverallState:
    """
    Node RAG research: Sử dụng RAG system để tìm kiếm và trả về kết quả
    """
    # Sử dụng RAG system để tìm kiếm thông tin
    rag_result = ask_question(
        question=state["rag_query"],
        collection_name="document_embeddings",
        max_contexts=5,
        score_threshold=0.3,
        language="vietnamese"
    )
    
    # Cập nhật state với kết quả RAG
    return {
        "rag_query_result": [rag_result["answer"]],
        "source_gathered": [source["filename"] for source in rag_result["sources"]],
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

    llm = ChatOpenAI(
        model=reasoning_model,
        base_url="http://localhost:1234/v1",  # Default LMStudio local server
        api_key="lm-studio",  # LMStudio doesn't require a real API key
        temperature=0.7,
    )

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
    formatted_prompt = answer_instructions.format(
        research_topic=get_research_topic(state["user_messages"]),
        summaries="\n\n---\n\n".join(state["rag_query_result"]),
        rag_loop_count=state.get("rag_loop_count", 0)
    )

    llm = ChatOpenAI(
        model=reasoning_model,
        base_url="http://localhost:1234/v1",  # Default LMStudio local server
        api_key="lm-studio",  # LMStudio doesn't require a real API key
        temperature=0.7,
    )

    result = llm.invoke(formatted_prompt)

    return {
        "user_messages": [AIMessage(result.content)]
    }

builder = StateGraph(OverallState, config_schema=Configuration)


builder.add_node("generate_query", generate_query)
builder.add_node("rag_research", rag_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)


builder.add_edge (START, "generate_query")

builder.add_conditional_edges("generate_query", continue_rag_process, ["rag_research"])

builder.add_edge("rag_research", "reflection")

builder.add_conditional_edges("reflection", evaluate_research, ["finalize_answer", "rag_research"])

builder.add_edge("finalize_answer", END)

graph = builder.compile(name="agentic-rag")