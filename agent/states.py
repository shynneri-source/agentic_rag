from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass, field
from langgraph.graph import add_messages
import operator

class OverallState(TypedDict):
    user_messages: Annotated[list, add_messages]
    rag_query: Annotated[list, operator.add]
    rag_query_result: Annotated[list, operator.add]
    source_gathered: Annotated[list, operator.add]
    initial_rag_query_count: int
    max_rag_loops: int
    rag_loop_count: int
    reasoning_model: str

class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    rag_loop_count: int
    number_of_rag_queries: int

class Query(TypedDict):
    query: str
    rationale: str

class QueryGenerationState(TypedDict):
    rag_query: list[Query]

class rag_query_state(TypedDict):
    rag_query: str
    id: str

@dataclass(kw_only=True)
class RagStateOutput:
    running_summary: str = field(default=None)  # Final report

