from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    is_sufficient: bool = Field(
        default=False,
        description="Indicates whether the reflection is sufficient."
    )
    knowledge_gap: str = Field(
        default="",
        description="Describes the knowledge gap identified during reflection."
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="A list of follow-up queries generated during reflection."
    )

class rag_query_list(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )

class FinalAnswer(BaseModel):
    content: str = Field(
        description="The final answer content"
    )
    summary: str = Field(
        description="A brief summary of the research process"
    )