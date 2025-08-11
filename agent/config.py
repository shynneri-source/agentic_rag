import os 
from pydantic import BaseModel, Field
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig


model = "qwen/qwen3-4b"

class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default=model,
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default=model,
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    rag_model: str = Field(
        default=model,
        metadata={
            "description": "The name of the language model to use for the agent's RAG."
        },
    )

    answer_model: str = Field(
        default=model,
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    max_rag_loops: int = Field(
        default=4,
        metadata={"description": "The maximum number of RAG loops to perform."},
    )
    
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)


