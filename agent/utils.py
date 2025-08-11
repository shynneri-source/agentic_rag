from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(user_messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(user_messages) == 1:
        research_topic = user_messages[-1].content
    else:
        research_topic = ""
        for user_message in user_messages:
            if isinstance(user_message, HumanMessage):
                research_topic += f"User: {user_message.content}\n"
            elif isinstance(user_message, AIMessage):
                research_topic += f"Assistant: {user_message.content}\n"
    return research_topic