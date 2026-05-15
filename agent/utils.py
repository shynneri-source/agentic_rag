from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(user_messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    If there's conversation history (multiple messages), include the full context.
    The latest user message is treated as the current question.
    """
    if len(user_messages) == 1:
        # Single message — no history
        return user_messages[-1].content
    else:
        # Build a formatted conversation history string
        # Extract the last user message as the current question
        parts = []
        has_history = len(user_messages) > 2  # more than 1 exchange

        if has_history:
            parts.append("<conversation_history>")
            for msg in user_messages[:-2]:  # all but the last exchange
                if isinstance(msg, HumanMessage):
                    parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    parts.append(f"Assistant: {msg.content}")
            parts.append("</conversation_history>")

        # The most recent user message is the current question
        # Find the last HumanMessage
        current_question = ""
        for msg in reversed(user_messages):
            if isinstance(msg, HumanMessage):
                current_question = msg.content
                break

        if has_history:
            # Include the immediate previous exchange for context
            parts.append("<last_exchange>")
            for msg in user_messages[-2:]:
                if isinstance(msg, HumanMessage):
                    parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    parts.append(f"Assistant: {msg.content}")
            parts.append("</last_exchange>")

        parts.append(f"<current_question>{current_question}</current_question>")

        return "\n".join(parts)


def get_plain_conversation_history(user_messages: List[AnyMessage]) -> str:
    """Get a plain text summary of the conversation history (no tags)."""
    parts = []
    for msg in user_messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
    return "\n".join(parts)


def get_latest_question(user_messages: List[AnyMessage]) -> str:
    """Extract just the latest user question from messages."""
    for msg in reversed(user_messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""