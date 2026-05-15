"""
LLM-based memory extraction from conversation exchanges.
After each conversation turn, extracts salient facts worth remembering
across sessions (user preferences, personal info, decisions, etc.).
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.model import get_llm


class MemoryItem(BaseModel):
    """A single memory extracted from a conversation exchange."""
    content: str = Field(description="The memory fact to remember (e.g., 'User enjoys Python for data analysis')")
    memory_type: str = Field(description="Type: 'personality', 'communication_style', 'preference', 'hobby', 'interest', 'goal', 'fact', or 'other'")
    importance: int = Field(description="Importance 1-5 (5=very important to remember, 3=moderately useful, 1=trivial)")


class MemoryExtraction(BaseModel):
    """Structured memory extraction result."""
    memories: List[MemoryItem] = Field(description="Extracted memories from the conversation exchange")


MEMORY_EXTRACTION_PROMPT = """You are an AI assistant that selectively extracts meaningful memories from conversations.
Your goal is to remember what makes this user unique — their personality, communication style,
hobbies, passions, goals, and preferences — and NOT waste space on trivial chat content.

Given a user message and your response, decide if anything is WORTH remembering.

## WHAT TO EXTRACT (only if genuinely present):
1. **Personality**: User's character traits, attitudes, sense of humor, values (e.g., "User is detail-oriented and enjoys deep technical discussions")
2. **Communication style**: How the user likes to communicate — formal/casual, direct/elaborate, prefers bullet points vs paragraphs (e.g., "User prefers concise, direct answers with examples")
3. **Hobbies & Interests**: Things the user enjoys doing, topics they're passionate about (e.g., "User is passionate about woodworking and DIY projects")
4. **Preferences**: Stated preferences about tools, languages, formats, style (e.g., "User prefers Python over JavaScript for backend work")
5. **Goals & Projects**: What the user is working on or trying to achieve (e.g., "User is building a RAG system with LangGraph")
6. **Personal context**: Important facts about background, profession, location, skills

## WHAT NOT TO EXTRACT (return empty list):
- Greetings, hello/goodbye, "how are you" exchanges
- Thanks and pleasantries ("thank you", "you're welcome", "have a nice day")
- Generic questions ("what is AI?", "explain X") — these are one-time queries, not memories
- Clarification questions ("can you elaborate?", "what do you mean?")
- Single-word or very short exchanges
- Content that is already obvious or generic

## Guidelines:
- Be VERY selective. Most exchanges should return NO memories.
- A memory is only worth saving if it would genuinely help personalize future conversations.
- Rate importance: 5 = must remember (core identity), 4 = very useful, 3 = nice context, 2 = minor detail, 1 = trash
- Keep memory content concise (under 120 chars).
- For communication style: infer from HOW the user writes, not what they say.

User message: {user_message}

Assistant response: {assistant_response}

Extract memories from this exchange (return empty list if nothing worth remembering).
"""


class MemoryExtractor:
    """Extracts memories from conversation exchanges using the LLM."""

    def __init__(self):
        self.llm = get_llm()
        self.structured_llm = self.llm.with_structured_output(MemoryExtraction)

    def extract(
        self,
        user_message: str,
        assistant_response: str,
    ) -> List[MemoryItem]:
        """Extract memories from a single conversation exchange."""
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        try:
            result: MemoryExtraction = self.structured_llm.invoke(prompt)
            # Filter to only meaningful memories (importance >= 3) and limit
            memories = [
                m for m in result.memories
                if m.importance >= 3 and m.content.strip()
            ]
            return memories[:3]  # max 3 memories per exchange (be selective)
        except Exception as e:
            print(f"Memory extraction error: {e}")
            return []


# Singleton
_extractor_instance: Optional[MemoryExtractor] = None


def get_memory_extractor() -> MemoryExtractor:
    """Get or create the singleton memory extractor."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MemoryExtractor()
    return _extractor_instance
