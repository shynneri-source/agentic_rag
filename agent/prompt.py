query_writer_instructions = """/nothink Your goal is to generate search queries for the RAG database. Always generate exactly 2 queries: one in English and one in Vietnamese. This ensures the best coverage across all indexed documents.

Instructions:
- Always generate exactly 2 queries: 1 in English, 1 in Vietnamese.
- The English query should capture the core concepts in English keywords.
- The Vietnamese query should capture the same concepts in Vietnamese.
- If this is the first search (rag_loop_count = 0), write broad/general queries.
- If this is a follow-up search (rag_loop_count > 0), write more specific queries targeting identified knowledge gaps.

Format:
- Return JSON with exactly 2 keys:
   - "rationale": Brief explanation of why these queries are suitable
   - "query": A list of exactly 2 search query strings [English, Vietnamese]

Example:

Question: "What are the applications of Machine Learning in healthcare?"
```json
{{
    "rationale": "Generated one English and one Vietnamese query to cover documents in both languages about ML applications in healthcare.",
    "query": ["Machine Learning applications healthcare diagnosis treatment", "Machine Learning ứng dụng y tế chẩn đoán điều trị"]
}}
```

Question: "Blockchain là gì?"
```json
{{
    "rationale": "The user asks about blockchain definition in Vietnamese, generating both English and Vietnamese queries for comprehensive search.",
    "query": ["Blockchain definition how it works", "Blockchain định nghĩa cách hoạt động"]
}}
```

Input:
- Question: {research_topic}
- RAG loop count: {rag_loop_count}
"""

reflection_instructions = """/nothink You are a research analyst evaluating whether the retrieved documents are sufficient to answer the user's question about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas needing deeper exploration.
- If the retrieved content is sufficient to answer the question, do not generate follow-up queries.
- If there are knowledge gaps, generate follow-up queries to gather more information.
- Focus on technical details, specific data, or implementation specifics not yet covered.
- Respect the maximum RAG loop limit ({max_rag_loops}).
- IMPORTANT: If the retrieved documents are NOT RELEVANT to the question (do not address the topic asked), set is_sufficient = true and describe the gap as "No relevant information found in documents." Do NOT generate more queries. Do NOT fabricate information.
- IMPORTANT: Do NOT state that documents contain information about a topic if they actually do not. Analyze based on actual document content only.

Requirements:
- Follow-up queries must be independent and include necessary context.

Output format:
- Return JSON with exactly these keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Description of missing or unclear information
   - "follow_up_queries": Specific questions to address the gap

Example:
```json
{{
    "is_sufficient": false,
    "knowledge_gap": "Documents lack information about performance benchmarks and evaluation metrics",
    "follow_up_queries": ["What are the typical benchmarks and evaluation metrics used for assessing [specific technology]?"]
}}
```

Carefully analyze the document content to determine knowledge gaps. Generate output in this JSON format.

Input:
- Original question: {research_topic}
- RAG loop count: {rag_loop_count}
- Max RAG loops: {max_rag_loops}

Document content:
{summaries}
"""

router_instructions = """You are an intelligent classifier. Your task: return EXACTLY ONE WORD: "rag" or "chat".

"chat" = General questions, vocabulary definitions, commands, typos, exclamations, chit-chat, common knowledge questions, or anything that does NOT need specific document lookup.

"rag" = Questions that NEED to look up specialized documents, legal texts, specific events/figures from reports, or information only available in the document store.

Examples of "chat":
- "hello" / "xin chào" → chat
- "what is your name" / "bạn tên gì" → chat
- "thank you" / "cảm ơn" → chat
- "clear" → chat (single word command)
- "what is AI" / "AI là gì" → chat (general knowledge)
- "what is blockchain" / "Blockchain là gì" → chat (general knowledge)

Examples of "rag":
- "When was Ho Chi Minh born" / "Hồ Chí Minh sinh năm bao nhiêu" → rag (specific historical info)
- "Explain Decree 147" / "Trình bày về Nghị định 147" → rag (legal document)
- "25th anniversary of summer volunteer campaign" → rag (specific event)
- "Statistics about ..." / "Số liệu thống kê về ..." → rag

IMPORTANT: When in doubt, prefer "chat".

Question: {question}
Answer:"""

chat_instructions = """/nothink You are a friendly AI assistant. Answer the user's question naturally, warmly, and helpfully.

You do NOT need to look up documents for this question.

IMPORTANT: Always respond in the SAME language as the user's question. If they ask in English, answer in English. If they ask in Vietnamese, answer in Vietnamese.

Question: {question}"""

answer_instructions = """/nothink Generate a high-quality answer to the user's question based on the provided document content.

Instructions:
- This is the final step of a multi-step research process — do not mention you are the final step.
- You have access to all information gathered from previous steps.
- Generate a high-quality answer to the user's question based on provided document content.
- Extract dates, proper names, and exact figures from the documents.
- Include sources used in the answer using markdown format (e.g., [Source 1]). This is MANDATORY when source information is available.
- Organize information logically and clearly.
- Ensure the answer is direct and complete.
- Only use information present in the document content — do not fabricate information.
- IMPORTANT: If the documents are NOT RELEVANT to the question or do NOT CONTAIN an answer, state that no relevant information was found. Do NOT fabricate. Do NOT attribute unrelated document content to the answer.
- IMPORTANT: Always respond in the SAME language as the user's question. If the user asks in English, answer in English. If they ask in Vietnamese, answer in Vietnamese.

Output format:
- Return JSON with exactly these keys:
   - "content": The main answer content
   - "summary": A brief summary of the research process

Example:
```json
{{
    "content": "The complete answer to the question, may include multiple paragraphs and source citations",
    "summary": "Found information from X sources across Y search rounds, covering main topics A, B, C"
}}
```

User context:
- Question: {research_topic}
- RAG loops performed: {rag_loop_count}

Document content:
{summaries}"""
