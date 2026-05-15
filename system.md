# Agentic RAG System — NLP & System Architecture

## 1. System Overview

This system implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline using a **LangGraph-based state machine** orchestrated by a **Qwen3.5-4B** language model. The system processes Vietnamese-language questions, determines intent, iteratively searches a vector database of documents, reflects on search results, and generates final answers.

**Tech Stack:**
- **Orchestration:** LangGraph (StateGraph with conditional edges + Send fan-out)
- **LLM:** Qwen3.5-4B-Q4_K_M (quantized 4-bit) served via LMStudio-compatible API (configurable via `LLM_BASE_URL` in `.env`)
- **Embeddings:** `Qwen/Qwen3-Embedding-0.6B` (SentenceTransformer, 1024-dim)
- **Vector DB:** Qdrant (local, port 6333, cosine distance)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **API:** FastAPI with Server-Sent Events (SSE) streaming
- **Frontend:** Single-page HTML chat interface

---

## 2. Natural Language Processing (NLP) Components

### 2.1 Document Chunking (NLP Preprocessing)

**Module:** `rag/documents_processing.py` — `DocumentChunker` class

**What:** Raw Vietnamese documents (`.txt` files) are split into semantically coherent chunks for embedding and retrieval.

**How:**
- Uses `RecursiveCharacterTextSplitter` from LangChain
- Vietnamese-optimized separator hierarchy: `\n\n` (paragraph) → `\n` (line) → `. ` (sentence) → `.` → ` ` (word) → `` (character)
- Chunk size: **1000 characters**, overlap: **200 characters** (20% overlap ensures boundary context preservation)
- Each chunk gets metadata: `chunk_id`, `filename`, `source`, `document_index`, `chunk_index`

**NLP Rationale:**
Vietnamese is a **syllable-tone language** where words are composed of space-separated syllables (e.g., "thanh niên" = 2 syllables, 1 word). The character-level splitter handles Vietnamese adequately since the `RecursiveCharacterTextSplitter` recursively breaks at natural boundaries. The 200-character overlap prevents information loss across chunk boundaries, particularly important for Vietnamese articles where key dates/names may span sentence boundaries.

---

### 2.2 Text Embedding (Semantic Representation)

**Module:** `rag/documents_embedding.py` — `DocumentEmbeddingManager` class / `core/model.py` — `ModelManager` singleton

**What:** Converts Vietnamese document chunks and user queries into dense vector representations for semantic similarity search.

**Model:** `Qwen/Qwen3-Embedding-0.6B`
- Architecture: Transformer-based encoder (600M parameters)
- Output dimension: **1024**
- Auto device selection: CUDA > MPS > CPU

**How:**
1. Chunk texts are passed through the embedding model in batches (batch_size=32)
2. Output vectors are normalized to unit length for cosine similarity
3. Embeddings are stored in Qdrant as `PointStruct` objects alongside their payload (content + metadata)
4. Query-side: `search_similar_documents()` generates a query embedding on-the-fly and searches Qdrant

**NLP Rationale:**
Dense retrieval using bi-encoder transformers significantly outperforms keyword-based methods for Vietnamese, where morphological variants and synonyms ("sinh nhật" vs. "ngày sinh") would be missed by exact-term matching. The 1024-dim space captures semantic relationships — articles about "Bắc Giang thanh niên tình nguyện" and "Bắc Giang lễ kỷ niệm" will have similar vectors even without shared exact terms.

---

### 2.3 Semantic Search (Vector Retrieval)

**Module:** `core/model.py` — `ModelManager.search_similar_documents()`

**What:** Retrieves document chunks most semantically similar to a query from the Qdrant vector database.

**How:**
1. Query is embedded using the same SentenceTransformer model
2. Qdrant performs **cosine distance** search: `cosine_similarity(A, B) = (A·B) / (|A|·|B|)`
3. Results are filtered by `score_threshold=0.35` and limited to `limit=1` documents per query
4. Raw document content is returned (not LLM-generated summaries)

**NLP Rationale:**
Cosine similarity in embedding space captures semantic relatedness even with zero lexical overlap. A query about "lễ kỷ niệm thanh niên" will match documents about "chiến dịch tình nguyện" because their embeddings are close in the 1024-dim space. The 0.35 threshold acts as a **relevance floor** to prevent garbage retrieval, while limit=1 prevents confusion from multi-document conflicts (a known issue with small LLMs).

---

### 2.4 Keyword Relevance Filter (Post-Retrieval NLP Validation)

**Module:** `agent/agent.py` — `_is_relevant()`

**What:** A lexical overlap filter that validates whether a retrieved document actually contains content related to the query. This guards against **semantic false positives** — documents that are vector-close but topically unrelated.

**How:**
1. Extract all alphanumeric tokens from the query (`\w+` regex)
2. Remove Vietnamese stop words (`là`, `của`, `và`, `có`, etc.) and tokens < 3 characters
3. Check if any remaining token appears in the document content (case-insensitive substring match)
4. If no overlap, the document is rejected as "not found"

**NLP Rationale:**
Embedding-based search can return false positives due to coincidental semantic proximity (e.g., English "clear" retrieving Vietnamese "Hà Nội sạch" because "clear" and "sạch" share semantic space). This simple **keyword cross-check** compensates for the model's inability to verify topical relevance — a critical safeguard since the 4B model reliably hallucinates when given irrelevant context.

---

### 2.5 Intent Classification (Router — LLM-based NLU)

**Module:** `agent/agent.py` — `router_node()` / `agent/prompt.py` — `router_instructions`

**What:** Classifies user input as either `"rag"` (needs document retrieval) or `"chat"` (general conversation, commands, typos, or general knowledge).

**How:**
1. User message is formatted into `router_instructions` prompt
2. LLM is invoked with a zero-shot classification instruction
3. Response is extracted and lowercased; if `"rag"` is present, route to RAG pipeline; else route to chat
4. The prompt includes explicit examples for both categories

**Prompt design:**
```
chat = general questions, word definitions, commands, typos, exclamations, 
      chitchat, common knowledge, anything not needing specific documents
rag  = questions needing specialized documents, legal texts, specific events,
      statistical data, information only in the knowledge base

When in doubt, prefer "chat".
```

**NLP Rationale:**
This is a **zero-shot text classification** task. The LLM acts as a semantic router based on its pre-training knowledge of what constitutes "document-worthy" vs. "common-sense" questions. The few-shot examples in the prompt shape the decision boundary. The default bias toward "chat" (vs. the original "rag") prevents wasted RAG cycles on trivial queries.

---

### 2.6 Query Generation (LLM-based Query Expansion)

**Module:** `agent/agent.py` — `generate_query()` / `agent/prompt.py` — `query_writer_instructions`

**What:** Generates search queries for the vector database from the user's original question. Implements **query expansion** and **query reformulation** to improve retrieval coverage.

**How:**
1. Uses **structured output** via `llm.with_structured_output(rag_query_list)` enforcing a Pydantic schema
2. The prompt receives `research_topic` (the user's question) and `rag_loop_count`
3. First pass (`rag_loop_count=0`): generates 1-2 general queries
4. Subsequent passes: generates more specific, targeted queries
5. Output schema: `{"rationale": str, "query": List[str]}`

**NLP Rationale:**
Query expansion addresses the **vocabulary mismatch problem** — the user's question may not contain the exact terms used in relevant documents. By reformulating the same question multiple ways (general → specific), the system casts a wider semantic net. The structured output ensures parseable, valid query lists.

---

### 2.7 Reflection & Evaluation (LLM-based Self-Assessment)

**Module:** `agent/agent.py` — `reflection()` / `agent/prompt.py` — `reflection_instructions`

**What:** Analyzes all retrieved document content to determine whether sufficient information exists to answer the user's question. This is a **self-evaluation / meta-cognition** step.

**How:**
1. All accumulated `rag_query_result` entries are joined with `\n\n---\n\n` separators
2. The LLM is called with **structured output** (`Reflection` schema):
   - `is_sufficient`: boolean — whether current docs answer the question
   - `knowledge_gap`: string — what's missing
   - `follow_up_queries`: List[str] — new queries to fill the gap
3. The state's `rag_loop_count` is incremented
4. The result drives the conditional edge: if sufficient or max loops reached → finalize; else → search again

**NLP Rationale:**
This implements a **multi-hop reasoning loop** where the LLM evaluates the sufficiency of its own retrieved evidence. The model must:
1. **Extract key information** (dates, numbers, names) from raw document content
2. **Compare** extracted info against the original question
3. **Identify gaps** — what specific information is still missing
4. **Generate targeted queries** to fill those gaps

The structured output ensures the evaluation is parseable by the state machine. The max-loop guard prevents infinite cycles, important because the 4B model tends to be conservative (often deeming information insufficient even when present).

---

### 2.8 Answer Generation (LLM-based Text Generation)

**Module:** `agent/agent.py` — `finalize_answer()` / `agent/prompt.py` — `answer_instructions`

**What:** Generates the final Vietnamese answer from all accumulated document content.

**How:**
1. All document content in `rag_query_result` is joined and passed as context
2. LLM with structured output (`FinalAnswer` schema) generates:
   - `content`: the answer text (Vietnamese, with source citations)
   - `summary`: brief description of the research process
3. The prompt emphasizes: extract exact dates/names/numbers, cite sources as `[Nguồn X]`, only use provided content

**NLP Rationale:**
This is a **constrained text generation** task. The model must produce fluent Vietnamese while being strictly **faithful** to the retrieved documents. The structured output (`FinalAnswer`) decouples the answer from the summary, enabling the streaming frontend to display either.

---

## 3. Agentic Workflow (LangGraph State Machine)

### 3.1 State Graph Structure

```
START → [router] → conditional
    ├── "chat" → [chat] → END
    └── "rag" → [generate_query] → conditional
        ├── "chat" (0 queries) → [chat] → END
        └── "rag_research" (fan-out via Send) → [reflection] → conditional
            ├── "finalize_answer" (sufficient OR max loops) → END
            └── "rag_research" (follow-up queries via Send) → [reflection] → ...
```

### 3.2 State Types (TypedDict with Reducers)

**`OverallState`:** Central state with LangGraph reducers:
- `user_messages`: `Annotated[list, add_messages]` — accumulates chat history
- `rag_query`: `Annotated[list, operator.add]` — all queries ever generated
- `rag_query_result`: `Annotated[list, operator.add]` — all document contents retrieved
- `source_gathered`: `Annotated[list, operator.add]` — all source filenames
- Control fields: `intent`, `rag_loop_count`, `max_rag_loops`, `initial_rag_query_count`

**`QueryGenerationState`:** `rag_query: list[Query]` — queries from generate_query

**`rag_query_state`:** `rag_query: str, id: str` — individual query sent to rag_research via Send

**`ReflectionState`:** `is_sufficient`, `knowledge_gap`, `follow_up_queries`, `rag_loop_count`, `number_of_rag_queries`

### 3.3 Fan-Out Pattern with Send()

The LangGraph `Send()` API enables **dynamic parallel execution**. When `generate_query` produces N queries, `continue_rag_process` creates N `Send("rag_research", ...)` objects, each with a different query string. These execute in parallel, each calling `rag_research` independently and contributing their results to the shared state via the `operator.add` reducer.

This pattern repeats in the reflection loop — each `follow_up_queries` list fans out to parallel rag_research calls.

---

## 4. Streaming Architecture (app.py)

The FastAPI `/api/chat/stream` endpoint uses **Server-Sent Events (SSE)** to stream agent progress in real-time:

| Event Type | Trigger Node | Content |
|---|---|---|
| `thinking` | Before any node | "Đang khởi tạo agent..." |
| `query_generation` | `generate_query` | List of generated queries |
| `rag_search` | `rag_research` | Actual query text + results count |
| `reflection` | `reflection` | Sufficient/insufficient + knowledge gap |
| `answer` | `finalize_answer` | Final answer text |
| `complete` | After END | Summary stats |
| `error` | Exception | Error message |

The frontend renders these events with icons (🔎, 🔄, ✅, ❌) and progressive disclosure.

---

## 5. Key NLP Design Decisions & Tradeoffs

### 5.1 Raw Content vs. LLM Summaries (RAG Research)

**Decision:** `rag_research` returns raw document content instead of LLM-generated summaries.

**Rationale:** The Qwen3.5-4B model frequently **hallucinates or misses key information** when summarizing. Raw content preserves all information — dates, names, numbers — for the reflection and finalization steps to process directly. The tradeoff is larger state size, but this is acceptable because only 1 document per query is returned (limit=1).

### 5.2 Single Document Per Query

**Decision:** `rag_research` searches with `limit=1` and `score_threshold=0.35`.

**Rationale:** When multiple documents are retrieved, the 4B model often picks the wrong one (e.g., choosing a national-event date over a local-event date). Returning only the **most relevant** document avoids this confusion. The agent compensates by running multiple queries per round and multiple rounds as needed.

### 5.3 Date Extraction via Regex

**Decision:** A regex pattern converts the raw document format `Nội dung: 09:25 13/10/20241155` into `Ngày đăng: 13/10/2024 - Thời gian: 09:25 | Nội dung:`.

**Rationale:** The model fails to parse dates embedded in format strings alongside word counts. Making the date explicit in the text triples the likelihood of correct extraction. This is a **prompt-engineering-by-data-cleaning** approach.

### 5.4 Keyword Relevance Filter

**Decision:** Post-retrieval filter checking query-document keyword overlap.

**Rationale:** The embedding model's semantic search can retrieve topically unrelated documents (e.g., "clear" → "Hà Nội sạch"). Since the 4B model can't reliably detect such mismatches (it hallucinates relevance), a simple lexical filter provides a hard guarantee against irrelevant context reaching the LLM.

### 5.5 Fallback Routing

**Decision:** When `generate_query` produces 0 queries (e.g., for commands, typos, or nonsensical input), route to `chat` instead of continuing to RAG.

**Rationale:** The LLM may be unable to generate search queries for non-questions. Rather than stalling the graph (previously: empty Send list → stuck), the agent falls back to a conversational response. Similarly, the router now defaults to "chat" when uncertain, preventing wasted RAG cycles.

---

## 6. Prompt Engineering Summary

| Prompt | Technique | Purpose |
|---|---|---|
| `router_instructions` | Zero-shot classification + few-shot examples | Route between chat and RAG |
| `query_writer_instructions` | Structured output (List[str]) + context-dependent generation | Generate search queries |
| `reflection_instructions` | Structured output (Reflection schema) + anti-hallucination rules | Evaluate information sufficiency |
| `answer_instructions` | Structured output (FinalAnswer schema) + citation requirement | Generate final answer |
| `chat_instructions` | /nothink directive + personality prompt | Conversational response |

**Note on `/nothink`:** All prompts (except router) begin with `/nothink`, a LMStudio directive that disables chain-of-thought reasoning, forcing the model to output directly. This reduces latency and token usage for structured tasks.

---

## 7. Model Configuration

**Qwen3.5-4B-Q4_K_M Quantization:**
- **Base:** Qwen3.5 (3.5 generation from Alibaba's Qwen family)
- **Size:** 4 billion parameters
- **Quantization:** Q4_K_M (4-bit k-quant, medium) — reduces memory footprint from ~8GB to ~2.5GB
- **Temperature:** 0.7 (balance between creativity and determinism)
- **API:** OpenAI-compatible via LMStudio (URL configured via `LLM_BASE_URL` in `.env`)

**Embedding Model:**
- **Model:** Qwen/Qwen3-Embedding-0.6B
- **Device:** Auto-detected (CUDA > MPS > CPU)
- **Dimension:** 1024
- **Distance metric:** Cosine (compatible with Qdrant's Distance.COSINE)

---

## 8. Configuration

Defined in `agent/config.py` and overridable via `app.py`:

| Parameter | Default | Description |
|---|---|---|
| `query_generator_model` | Qwen3.5-4B-Q4_K_M.gguf | Model for query generation |
| `reflection_model` | Qwen3.5-4B-Q4_K_M.gguf | Model for reflection/evaluation |
| `rag_model` | Qwen3.5-4B-Q4_K_M.gguf | Model for RAG (unused in current impl) |
| `answer_model` | Qwen3.5-4B-Q4_K_M.gguf | Model for final answer |
| `max_rag_loops` | 3 (app.py override) | Max RAG evaluation cycles |
| `number_of_initial_queries` | 2 (app.py override) | Initial queries per question |

---

## 9. Limitations & Known Issues

1. **Model capacity:** The 4B model struggles with multi-hop reasoning (distinguishing "national event date" from "local event date"). Solutions: `limit=1` retrieval + regex date cleaning.
2. **Hallucination tendency:** The model sometimes invents information when given irrelevant context. Solutions: Keyword relevance filter + anti-hallucination prompts.
3. **Vietnamese language:** No dedicated Vietnamese tokenizer/NER. Keyword splitting treats syllables as words, which works adequately but misses Vietnamese compound words.
4. **Single document retrieval:** `limit=1` trades recall for precision. Some questions might need multiple documents, compensated by multi-query rounds.
5. **Static threshold:** The 0.35 score threshold is fixed; queries scoring below this return nothing even if the desired document exists.
