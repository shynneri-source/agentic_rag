# Agentic RAG — Quick Start Guide

## System Requirements

- macOS (Apple Silicon or Intel) / Linux
- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) (package manager, recommended) or pip

## 1. Install Dependencies

```bash
git clone https://github.com/shynneri-source/agentic_rag.git
cd agentic_rag
uv sync  # or: pip install -r requirements.txt
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain`, `langchain-core`, `langchain-openai` | LLM communication via OpenAI-compatible API |
| `langchain-text-splitters` | Text chunking |
| `langgraph` | Agent workflow (state machine) |
| `sentence-transformers` | Embedding model |
| `qdrant-client` | Vector database client |
| `fastapi`, `uvicorn` | Web server |
| `pydantic` | Data validation |

## 2. Start External Services

### 2a. LLM Server

The system connects to an OpenAI-compatible API server. Configure via `.env`:

```env
LLM_BASE_URL=http://localhost:8000/v1    # Your LLM server URL
LLM_API_KEY=not-needed                     # API key if required
LLM_MODEL=Qwen3.5-4B-Q4_K_M.gguf          # Model name
LLM_TEMPERATURE=0.7                        # Sampling temperature
```

Copy the template and edit with your settings:

```bash
cp .env.example .env
# Edit .env with your LLM server details
```

### 2b. Qdrant Vector Database

Run Qdrant via Docker:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Alternatively, download the binary from [qdrant.tech](https://qdrant.tech/documentation/guides/installation/).

## 3. Prepare Data

### Step 1: Place documents in `documents/`

```bash
# Default: documents/data.txt
# Supported: .txt, .md, .text
ls documents/
```

### Step 2: Chunk documents

Split documents into smaller segments:

```bash
uv run python rag/documents_processing.py
```

Output: `documents_chunks.json`

### Step 3: Generate embeddings and upload to Qdrant

```bash
uv run python rag/documents_embedding.py
```

This script will:
- Read chunks from `documents_chunks.json`
- Generate embeddings using `Qwen/Qwen3-Embedding-0.6B`
- Upsert into Qdrant collection `document_embeddings`
- Display collection statistics

## 4. Run the System

### 4a. Web Interface (recommended)

```bash
uv run python app.py
```

Access at: http://localhost:3000

API endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/chat` | POST | Synchronous chat |
| `/api/chat/stream` | POST | Streaming chat (SSE) |
| `/api/chat/sessions?session_id=...` | GET | List conversations |
| `/api/chat/history?conversation_id=...` | GET | Get conversation messages |
| `/api/chat/sessions/{conversation_id}` | DELETE | Delete conversation |
| `/api/chat/history?session_id=...` | DELETE | Clear all conversations |
| `/api/memories?session_id=...` | GET | List long-term memories |
| `/api/memories/search?session_id=...&query=...` | GET | Search memories |
| `/api/memories/{memory_id}` | DELETE | Delete a memory |
| `/api/memories?session_id=...` | DELETE | Clear all memories |
| `/api/health` | GET | Health check |

### 4b. Interactive CLI

```bash
# Interactive mode
uv run python example.py

# Single question
uv run python example.py "Your question here"
```

### 4c. Batch Processing

Process multiple questions from a JSON file:

```bash
uv run python process_questions.py
```

Input: `datatest/test.json` (array of question objects)
Output: `agent_results.json`, `agent_results_intermediate.json`

### 4d. Direct RAG Test (bypasses agent)

```bash
uv run python rag/rag_example.py
```

## 5. Agent Workflow Architecture

```
START → router (LLM intent classification)
         │
         ├── "chat" → direct response (no RAG) → END
         │
         └── "rag" → generate_query → [Send] → rag_research → reflection → evaluate_research
                                                                              │
                                                          ┌────────────────────┘
                                                          ▼ (if insufficient)
                                                    [Send] → rag_research ...
                                                          │
                                                          ▼ (if sufficient or max loops)
                                                    finalize_answer → END
```

## 6. Configuration

### Environment Variables (override defaults)

| Variable | Description | Default |
|----------|-------------|---------|
| `QUERY_GENERATOR_MODEL` | Query generation model | `Qwen3.5-4B-Q4_K_M.gguf` |
| `REFLECTION_MODEL` | Evaluation model | `Qwen3.5-4B-Q4_K_M.gguf` |
| `RAG_MODEL` | RAG model | `Qwen3.5-4B-Q4_K_M.gguf` |
| `ANSWER_MODEL` | Answer generation model | `Qwen3.5-4B-Q4_K_M.gguf` |
| `MAX_RAG_LOOPS` | Max RAG evaluation cycles | 4 |
| `NUMBER_OF_INITIAL_QUERIES` | Initial search queries | 3 |

### Key Configuration Files

| File | Configuration |
|------|---------------|
| `.env` | LLM URL, API key, model, Qdrant host/port, embedding model |
| `core/model.py` | Reads automatically from `.env` |
| `agent/config.py` | LangGraph configuration schema |
| `store/conversation_store.py` | SQLite DB path, table schema |
| `store/memory_store.py` | Qdrant memory collection name, embedding dimension |
| `store/memory_extractor.py` | Memory extraction prompt, importance threshold |
| `rag/documents_processing.py` | Chunk size, overlap, separators |
| `rag/documents_embedding.py` | Qdrant config, batch sizes |

### Default Chunk Parameters

| Parameter | Value |
|-----------|-------|
| `chunk_size` | 1000 characters |
| `chunk_overlap` | 200 characters |
| `separators` | `["\n\n", "\n", ". ", ".", " ", ""]` |

### Default Qdrant Parameters

| Parameter | Value |
|-----------|-------|
| Collection | `document_embeddings` |
| Distance | Cosine |
| Embedding dimension | 1024 |
| Batch size (embed) | 32 |
| Batch size (upsert) | 100 |

## 7. Notes

- **macOS GPU**: System auto-detects MPS on Apple Silicon. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if you encounter MPS errors (already set in code).
- **Python 3.14**: Project requires Python >= 3.14.
- **No GPU**: Embedding model falls back to CPU if CUDA or MPS is unavailable.
- **Qdrant must be running**: All embedding and agent scripts require Qdrant.
- **LLM Server must be running**: The agent requires a running OpenAI-compatible API server.

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `uv sync` or `pip install -r requirements.txt` |
| Cannot connect to Qdrant | Check `docker ps` or run `docker start qdrant` |
| LLM returns errors | Verify LLM server is running and URL is correct in `.env` |
| CUDA errors | System auto-detects device, falls back to CPU |
| `nan` embeddings on macOS | `attn_implementation="eager"` is configured to avoid SDPA errors |
| Memory extraction slow | Memory extraction runs as background task; does not affect response time |
| Memories not appearing | Check that `conversation_memories` collection exists in Qdrant |
| Conversation lost after restart | SQLite DB persists in `data/conversations.db`; ensure directory is writable |
