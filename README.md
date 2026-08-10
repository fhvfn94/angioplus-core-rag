# AngioPlus Core AI Support Assistant

RAG-based AI assistant for technical support of the AngioPlus Core medical software.

The system answers questions strictly from indexed product documentation and is intended to support L1/L2 technical support workflows.

## Current Status

The MVP is operational.

Implemented:

- Telegram bot
- text questions
- voice questions
- PDF / DOCX / XLSX ingestion
- local multilingual embeddings
- Qdrant vector search
- DeepSeek answer generation
- query normalization
- short conversation memory
- follow-up question rewriting
- source metadata
- citation of chunks actually used by the LLM
- secret filtering
- fallback when documentation does not contain the answer

## Architecture

### Ingestion

documents
→ parsing
→ structured chunking
→ metadata
→ BAAI/bge-m3 embeddings
→ Qdrant

### Runtime

Telegram message
→ query normalization
→ optional conversation-context rewrite
→ BAAI/bge-m3 query embedding
→ Qdrant retrieval
→ direct-answer gate
→ DeepSeek generation
→ source citation extraction
→ secret/output safety filter
→ Telegram response

### Voice

Telegram voice
→ Gemini audio transcription
→ text query
→ normal RAG pipeline

Voice transcription may later be migrated to a local Whisper/faster-whisper implementation.

## Technology Stack

- Python 3.12
- FastAPI
- aiogram
- Docker Compose
- Qdrant
- sentence-transformers
- BAAI/bge-m3
- DeepSeek API
- OpenAI-compatible Python SDK for DeepSeek API access
- Google Gemini API for voice transcription

Important:

The OpenAI Python package is currently used only as an OpenAI-compatible client for DeepSeek.

Requests are sent to:

https://api.deepseek.com

The project does NOT currently use OpenAI models or the OpenAI API.

## Embeddings

Provider:

sentence_transformers

Model:

BAAI/bge-m3

Dimension:

1024

Distance:

COSINE

Device:

CPU

Qdrant collection:

angioplus_documents_bge_m3

Current indexed dataset:

6 documents
118 chunks

## LLM

Provider:

DeepSeek

Current model:

deepseek-v4-flash

The LLM receives only retrieved and sanitized document context.

If the available context does not answer the question, the assistant must respond:

"Такой информации нет в имеющейся документации."

## Retrieval

Current retrieval depth:

top_k = 10

This value was increased from 5 after regression testing showed that a correct IFU chunk could appear at position 6.

Example:

Question:
"Где посмотреть версию программного обеспечения?"

Relevant IFU chunk:
position 6

With top_k=5 the answer was missed.
With top_k=10 the correct answer is retrieved.

Do not reduce top_k without regression testing.

## Source Priority

Source authority:

1. IFU / official regulatory documentation
2. Official service and administrator documentation
3. Release notes / known issues
4. L1/L2 internal support materials
5. FAQ / training materials
6. Commercial materials

If IFU conflicts with another source, IFU wins.

FAQ and training materials may supplement IFU but must not override it.

## Source Citations

The LLM receives numbered retrieved chunks:

[Chunk 1]
[Chunk 2]
...

For successful answers it returns an internal marker:

[[USED_CHUNKS: 2, 6]]

The application removes this marker before sending the response to the user.

Only metadata from the cited chunks is shown in the Telegram "Источники" section.

This prevents unrelated top-ranked Qdrant results from being presented as answer sources.

## Conversation Context

Conversation context is enabled.

Current configuration:

CONVERSATION_TTL_SECONDS=1800
CONVERSATION_MAX_TURNS=3

The system can rewrite follow-up questions into standalone retrieval questions.

Example:

User:
"Кто может устанавливать AngioPlus Core?"

Follow-up:
"А кто может его обслуживать?"

The second question may use recent conversation context to resolve the reference.

## Query Normalization

Query normalization is enabled before embedding and retrieval.

The original user question and the normalized/standalone versions are tracked separately in logs.

## Safety

The assistant must:

- answer only from retrieved documentation
- never invent missing procedures
- never expose passwords, API keys, tokens or credentials
- distinguish mandatory IFU requirements from recommendations
- distinguish "not evaluated" from "prohibited"
- return a documentation-not-found response when direct support is absent

Secrets are sanitized before context is sent to external LLM providers.

Generated output also passes through a secret filter.

## Documents

Current knowledge base includes:

- AngioPlus Core v2.5 IFU_RU_Final.pdf
- 2025July.pdf
- Common Info to Distributor - commercial (1).pdf
- Competitor_2025.pdf
- Development_Questions_L1_L2.docx
- Q&A List ENG.xlsx

## Qdrant

Current collection:

angioplus_documents_bge_m3

Current state after ingestion:

points = 118
status = green
vector size = 1024
distance = COSINE

## Main Environment Configuration

Secrets must never be committed.

Important configuration:

EMBEDDING_PROVIDER=sentence_transformers
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=4

QDRANT_COLLECTION=angioplus_documents_bge_m3

LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash

QUERY_NORMALIZATION_ENABLED=true
QUERY_CONTEXT_ENABLED=true

CONVERSATION_TTL_SECONDS=1800
CONVERSATION_MAX_TURNS=3

QUERY_LOW_SCORE_LOG_THRESHOLD=0.45

## Development Workflow

Development should normally be performed locally.

Recommended flow:

local VS Code
→ implementation
→ tests
→ git commit
→ git push origin main
→ server git pull --ff-only origin main
→ rebuild/restart affected Docker services
→ production smoke test

Avoid editing production server code directly except for emergency diagnostics.

## Current Known Issues / Improvements

### Response latency

Typical request:

- local embedding: ~0.5–1.0 s
- Qdrant: ~0.1 s
- DeepSeek gate: ~1–2 s
- DeepSeek generation: ~3–8 s

Large answers may take significantly longer.

The direct-answer gate currently causes a second DeepSeek request.

A future optimization may merge gate + generation into one LLM request after regression testing proves grounding remains safe.

### Voice transcription

Voice currently depends on Gemini.

Possible future migration:

faster-whisper / whisper.cpp

This would remove an external dependency and allow local speech recognition.

### Retrieval

Dense BGE-M3 retrieval works, but further testing may justify:

- hybrid dense + lexical search
- reranking
- query expansion
- metadata-aware retrieval

Do not implement these without measuring retrieval failures first.

## Development Principle

Accuracy > creativity.

Do not change architecture because an alternative sounds theoretically better.

Every retrieval, safety or prompt change must be validated against regression questions.