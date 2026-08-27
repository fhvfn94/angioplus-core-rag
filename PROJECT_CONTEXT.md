# PROJECT CONTEXT — AngioPlus Core RAG

Last updated: 2026-08

## Purpose

This file is the compact current-state memory for AI coding agents.

Before making architectural changes:

1. Read this file.
2. Read README.md.
3. Inspect current source code.
4. Run `git status`.
5. Do not assume older architecture still exists.

## Current State

The AngioPlus Core RAG MVP is running in production on a Docker server.

Main services:

- bot
- rag
- qdrant

Telegram bot is operational.

## Current Production Architecture

Text request:

Telegram
→ bot
→ FastAPI /ask
→ query normalization
→ optional conversation rewrite
→ local BAAI/bge-m3 embedding
→ Qdrant search
→ DeepSeek direct-answer gate
→ DeepSeek answer generation
→ USED_CHUNKS parsing
→ source filtering
→ output safety filter
→ Telegram

Voice:

Telegram voice
→ Gemini transcription
→ same text RAG pipeline

## Embeddings

Provider:
sentence_transformers

Model:
BAAI/bge-m3

Device:
cpu

Dimension:
1024

Normalize embeddings:
true

Collection:
angioplus_documents_bge_m3

Distance:
COSINE

Current points:
118

## Retrieval

Current top_k:

10

Do NOT revert to 5.

Reason:

Regression test:

"Где посмотреть версию программного обеспечения?"

Relevant IFU chunk ranked #6.

top_k=5 failed.
top_k=10 succeeded.

## Source Citation System

Retrieved context is numbered by chunk.

LLM is instructed to append:

[[USED_CHUNKS: x, y]]

`app/main.py` parses the marker.

The marker is removed before user output.

`build_sources()` receives only cited chunks when citations are available.

This was implemented because previous source display simply showed the first three Qdrant results, which could be unrelated to the generated answer.

## Source Priority

Authority:

IFU
> official technical docs
> L1/L2
> FAQ/training
> commercial

IFU overrides lower-priority sources.

Do not automatically boost all IFU chunks independent of semantic relevance.

## LLM

Provider:
DeepSeek

Model:
deepseek-v4-flash

API:
https://api.deepseek.com

Python package:
openai

Important:

The project is NOT using OpenAI's API.

The `openai` Python SDK is used because DeepSeek exposes an OpenAI-compatible API.

## Gate

The system currently performs two external LLM calls for successful questions:

1. direct-answer gate
2. answer generation

This improves grounding but increases latency.

Do not remove the gate without regression tests.

Possible future optimization:
single-call answer/refusal generation.

## Query Context

Enabled:

QUERY_CONTEXT_ENABLED=true

TTL:
1800 seconds

Max turns:
3

Query normalization:
enabled

Follow-up rewriting exists.

Needs additional regression testing.

## Voice

### PRODUCTION (current live path)

Transcription:

Telegram voice -> Gemini transcription -> existing text RAG pipeline

Configured with:

GEMINI_AUDIO_MODEL=gemini-3.5-flash

Server resource limits must be evaluated before adding another ML model because BGE-M3 already consumes significant RAM.

### DEVELOPMENT BRANCH (feature/livekit-voice-agent, phases D1.x / D2.1)

Real microphone -> LiveKit Room -> Silero VAD -> faster-whisper -> transcript
-> RAG /ask -> answer published back to the room (as text/data).

Phase D2.1 connected the transcribed utterance to the existing RAG response
path: each FINAL_TRANSCRIPT is sent to POST /ask (via voice.rag_client) and the
answer is published back to the speaking participant as a data message. The
text data-message path and the voice path share one `_answer_and_publish`
helper. `AudioTrackListener` gained an `on_final(text, participant)` session
callback so the agent can attribute each spoken utterance to its caller.

- Transcript IS now connected to RAG (voice -> /ask -> answer, delivered as
  text/data). TTS is NOT yet implemented; the answer is not spoken yet.
- SIP/telephony NOT implemented.
- Uses local `faster-whisper` (model=small, device=cpu, compute_type=int8;
  beam_size=1, best_of=1, temperature=0) with a Russian-only default
  (`STT_LANGUAGE=ru`, skipping language auto-detection) and a concise
  AngioPlus vocabulary `initial_prompt` (override via WHISPER_INITIAL_PROMPT;
  set empty to disable).

Current known quality/latency issue (real mic observed):
- Russian transcription accuracy on short technical phrases is not yet good
  enough (e.g. "Как выполнить калибровку катетера?" mistranscribed as
  "выполнить стрелябровку качеры." etc.).
- faster-whisper `small`/CPU/int8 is slower than realtime (Silero logs
  "inference is slower than realtime", ~0.66 s delay) and several seconds of
  total STT processing after an utterance.

Phase D1.5 added a benchmark corpus (voice/stt_phrases.py) and a human-run
STT benchmark helper (scripts/stt_benchmark.py) so accuracy and latency can be
measured A/B against recorded WAV files instead of guessed.

## Current Server Constraints

Server is small:

approximately 4 GB RAM
2 CPU
2 GB swap

BGE-M3 runs locally on CPU.

Do not add a large local LLM or large Whisper model without resource testing.

## Current Documents

6 documents indexed:

1. 2025July.pdf
2. AngioPlus Core v2.5 IFU_RU_Final.pdf
3. Common Info to Distributor - commercial (1).pdf
4. Competitor_2025.pdf
5. Development_Questions_L1_L2.docx
6. Q&A List ENG.xlsx

Total chunks:
118

## Confirmed Regression Cases

PASS:

"Где посмотреть версию программного обеспечения?"

Expected:
HELP&ABOUT on PStation toolbar.

Correct source:
IFU → 12 Техническая поддержка → pages 10–11.

PASS:

"Кто может устанавливать и активировать AngioPlus Core?"

PASS:

"Что делать, если программа не реагирует на действия пользователя?"

PASS:

"Можно ли использовать AngioPlus Core для хранения медицинских карт пациентов?"

PASS:

"Как приготовить борщ?"

Expected:
"Такой информации нет в имеющейся документации."

No unrelated sources should be displayed for gate_false.

## Known Quality Issue

The angiography requirements answer has occasionally produced overly strong wording such as:

"точное измерение невозможно"

or ambiguous wording such as:

"короткое сужение"

Need prompt/regression refinement.

Preferred distinction:

- accuracy may be reduced / cannot be guaranteed
- minimal vessel foreshortening
- "not evaluated" must not become "prohibited"

## Git Workflow

Current source of truth:

GitHub `main`

Normal development:

local VS Code
→ commit
→ push
→ production pull

Production server should not normally become the primary development environment.

## Secrets

Never print or commit:

- DEEPSEEK_API_KEY
- GEMINI_API_KEY
- TELEGRAM_BOT_TOKEN
- GIGACHAT_CREDENTIALS

`.env` is environment-specific and must remain outside Git.

## Immediate Next Development Tasks

Recommended priority:

1. Regression-test conversation memory/follow-up rewriting.
2. Build a structured 30–50 question evaluation dataset.
3. Fix remaining medical wording issues.
4. Measure whether DeepSeek gate can safely be removed/merged.
5. Evaluate local faster-whisper for voice transcription.
6. Consider hybrid retrieval/reranking only after measured retrieval failures.

## Instructions for AI Coding Agent

Do not rewrite working architecture without evidence.

Before editing:

- inspect actual files
- inspect tests
- inspect git diff
- confirm assumptions against current code

Make small changes.

Run tests after each logical change.

Never expose secrets.

Do not treat README plans from older commits as current architecture.

When uncertain, inspect code rather than guessing.