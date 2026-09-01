# PROJECT CONTEXT — AngioPlus Core RAG

Last updated: 2026-08 (D2.2b)

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
STT benchmark helper (scripts/stt/stt_benchmark.py) so accuracy and latency can be
measured A/B against recorded WAV files instead of guessed.

### PHASE D2.2 — Qwen3-TTS 0.6B evaluation (ISOLATED PoC, NOT integrated)

Qwen3-TTS is NOT yet connected to the voice assistant. This phase only
prepares an isolated proof-of-concept to decide whether Qwen3-TTS 0.6B is
suitable as the future TTS engine BEFORE wiring TTS into LiveKit.

- Production/current voice path remains:
  microphone -> LiveKit -> Silero -> faster-whisper -> RAG -> text/data response
- TTS -> LiveKit is NOT implemented yet.
- Official model verified upstream: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
  (license: apache-2.0). Python package: `qwen-tts`.
- New isolated artifacts (do not touch the LiveKit/STT/VAD/RAG pipeline):
  - `voice/tts_phrases.py` — AngioPlus Russian TTS evaluation phrase set
    (short / normal / long buckets; exercises AngioPlus, AngioPlus Core,
    PStation, μFR, катетер, калибровка, ангиография, коронарная артерия,
    стеноз, виртуальный стент).
  - `scripts/tts/qwen_tts_poc.py` — human-run PoC:
    text -> Qwen3-TTS 0.6B -> Russian -> WAV, with latency/resource measurements.
  - `requirements-qwen-tts.txt` — ISOLATED dependency file. DO NOT install into
    the shared `.venv`: qwen-tts==0.1.1 pins transformers==4.57.3, conflicting
    with the project's pinned transformers==4.57.6.
- Streaming/TTFA: the official high-level `generate_custom_voice()` Python API
  is non-streaming at this level (returns a complete wavs,sr tuple), so true
  TTFA cannot yet be measured; streaming would need the lower-level Dual-Track
  path (out of scope here). This is reported honestly, not faked.
- Tests: `tests/test_qwen_tts_poc.py` (mocks / never downloads the model).

### PHASE D2.2b — Qwen3-TTS architecture decision (RESEARCH, verified upstream)

RESEARCH-ONLY conclusion. No TTS change has been wired into LiveKit/voice;
the STT/VAD/RAG path is untouched. The following was verified by reading the
official `QwenLM/Qwen3-TTS` source (`qwen_tts` package, v0.1.1), the official
README, the official vLLM-Omni README section, and the Aliyun DashScope
model-studio real-time API docs. CPU synthesis quality is ACCEPTED; CPU
latency is NOT accepted (see numbers below), so we proceed to GPU.

VERIFIED STREAMING FACTS (from source, not marketing):
- The open-source `qwen-tts` Python package does NOT expose true streaming
  AUDIO output for ANY `generate_*` method, including `generate_custom_voice()`
  used by our 0.6B-CustomVoice checkpoint. Every `generate_*` returns a
  complete `(wavs, sr)` tuple only.
- The `non_streaming_mode` parameter on `generate_custom_voice() /
  generate_voice_design() / generate_voice_clone()` only changes how TEXT is
  fed into the prompt (default `True` = whole-text input). The docstring states
  explicitly it "only simulates streaming text input when set to false, rather
  than enabling true streaming input or streaming generation." It does NOT
  change audio output to a stream.
- The underlying `Qwen3TTSForConditionalGeneration.generate(...)` runs the
  whole autoregressive codec decode then decodes ALL codes at once; no
  per-chunk audio yield exists in the local package.
- TRUE streaming ("Dual-Track hybrid", ~97 ms first-audio claim) is real but
  currently exposed ONLY through Aliyun's DASHSCOPE HOSTED realtime WebSocket
  API (`qwen3-tts-flash-realtime` / `qwen3-tts-instruct-flash-realtime`,
  `response.audio.delta` events). It is NOT self-hostable from the open-source
  package today without reimplementing the streaming loop.
- vLLM (vLLM-Omni) officially supports Qwen3-TTS "day-0", but the official
  README states it currently supports ONLY OFFLINE inference; online serving
  and streaming are explicitly "will be supported later". So vLLM is NOT yet a
  streaming path for us.
- 0.6B-CustomVoice does NOT support instruction control either: source forces
  `instruct=None` for `tts_model_size in "0b6"` (only 1.7B-CustomVoice supports
  instruct). This is expected and unrelated to our greeting/QA phrasing.

VERIFIED MODEL / HARDWARE FACTS:
- Config confirms 0.6B (talker: hidden 1024, 28 layers, 16 heads / GQA 8 KV,
  SwiGLU; code predictor: 5 layers). bf16 weights ~= 1.2-1.6 GB; with activations
  + KV cache the model fits comfortably in an 8 GB GPU in bf16. So 8 GB is the
    practical footprint for inference; 12-16 GB gives headroom.
- Model supports flash_attn_2, SDPA, flex_attn AND eager attention
  (`_supports_flash_attn/_supports_sdpa/_supports_flex_attn = True`; default
  `eager_attention_forward`). FlashAttention-2 is RECOMMENDED but NOT required;
  consumer GPUs can run eager/SDPA without installing flash-attn.
- FlashAttention-2 requires FP16/BF16 and NVIDIA Ampere+ (or compatible). On
  Windows flash-attn requires a prebuilt wheel / CUDA toolchain; it is easier
  on Linux CUDA. CPU path already verified slow (below).
- transformers is pinned `==4.57.3` by `qwen-tts==0.1.1`, conflicting with the
  project's pinned `transformers==4.57.6`. TTS must stay dependency-isolated.

RECORDED CPU MEASUREMENT (this machine, warm cache, no GPU):
- model load ~20.7 s; synthesis ~33.65 s for ~3.76 s audio -> RTF ~8.95,
  RAM ~4.5 GB, device cpu. Quality accepted; latency NOT accepted.

INFERENCE ARCHITECTURE CHOICE (decision):
- TODAY: use direct PyTorch/CUDA inference via the `qwen-tts` package
  (`Qwen3TTSModel`, bf16, cuda, optional SDPA/eager) in a SEPARATE dedicated
  TTS inference service, isolated from the voice-agent process. Non-streaming
  per-utterance synthesis is the only locally verified path; it is fine for
  MVP because our RAG answer comes back complete before we speak.
- True streaming / barge-in: NOT available from the self-hosted open-source
  stack today. If low TTFA streaming becomes mandatory, re-evaluate Aliyun
  DashScope realtime (hosted, latency-sensitive, data leaves device) OR await
  vLLM-Omni online/streaming support. This is a later decision, NOT D2.3.
- vLLM-Omni offline inference is kept as a faster batch alternative; not the
  production path until online/streaming ships.
- TTS MUST be a separate service because of the transformers pin conflict and
  so its CUDA/bfloat16 runtime does not destabilize the STT/VAD/RAG process.

RECOMMENDED GPU:
- Development minimum: an NVIDIA RTX 4060 Ti 16 GB (or 3060 12 GB) is enough
  to run 0.6B in bf16 with headroom; 8 GB (e.g. 3070/4060) likely enough to
  just load and run but tight. Recommend 16 GB for dev comfort.
- Production: a single NVIDIA Ada/Ampere 16-24 GB card (e.g. RTX 4090, RTX
  A4500/A5000, or L4 24 GB) is far more than needed for 0.6B and gives huge
  latency headroom; a 12-16 GB card is the sensible sweet spot for this
  checkpoint. We care about latency, not batch throughput.
- Cloud first is recommended: rent an NVIDIA L4 24 GB or an RTX A6000/4090 24 GB
  GPU instance (or a 16 GB A10G) for a real GPU benchmark before buying.

PROPOSED LATENCY TARGETS:
- Warm model required (load ~1-3 s on NVMe + GPU; keep process alive).
- acceptable MVP: TTS TTFA <= 1.5 s; TTS RTF <= 0.5; total FINAL_TRANSCRIPT ->
  first spoken audio <= 3.5 s.
- good production target: TTS TTFA <= 0.5 s; TTS RTF <= 0.1-0.2;
  total <= 2.0 s.
- These are targets pending a real GPU benchmark; not verified numbers yet.

NEXT (do NOT start D2.3): run `scripts/tts/qwen_tts_poc.py` on a rented CUDA GPU
(eager/SDPA first, then flash-attn) to measure load time, synthesis RTF, actual
VRAM, and confirm bf16. Then measure end-to-end FINAL_TRANSCRIPT -> first audio
with a NON-STREAMING integration before considering a streaming path.

### ENVIRONMENT / COMPATIBILITY (recorded 2026-08)

- Local dev machine: Windows 11, AMD Ryzen 7 7730U, 15.3 GB RAM, NO CUDA GPU
  (torch sees torch.cuda.is_available() == False; integrated Radeon).
- Project venv: Python 3.14.3, torch 2.13.0+cpu, transformers 4.57.6,
  numpy 2.5.1, scipy 1.18.0. qwen-tts / librosa / soundfile are NOT installed
  in the shared venv and must not be added there.
- Qwen3-TTS officially recommends a fresh Python 3.12 environment and a CUDA
  GPU for reasonable speed. CPU-only execution is possible but expected to be
  slow; the manual test must be done in an isolated venv (see
  requirements-qwen-tts.txt). No Qwen model has been downloaded yet in D2.2;
  the human must consciously run `scripts/tts/qwen_tts_poc.py` (or a manual HF
  download) in the isolated env after the PoC was prepared.

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
7. After D2.2b (architecture decided, CPU rejected): run the Qwen3-TTS 0.6B
   PoC on a rented CUDA GPU (eager/SDPA, then flash-attn) to benchmark load
   time, synthesis RTF, and real VRAM in bf16, then measure end-to-end
   FINAL_TRANSCRIPT -> first audio with a NON-STREAMING TTS integration. Do NOT
   start D2.3 (streaming/barge-in) until the non-streaming GPU path is proven.

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