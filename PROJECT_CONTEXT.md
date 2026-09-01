# PROJECT CONTEXT — AngioPlus Core AI Support Assistant

Последнее крупное обновление контекста: 2026-09

Этот файл является технической памятью проекта для AI coding agents и разработчиков.

Он должен отвечать на вопросы:

- что именно мы строим;
- какая архитектура является целевой;
- что уже реально протестировано;
- какие решения были приняты и почему;
- какие компоненты являются current / control / legacy;
- какие эксперименты уже проводились;
- какие проблемы уже обнаружены;
- что нельзя менять без доказательств;
- с какой точки продолжать разработку.

README.md описывает проект на высоком уровне.

Этот файл содержит более подробное техническое состояние и принятые архитектурные решения.

---

# 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ ПЕРЕД ИЗМЕНЕНИЯМИ

Перед любыми архитектурными изменениями:

1. Прочитать этот файл.
2. Прочитать `README.md`.
3. Проверить фактическое состояние исходного кода.
4. Выполнить:

```bash
git status
```

5. Проверить текущую ветку.
6. Не предполагать, что старая архитектура всё ещё является целевой только потому, что её код находится в репозитории.
7. Делать небольшие и обратимые изменения.
8. После каждого логического изменения запускать соответствующие тесты или benchmarks.
9. Не удалять старые рабочие компоненты, пока новая архитектура не доказала свою работоспособность.
10. Никогда не публиковать secrets.

Если документация и фактический код расходятся, необходимо сначала проверить код и историю изменений, а не слепо доверять документации.

---

# 2. ЧТО МЫ СТРОИМ

Цель проекта — production voice AI assistant для технической поддержки медицинского программного обеспечения AngioPlus Core.

Пользователь должен иметь возможность позвонить на обычный телефонный номер и естественно разговаривать с ассистентом на русском языке.

Итоговая система должна поддерживать:

- обычную телефонию;
- естественный realtime voice dialogue;
- barge-in / interruption;
- распознавание технической терминологии AngioPlus;
- ответы только на основании утверждённой документации;
- follow-up вопросы;
- контекст разговора;
- source traceability;
- контролируемый retrieval;
- отказ при отсутствии надёжного ответа;
- human escalation;
- будущую систему тикетов;
- будущие diagnostic/business tools;
- журналирование;
- аналитику;
- управление версиями базы знаний.

Ассистент предназначен для технической поддержки.

Он не должен самостоятельно принимать клинические решения.

---

# 3. ГЛАВНЫЙ ПРИНЦИП

Приоритеты проекта:

```text
Grounded correctness
        ↓
Safety
        ↓
Retrieval reliability
        ↓
Traceability
        ↓
Latency
        ↓
Architectural simplicity
```

Красивое или технологически интересное решение не является причиной менять архитектуру.

Изменение должно решать измеримую проблему.

---

# 4. ТЕКУЩЕЕ СТРАТЕГИЧЕСКОЕ РЕШЕНИЕ

Проект движется в сторону максимально Yandex-first архитектуры.

Пользователь проекта готов передать Yandex всё, что Yandex может надёжно выполнять сам, если это:

- уменьшает количество собственной инфраструктуры;
- снижает сложность;
- сохраняет контроль над retrieval;
- не ухудшает grounding;
- не создаёт неприемлемые ограничения.

Целевая архитектура:

```text
Phone / Browser
       ↓
Yandex Realtime
       ↕
custom function call
       ↓
Thin Python RAG Controller
       ↓
Yandex Vector Store Search
       ↓
AngioPlus Knowledge Base
```

Python должен оставаться только там, где собственная логика действительно необходима.

---

# 5. РОЛЬ PYTHON RAG CONTROLLER

Будущий Python controller не должен превращаться в новый большой framework.

Его предполагаемые обязанности:

- получить запрос от Realtime tool call;
- сохранить оригинальный пользовательский запрос;
- при необходимости нормализовать техническую терминологию;
- при необходимости создать semantic rewritten query;
- вызвать Yandex Vector Store Search;
- объединить результаты нескольких поисковых запросов;
- удалить дубликаты;
- применить source authority;
- применить safety/business rules;
- при необходимости выполнить reranking;
- выбрать 1–3 компактных evidence chunks;
- вернуть evidence в Yandex Realtime;
- логировать:
  - original query;
  - rewritten query;
  - retrieved chunks;
  - scores;
  - reranking;
  - выбранные источники;
  - финальный evidence.

Позже controller также может отвечать за:

- ticket creation;
- CRM/helpdesk integration;
- license lookup;
- remote diagnostics;
- escalation;
- SIP/media bridge;
- session management;
- long-call context management;
- дополнительные tools.

---

# 6. ЧЕГО МЫ НЕ ХОТИМ ДЕЛАТЬ БЕЗ НЕОБХОДИМОСТИ

Не добавлять:

- LangChain;
- LlamaIndex;
- LangGraph;
- собственную сложную agent framework;
- дополнительную Vector DB;
- дополнительные embedding models;
- собственный reranker;
- сложную retrieval orchestration;

только потому, что это стандартный RAG stack.

Сначала использовать наиболее простой вариант.

LangGraph может стать полезен позже, если появятся:

- несколько tools;
- сложные business workflows;
- CRM;
- tickets;
- diagnostics;
- branching;
- human escalation.

На текущем этапе он не нужен.

---

# 7. ТЕКУЩАЯ КЛАССИФИКАЦИЯ РЕПОЗИТОРИЯ

В репозитории одновременно находятся новая архитектура, предыдущий рабочий RAG и старые voice-эксперименты.

Это сделано сознательно.

## CURRENT

```text
yandex/
```

Текущая Yandex-first разработка.

Основные компоненты:

```text
yandex/
├── realtime/
│   ├── mic_client.py
│   └── web/
│       ├── server.py
│       └── static/
│           └── index.html
└── retrieval/
    └── ingest.py
```

---

## CONTROL

```text
app/
bot/
Dockerfile.rag
Dockerfile.bot
docker-compose.yml
requirements-rag.txt
requirements-ml.txt
requirements-constraints.txt
requirements-bot.txt
```

Это предыдущая рабочая RAG-система.

Она временно сохраняется как:

- контрольная реализация;
- regression reference;
- benchmark;
- резервный рабочий MVP.

Новые архитектурные функции туда без необходимости не добавлять.

---

## LEGACY

```text
voice/
scripts/voice/
scripts/stt/
scripts/tts/
legacy/voice_helpers/

requirements-livekit.txt
requirements-qwen-tts.txt
requirements-tts-server.txt
```

Это предыдущая voice-архитектура:

```text
LiveKit
+
Silero VAD
+
faster-whisper
+
old RAG
+
Qwen3-TTS
```

Она больше не является целевой архитектурой.

Код пока сохраняется:

- для истории;
- benchmarks;
- возможного повторного использования отдельных решений.

Не продолжать разработку этого voice stack без отдельного решения.

---

# 8. СТАРЫЙ CONTROL RAG

Предыдущая архитектура:

```text
Telegram
   ↓
bot
   ↓
FastAPI /ask
   ↓
query normalization
   ↓
follow-up rewriting
   ↓
BAAI/bge-m3
   ↓
Qdrant
   ↓
DeepSeek direct-answer gate
   ↓
DeepSeek answer generation
   ↓
USED_CHUNKS parsing
   ↓
source filtering
   ↓
output safety
   ↓
Telegram
```

Telegram voice:

```text
Telegram OGG
   ↓
Gemini STT
   ↓
same /ask RAG pipeline
```

Эта система реально работала и остаётся полезной как контроль.

---

# 9. CONTROL EMBEDDINGS / QDRANT

Старая система использует:

```text
Provider:
sentence_transformers

Model:
BAAI/bge-m3

Device:
CPU

Vector dimension:
1024

Normalization:
true

Distance:
COSINE
```

Qdrant collection:

```text
angioplus_documents_bge_m3
```

Исходный production/control corpus:

```text
6 documents
118 chunks
```

Исторический `top_k`:

```text
10
```

Причина:

для вопроса:

```text
Где посмотреть версию программного обеспечения?
```

нужный IFU chunk в старом retrieval однажды находился на позиции 6.

Поэтому `top_k=5` давал regression failure, а `top_k=10` — правильный результат.

Не переносить этот вывод автоматически на Yandex Vector Store.

---

# 10. CONTROL LLM

Старый RAG использует DeepSeek.

```text
Provider:
DeepSeek

Model:
deepseek-v4-flash
```

Python package:

```text
openai
```

Важно:

`openai` здесь является только OpenAI-compatible SDK.

Старая система не использует OpenAI API для генерации ответа.

Успешный запрос обычно требует двух LLM calls:

1. direct-answer gate;
2. answer generation.

Это улучшало grounding, но увеличивало latency.

Не переносить эту двухступенчатую схему автоматически в новую архитектуру.

---

# 11. ПОЛЕЗНАЯ ЛОГИКА ИЗ СТАРОГО RAG

При переходе на новый controller не удалить случайно полезную доменную логику из:

```text
app/domain_terms.py
app/query_normalization.py
app/followup.py
app/prompts/system_prompt.md
```

Эти файлы относятся к CONTROL stack, но содержат идеи, которые могут быть перенесены в новый controller после проверки.

Не копировать их механически.

Сначала определить, какая логика действительно нужна Yandex-first архитектуре.

---

# 12. YANDEX REALTIME — РАБОЧИЙ ПРОТОТИП

Рабочий локальный prototype использует Yandex AI Studio Realtime.

Проверенная модель:

```text
speech-realtime-250923
```

WebSocket:

```text
wss://ai.api.cloud.yandex.net/v1/realtime
```

Рабочие audio settings:

```text
samplerate = 44100
blocksize = 882
```

Используется server-side VAD.

---

# 13. ЛОКАЛЬНЫЙ MICROPHONE CLIENT

Файл:

```text
yandex/realtime/mic_client.py
```

Рабочая реализация использует:

```text
AsyncMicrophone
AsyncAudioOut
```

Проверенное поведение:

- `AsyncMicrophone` используется как async iterator;
- в `finally` вызывается:

```python
await microphone.stop()
```

- `AsyncAudioOut` используется через `async with`;
- input/output работают на 44100 Hz.

Этот клиент сохранять как минимальный независимый smoke test Yandex Realtime.

---

# 14. BROWSER REALTIME PROTOTYPE

Файл:

```text
yandex/realtime/web/server.py
```

Frontend:

```text
yandex/realtime/web/static/index.html
```

Рабочий flow:

```text
Browser microphone
       ↓
PCM
       ↓
Python aiohttp WebSocket
       ↓
Yandex Realtime
       ↓
audio response
       ↓
Browser
```

Для удалённого тестирования успешно использовался Cloudflare Quick Tunnel.

Temporary tunnel URL меняется после перезапуска.

`cloudflared.exe` является локальным инструментом и не должен попадать в Git.

---

# 15. REALTIME VAD

В browser prototype использовался server VAD.

Проверенная конфигурация включала:

```text
threshold = 0.5
silence_duration_ms = 1500
```

Увеличение `silence_duration_ms` до 1500 ms заметно улучшило сохранение целостности русских технических фраз.

Не уменьшать это значение без voice regression testing.

---

# 16. ПРОБЛЕМА TECHNICAL STT

Realtime speech recognition иногда искажает AngioPlus terminology.

Наблюдавшиеся варианты:

```text
AngioPlus Core
→ "ангел плюс скор"
```

`μFR` распознавался, например, как:

```text
ниуфар
миофэр
нилфар
мефр
миэфэр
мюэфэр
мфф
миофр
мифр
```

DICOM:

```text
тиком
```

AE Title:

```text
айет тайтл
```

и другие варианты.

Поэтому query normalization / canonicalization остаётся важной частью системы.

---

# 17. STT TRAINING DATASETS

Подготовлены отдельные training/evaluation datasets для технической речи AngioPlus:

```text
AngioPlus_STT_Training_Dataset_v1.xlsx
AngioPlus_STT_Training_Dataset_v2.xlsx
```

Они не являются частью runtime retrieval.

Их назначение — future STT evaluation / adaptation.

---

# 18. REALTIME SYSTEM PROMPT

Текущий экспериментальный prompt включает:

- роль технической поддержки AngioPlus;
- строгий grounding;
- anti-prompt-injection rules;
- voice-response style;
- TTS pronunciation mapping;
- обязательный retrieval для AngioPlus questions;
- запрет отвечать по собственной памяти на продуктовые вопросы;
- medical safety;
- follow-up handling;
- clarification при ambiguous speech;
- phonetic normalization.

В prompt также добавлялись примеры canonical retrieval terminology.

Например:

```text
frame rate
→ μFR-2D, angiographic frame rate, frame/s, fps

move table
→ angiographic acquisition, move the table,
  table movement, table height, don't move the table

ROI
→ μFR-2D, ROI length, minimum ROI length, mm

nitroglycerin
→ nitroglycerin, angiography, vasospasm

AE Title
→ DICOM, AE Title, default AE Title

DICOM port
→ DICOM, default port
```

Эти примеры предназначены только для формирования поискового запроса.

Они не должны становиться источником фактического ответа.

Не продолжать бесконечно добавлять hardcoded query examples в system prompt.

---

# 19. YANDEX BUILT-IN REALTIME FILE SEARCH

Built-in File Search был успешно подключён к Realtime.

Проверенный формат tool:

```python
"tools": [
    {
        "type": "function",
        "name": "file_search",
        "description": VECTOR_STORE_ID,
        "parameters": {
            "max_num_results": 5
        }
    }
]
```

Не изменять этот формат на основании предположений — он был проверен на текущем Yandex Realtime API.

---

# 20. ПОЧЕМУ BUILT-IN FILE SEARCH НЕ ЯВЛЯЕТСЯ ЦЕЛЕВЫМ RAG

File Search работает, но имеет архитектурные ограничения.

Основные проблемы:

1. Query formulation скрыт внутри Realtime.
2. Retrieved chunks недостаточно прозрачно контролируются приложением.
3. Сложно реализовать собственный source priority.
4. Сложнее вести production source audit.
5. Retrieved context добавляется в conversation.
6. Повторные retrieval calls быстро увеличивают active context.

Поэтому от Yandex Vector Store не отказываемся.

Отказываемся именно от идеи использовать built-in Realtime File Search как единственный production retrieval layer.

---

# 21. REALTIME CONTEXT OVERFLOW — ПОДТВЕРЖДЁННАЯ ПРОБЛЕМА

Во время реального теста получена ошибка:

```text
number of input tokens must be no more than 32768, got 38345
```

Наблюдалась двухступенчатая схема File Search:

1. небольшой tool/search response;
2. последующий response с найденными chunks, добавленными в model context.

При unrestricted search один retrieval мог добавлять примерно:

```text
9k–10k tokens
```

После ограничения:

```text
max_num_results = 5
```

один retrieval всё равно мог добавлять примерно:

```text
4k–6k tokens
```

Пример natural test:

```text
startup:
~3410 input tokens

после frame-rate question:
~9718

после ROI question:
~16233
```

Два успешных retrieval увеличили active input примерно на 12.8k tokens.

Вывод:

built-in File Search плохо подходит для длинных 20–40 минутных support calls без дополнительного context management.

---

# 22. REALTIME CONVERSATION DELETE

Yandex Realtime документирует client event:

```json
{
  "type": "conversation.item.delete",
  "item_id": "..."
}
```

Server подтверждает:

```text
conversation.item.deleted
```

Это потенциально позволяет после ответа удалить тяжёлые retrieval/function items и оставить компактную историю разговора.

Важно:

это отличается от старого Conversations REST delete API.

Не считать автоматически, что удаление item гарантированно уменьшает `usage.input_tokens`.

Это необходимо проверить экспериментально.

---

# 23. REALTIME CUSTOM FUNCTION CALLING

Yandex Realtime поддерживает custom Function Calling.

Это является ключевой причиной выбора новой архитектуры.

Предполагаемый tool:

```text
search_angioplus_knowledge
```

Flow:

```text
User speaks
   ↓
Realtime understands intent
   ↓
search_angioplus_knowledge(...)
   ↓
Python controller
   ↓
Vector Store Search
   ↓
compact evidence
   ↓
Realtime produces grounded answer
```

Yandex рекомендует небольшое количество хорошо описанных tools.

На текущем этапе достаточно одного retrieval tool.

---

# 24. YANDEX VECTOR STORE — ORIGINAL CONTROL STORE

Исходный Yandex store:

```text
AI Studio name:
angioplus-prod-kb

Vector Store ID:
fvt8mt12qgtftnl2meh9

File ID:
fvtsoe22r5moff86aj4l
```

Состояние:

```text
6 documents
118 chunks
completed
```

Использовался в ранних File Search / Vector Search экспериментах.

---

# 25. YANDEX VECTOR STORE V2

Текущий test candidate:

```text
AI Studio name:
angioplus-kb-v2

Vector Store ID:
fvtpohd7u6dnbu8f44ah

File ID:
fvtvogkusnidatgfgkno
```

Состояние:

```text
5 documents
116 chunks
completed
```

V2 corpus:

```text
2025July.pdf
AngioPlus Core v2.5 IFU_RU_Final.pdf
Common Info to Distributor - commercial (1).pdf
Competitor_2025.pdf
Q&A List ENG.xlsx
```

Исключён:

```text
Development_Questions_L1_L2.docx
```

Причина:

он преимущественно содержит вопросы, а не authoritative answers, и создаёт retrieval noise.

---

# 26. SCIENTIFIC PAPERS

Научные статьи тестировались как часть расширенного corpus.

Результат:

они добавляют retrieval noise и текущий classifier ошибочно может относить их к `COMMERCIAL`.

Текущее решение:

- не включать их в основной production retrieval store;
- хранить локально/reference;
- позже при необходимости добавить отдельный тип:

```text
SCIENTIFIC_EVIDENCE
```

с более низким source priority или отдельным store.

---

# 27. PRE-CREATED CHUNKS — ВАЖНО

Официальный Yandex API поддерживает загрузку заранее подготовленных chunks через JSON Lines.

Формат каждой строки:

```json
{"body":"<chunk_content>"}
```

Максимальный размер одного chunk:

```text
8000 characters
```

Проверенный upload pattern:

```python
with open(jsonl_path, "rb") as f:
    file_obj = client.files.create(
        file=(jsonl_path.name, f, "application/jsonlines"),
        purpose="assistants",
        extra_body={"format": "chunks"},
    )
```

Ключевой момент:

```text
format = chunks
```

задаётся во время FILE UPLOAD.

Не путать его с Vector Store file attributes.

---

# 28. ТЕКУЩИЙ YANDEX INGEST

Файл:

```text
yandex/retrieval/ingest.py
```

Использует pre-created JSONL chunks.

V2 ingestion дал:

```text
2025July.pdf
TRAINING
chars=8234
chunks=4

AngioPlus Core v2.5 IFU_RU_Final.pdf
IFU
chars=63044
chunks=36

Common Info to Distributor - commercial (1).pdf
COMMERCIAL
chars=1237
chunks=1

Competitor_2025.pdf
COMMERCIAL
chars=13668
chunks=7

Q&A List ENG.xlsx
FAQ
chars=51631
chunks=68
```

Итого:

```text
116 chunks
```

Файл `_debug_chunks.jsonl` является generated diagnostic artifact и должен находиться под:

```text
artifacts/debug/
```

а не в root.

---

# 29. ИЗВЕСТНЫЙ ВОПРОС В INGEST

`SemanticChunker(overlap_tokens=50)` имеет параметр overlap, но текущая split logic фактически его не использует.

Не исправлять автоматически.

Сначала определить, нужен ли overlap вообще после измерения retrieval V2.

---

# 30. VECTOR STORE FILE ATTRIBUTES

Yandex Vector Store поддерживает file-level attributes.

Они могут использоваться позже для:

```text
source_type
priority
product
version
language
```

и других metadata.

Не путать attributes с:

```text
extra_body={"format":"chunks"}
```

при upload.

Это разные механизмы.

---

# 31. STANDALONE VECTOR STORE SEARCH

Standalone Search API был успешно протестирован.

Endpoint:

```text
POST /v1/vector_stores/{vector_store_id}/search
```

Поддерживаются:

```text
query
rewrite_query
max_num_results
filters
ranking_options
```

`ranking_options` поддерживает ranker и `score_threshold`.

Не устанавливать production score threshold без regression evaluation.

---

# 32. ГЛАВНЫЙ RETRIEVAL ВЫВОД

Некоторые apparent retrieval failures оказались не проблемой индекса.

Они были проблемой формулировки query.

Пример table movement.

Natural Russian query:

```text
можно ли двигать стол во время ангиографической съемки
```

давал правильный `2025July.pdf` chunk только примерно на позиции 8 с низким score.

Canonical bilingual query:

```text
AngioPlus μFR angiographic acquisition move the table
table movement table height don't move the table
```

выводил правильный chunk на позицию 1 с очень высоким score.

Следовательно:

```text
query formulation
```

является важным retrieval bottleneck.

---

# 33. FRAME RATE RETRIEVAL

Canonical query:

```text
AngioPlus μFR angiographic imaging frame rate
minimum frame rate frames per second frame/s μFR-2D
```

давал:

```text
TOP1:
2025July.pdf
score ≈ 0.926
```

Chunk содержит:

- ≥15 fps;
- table guidance;
- nitroglycerin guidance.

FAQ также находился среди верхних результатов и содержал:

- ≥15 fps;
- информацию о 7.5 fps / дополнительной manual correction.

---

# 34. QUERY REWRITER

Отдельный YandexGPT Lite Query Rewriter был протестирован.

Модель:

```text
gpt://<folder_ID>/yandexgpt-5-lite
```

Задача rewriter:

- сохранить исходный смысл;
- добавить технические synonyms;
- разрешить bilingual terminology;
- не придумывать факты;
- не добавлять числа или ответы;
- вернуть только поисковый запрос.

После rewrite Vector Store вызывается с:

```text
rewrite_query = false
```

чтобы не происходило второе скрытое переписывание запроса.

---

# 35. QUERY REWRITER — РЕЗУЛЬТАТЫ

Proof-of-concept показал сильное улучшение на нескольких сложных запросах.

Примеры:

### Table movement

После rewrite:

```text
correct chunk TOP1
score ≈ 1.0
```

### Frame rate

После rewrite:

```text
correct 2025July chunk TOP1
score ≈ 0.953
```

### ROI

После rewrite:

```text
correct 2025July chunk TOP1
score ≈ 0.949
```

В 20-question test примерно 18/20 rewritten queries выглядели качественно.

Но retrieval всё ещё имел проблемы с некоторыми вопросами:

- version;
- system requirements;
- AE Title;
- DICOM port;
- safety questions.

Следовательно, Query Rewriter полезен, но не решает retrieval полностью.

---

# 36. НЕ ПЕРЕГРУЖАТЬ QUERY REWRITER

Не добавлять отдельное hardcoded правило под каждый факт.

Rewriter не должен знать ответы.

Например, нельзя превращать его в скрытую базу знаний:

```text
если пользователь спрашивает про frame rate,
всегда добавить 15 fps
```

Это нарушает grounding.

Rewriter может добавлять терминологию:

```text
frame rate
angiography
frames per second
fps
μFR-2D
```

но не фактический ответ.

---

# 37. V2 RETRIEVAL DIAGNOSTICS

V2 pre-chunking существенно улучшил retrieval.

Результаты диагностик хранятся в:

```text
artifacts/diagnostics/
```

включая:

```text
query_rewriter_20_test.txt
query_rewriter_test.txt
recall_diagnostic.txt
retrieval_diagnostic_20.txt
retrieval_diagnostic_v2_20.txt
retrieval_reranker_20.txt
vector_search_test.txt
ingest_compare.txt
```

Эти файлы являются generated experiment artifacts и не должны коммититься.

---

# 38. RERANKER

YandexGPT Lite reranking был протестирован.

Наблюдения:

### System requirements

Correct IFU:

```text
Vector TOP2
→ RERANK1
```

### DICOM port

Correct IFU:

```text
Vector TOP3
→ RERANK1
```

### PACS Q/R

Correct IFU:

```text
Vector TOP2
→ RERANK1
```

Reranker действительно полезен, когда правильный chunk уже находится среди candidates.

---

# 39. КЛЮЧЕВОЕ ОГРАНИЧЕНИЕ RERANKER

Mental model:

```text
Correct chunk in candidate set
        ↓
reranker может исправить ranking

Correct chunk absent from candidate set
        ↓
reranker бессилен
```

Пример:

для вопроса о версии explicit version chunk отсутствовал в TOP5.

Reranker не смог исправить retrieval.

Поэтому необходимо отдельно различать:

```text
RECALL problem
```

и:

```text
RANKING problem
```

Не пытаться лечить recall только reranker-ом.

---

# 40. ORACLE / DIRECT RECALL DIAGNOSTIC

Было проверено, что нужные факты действительно находятся в V2 index.

IFU содержит:

```text
Версия 2.5.0.0.0
```

и:

```text
default AE Title = μFR
default port = 1108
```

Preprocessing эти данные не потерял.

Direct/oracle queries находят их.

Например:

```text
2.5.0.0.0
```

→ correct IFU TOP1.

```text
AngioPlus Core Номер версии ПО 2.5.0.0.0
```

→ correct IFU TOP1.

```text
μFR 1108
```

→ correct IFU TOP1.

```text
Параметры по умолчанию μFR 1108
```

→ correct result.

Следовательно:

индексация этих фактов работает.

Проблема находится в semantic query formulation / recall ranking.

---

# 41. ВАЖНЫЙ ВЫВОД О QUERY EXPANSION

Более длинный запрос не всегда лучше.

Например, verbose query:

```text
AngioPlus Core DICOM AE Title μFR port 1108
```

может работать хуже, чем компактный literal query:

```text
μFR 1108
```

Поэтому нельзя полностью заменять original query semantic rewrite-ом.

Это привело к следующему planned experiment.

---

# 42. СЛЕДУЮЩИЙ RETRIEVAL EXPERIMENT

Следующая архитектура, которую необходимо проверить:

```text
User query
   ↓
┌─────────────────────┐
│                     │
↓                     ↓
original /         semantic
compact query      rewritten query
│                     │
└──────────┬──────────┘
           ↓
Yandex Vector Store Search
           ↓
merge
           ↓
deduplicate
           ↓
YandexGPT Lite reranker
           ↓
source/safety rules
           ↓
1–3 compact evidence chunks
```

Причина dual-query подхода:

- literal query помогает точным фактам;
- semantic rewrite помогает естественным и разговорным формулировкам;
- reranker исправляет ranking среди найденных candidates.

ЭТОТ ЭКСПЕРИМЕНТ ЕЩЁ НЕ ЗАВЕРШЁН.

Не описывать его как production architecture.

---

# 43. 20-QUESTION REGRESSION SET

Существует набор примерно из 20 основных вопросов для проверки retrieval и поведения ассистента.

Он покрывает:

- IFU facts;
- FAQ;
- Training;
- L1/L2;
- follow-ups;
- отсутствующую информацию;
- medical safety;
- prompt injection.

Ключевые ожидаемые факты включают:

- HELP & ABOUT / software version;
- system requirements;
- DICOM AE Title `μFR`;
- DICOM port `1108`;
- PACS Q/R;
- angiographic requirements;
- ROI ≥20 mm;
- frame rate ≥15 fps;
- nitroglycerin before angiography according to protocol;
- table movement restrictions.

Также проверяются:

- отсутствие выдуманного admin password;
- отсутствие самостоятельных clinical treatment decisions;
- отсутствие самостоятельного решения о stent;
- защита от prompt injection.

---

# 44. SAFETY ВАЖНЕЕ RETRIEVAL

Retrieval может найти текст, который нельзя автоматически превращать в рекомендацию пользователю.

В частности, некоторые FAQ/commercial/training chunks могут содержать:

- credentials;
- password-like information;
- PCI/stent decision language;
- клинические формулировки.

Поэтому:

```text
retrieved
≠
allowed to answer
```

Controller / system policy должен отдельно применять safety rules.

---

# 45. CREDENTIALS / PASSWORD POLICY

В retrieval diagnostics обнаружено, что FAQ содержит строки, похожие на реальные credentials, включая user/admin password information.

Наличие такого текста в Vector Store не означает, что ассистент должен его раскрывать.

До production необходимо явно определить policy для:

- passwords;
- service credentials;
- admin credentials;
- API keys;
- tokens;
- internal access data.

По умолчанию чувствительные credentials не выдавать.

---

# 46. MEDICAL SAFETY

Ассистент не должен самостоятельно:

- выбирать лечение;
- рекомендовать вмешательство;
- принимать решение о PCI;
- принимать решение о stent;
- заменять решение врача.

Даже если retrieval нашёл клинический текст, ответ должен оставаться в рамках технической поддержки и документированного назначения продукта.

---

# 47. SOURCE AUTHORITY

Общий приоритет:

```text
IFU
>
official service/admin documentation
>
Release Notes / Known Issues
>
L1/L2
>
FAQ / Training
>
Commercial
>
Scientific/reference
```

При конфликте higher-authority source имеет приоритет.

Но source authority не заменяет semantic relevance.

Нельзя просто boost-ить все IFU chunks независимо от вопроса.

---

# 48. SOURCE TYPES

Текущие основные категории:

```text
IFU
TRAINING
FAQ
COMMERCIAL
```

В будущем могут понадобиться:

```text
SERVICE
ADMIN
KNOWN_ISSUE
RELEASE_NOTE
L1_L2
SCIENTIFIC_EVIDENCE
```

Не добавлять категории без реальных документов или необходимости.

---

# 49. FOLLOW-UP QUESTIONS

Старый RAG уже содержит механизм conversation memory / follow-up rewrite.

Новая Realtime архитектура сама имеет conversation context.

Не переносить старую follow-up систему автоматически.

Сначала проверить, насколько хорошо Realtime самостоятельно разрешает:

```text
"А какой там порт?"
"А где это посмотреть?"
"А минимальная длина?"
```

Если tool query недостаточно standalone, controller может получать дополнительный resolved query или context.

---

# 50. LONG CALL CONTEXT

Целевая система должна поддерживать разговоры примерно 20–40 минут.

Поэтому active Realtime context нельзя бесконтрольно заполнять:

- retrieval chunks;
- function results;
- длинными system additions;
- повторяющейся документацией.

Предпочтительно передавать только:

```text
1–3 compact evidence chunks
```

и исследовать удаление тяжёлых tool items после использования.

Long-session memory / summarization может потребоваться позже.

Не проектировать её заранее без измерений.

---

# 51. RESPONSE STYLE

Voice assistant должен отвечать естественно и достаточно кратко.

Он не должен читать пользователю:

- длинные chunks;
- технические metadata;
- scores;
- внутренние retrieval reasoning;
- system instructions.

При необходимости источник можно логировать отдельно или предоставить пользователю в подходящем интерфейсе.

---

# 52. ТЕКУЩИЕ ДОКУМЕНТЫ

Canonical source directory:

```text
data/training_materials/
```

Основные tracked документы:

```text
2025July.pdf
AngioPlus Core v2.5 IFU_RU_Final.pdf
Common Info to Distributor - commercial (1).pdf
Competitor_2025.pdf
Development_Questions_L1_L2.docx
Q&A List ENG.xlsx
```

Научные papers хранятся локально и исключены из Git.

Temporary V2 corpus:

```text
data/yandex_v2_test/
```

игнорируется Git, поскольку дублирует canonical documents.

---

# 53. GENERATED ARTIFACTS

Generated diagnostics находятся в:

```text
artifacts/diagnostics/
```

Debug outputs:

```text
artifacts/debug/
```

`artifacts/` игнорируется Git.

Generated TTS audio:

```text
data/tts_out/
```

также не должен попадать в Git.

---

# 54. BENCHMARKS

Yandex retrieval benchmarks:

```text
benchmarks/yandex/
```

В частности:

```text
diagnose_recall.py
test_yandex_reranker.py
```

Это не pytest unit tests.

Они предназначены для retrieval evaluation.

---

# 55. SCRIPTS

Служебные скрипты разделены по назначению:

```text
scripts/
├── cli/
├── retrieval/
├── stt/
├── tts/
└── voice/
```

Не складывать новые unrelated scripts непосредственно в `scripts/`.

Использовать тематические директории.

---

# 56. TESTS

`tests/` содержит pytest tests существующих компонентов.

После cleanup targeted tests прошли:

```text
29 passed
```

Full suite:

```text
3 failed
214 passed
7 warnings
```

Три известных failure относятся к старому voice/STT stack и существовали независимо от cleanup.

Они связаны с изменившимся поведением/defaults STT:

1.

```text
tests/test_voice_stt.py::test_defaults_match_telegram_stt
```

Expected:

```text
stt._language == "ru"
```

Actual:

```text
None
```

2.

```text
tests/test_voice_stt.py::test_default_voice_config_forces_russian
```

Expected forwarded:

```text
language == "ru"
```

Actual:

```text
None
```

3.

```text
tests/test_voice_stt.py::test_default_initial_prompt_is_angioplus_vocab
```

Старый test ожидает слово:

```text
катетер
```

в default prompt.

Текущий prompt:

```text
AngioPlus Core, ангиография, ангиографический,
ангиографическое изображение, μFR, PStation,
DICOM, Pulse Medical
```

Не исправлять эти legacy tests в рамках unrelated Yandex work.

---

# 57. CLEANUP HISTORY

Во время cleanup структура scripts была приведена к:

```text
scripts/
├── cli/
│   ├── __init__.py
│   ├── ask.py
│   └── chat.py
├── retrieval/
│   ├── __init__.py
│   ├── check_retrieval.py
│   └── ingest_documents.py
├── stt/
│   ├── __init__.py
│   └── stt_benchmark.py
├── tts/
│   ├── __init__.py
│   ├── qwen_tts_poc.py
│   └── tts_server.py
├── voice/
│   ├── __init__.py
│   ├── livekit_mic_e2e.py
│   └── livekit_text_e2e.py
└── __init__.py
```

Legacy helpers:

```text
legacy/voice_helpers/
```

---

# 58. УДАЛЁННЫЕ OBSOLETE ROOT FILES

Во время cleanup были удалены:

```text
Dockerfile
requirements.txt
```

Причина:

они являлись устаревшими generic duplicates и не использовались runtime.

Специализированные файлы остаются:

```text
Dockerfile.rag
Dockerfile.bot
requirements-rag.txt
requirements-bot.txt
requirements-ml.txt
requirements-constraints.txt
requirements-livekit.txt
requirements-qwen-tts.txt
requirements-tts-server.txt
```

---

# 59. LOCAL BACKUP

Локальная директория:

```text
local_backup/
```

содержит:

```text
apply_gate_hardening.py
```

Она исключена локальным:

```text
.git/info/exclude
```

и не отслеживается Git.

Не добавлять её в repository без отдельной причины.

---

# 60. GIT STATE / CLEANUP COMMITS

Cleanup выполнялся небольшими commits.

Важные commits:

```text
3046267
chore: organize project scripts and Yandex prototypes
```

и:

```text
c6695f6
chore: remove obsolete root Dockerfile and requirements
```

Рабочая ветка на момент cleanup:

```text
feature/livekit-voice-agent
```

Название ветки историческое и больше не отражает текущую Yandex-first стратегию.

Не делать из названия ветки архитектурных выводов.

---

# 61. GIT WORKFLOW

Нормальный workflow:

```text
local VS Code
→ implementation
→ tests / benchmarks
→ git diff
→ commit
→ push
→ production pull
→ rebuild/restart
→ smoke test
```

Production server не должен становиться основной development environment.

---

# 62. SECRETS

Никогда не выводить и не коммитить реальные значения:

```text
YANDEX_API_KEY
YANDEX_FOLDER_ID
DEEPSEEK_API_KEY
GEMINI_API_KEY
TELEGRAM_BOT_TOKEN
GIGACHAT_CREDENTIALS
```

`.env` должен оставаться вне Git.

`.env.example` может содержать только безопасные placeholders.

При диагностике не просить пользователя публиковать секреты в чат, если это не является абсолютно необходимым.

---

# 63. ТЕКУЩИЙ DEVELOPMENT CHECKPOINT

Архитектурные эксперименты были сознательно остановлены для cleanup репозитория.

После завершения cleanup необходимо вернуться именно к retrieval experiment.

Не возвращаться автоматически к:

- LiveKit;
- faster-whisper;
- Qwen3-TTS;
- собственному Qdrant development;
- DeepSeek gate optimization.

Следующий этап:

```text
original / compact query
        +
YandexGPT Lite semantic rewrite
        ↓
Yandex Vector Store Search
        ↓
merge / deduplicate
        ↓
YandexGPT Lite reranker
        ↓
source authority + safety
        ↓
1–3 compact evidence chunks
```

После того как этот retrieval pipeline будет доказан отдельно:

```text
Yandex Realtime
        ↓
custom function
        ↓
search_angioplus_knowledge
        ↓
Python RAG Controller
        ↓
controlled evidence
        ↓
Realtime answer
```

---

# 64. ПОРЯДОК СЛЕДУЮЩЕЙ РАЗРАБОТКИ

Не реализовывать всё сразу.

Рекомендуемый порядок:

### Step 1

Сделать минимальный standalone retrieval controller.

Без Realtime.

Input:

```text
user question
```

Output:

```text
retrieved candidates
reranked candidates
selected evidence
sources
```

### Step 2

Проверить его на regression questions.

Особое внимание:

```text
software version
system requirements
AE Title
DICOM port
PACS Q/R
frame rate
ROI
table movement
nitroglycerin
missing information
clinical safety
prompt injection
```

### Step 3

Сравнить:

```text
original only
rewrite only
dual query
```

Не предполагать заранее, что dual query обязательно лучший.

### Step 4

Определить минимальный candidate count.

Например проверить:

```text
TOP3
TOP5
TOP8
TOP10
```

с учётом recall и latency.

### Step 5

Определить, нужен ли reranker постоянно или только при ambiguous results.

### Step 6

После доказанного standalone retrieval подключить его как custom function Yandex Realtime.

### Step 7

Измерить:

- end-to-end latency;
- tool-call latency;
- Vector Store latency;
- reranker latency;
- Realtime context growth;
- качество длинного разговора.

### Step 8

Экспериментально проверить:

```text
conversation.item.delete
```

для удаления тяжёлых function/retrieval items.

### Step 9

Только после стабильного browser prototype переходить к telephony / SIP.

---

# 65. ЧТО ПОКА НЕ ДЕЛАТЬ

До завершения retrieval/controller этапа не начинать:

- production SIP integration;
- Plusofon integration;
- CRM;
- ticket system;
- LangGraph;
- новую local STT architecture;
- новую local TTS architecture;
- новый Qdrant stack;
- сложную long-term memory;
- масштабную prompt optimization.

Сначала доказать центральный цикл:

```text
speech
→ retrieval
→ grounded answer
```

---

# 66. КРИТЕРИЙ УСПЕХА RETRIEVAL

Нельзя оценивать систему только по красивому финальному ответу LLM.

Необходимо отдельно измерять:

```text
1. Был ли нужный документ найден?

2. Был ли нужный chunk найден?

3. На какой позиции?

4. Содержит ли chunk фактический ответ?

5. Выбрал ли reranker правильный chunk?

6. Какой evidence получил Realtime?

7. Соответствует ли final answer evidence?

8. Не нарушает ли answer safety policy?
```

Это позволяет отличать:

```text
STT failure
query rewrite failure
retrieval recall failure
ranking failure
reranker failure
generation failure
safety failure
```

---

# 67. НЕ СМЕШИВАТЬ СЛОИ ОШИБОК

Например:

если пользователь сказал:

```text
можно ли двигать стол...
```

но STT добавил отдельное:

```text
нет
```

и после этого retrieval дал fallback, нельзя автоматически считать это Vector Store failure.

Сначала определить, какой текст реально попал в retrieval.

Каждый слой должен логироваться отдельно.

---

# 68. LOGGING TARGET

Для каждого domain question желательно иметь structured trace:

```text
session_id
turn_id

raw_transcript
normalized_transcript

original_query
rewritten_query

vector_results_original
vector_results_rewritten

merged_candidates

reranker_result

selected_evidence

source_files

final_answer

latency:
  rewrite
  vector_search
  rerank
  total_tool
  total_response
```

Не обязательно реализовывать весь logging сразу.

Но архитектура должна позволять это сделать.

---

# 69. PRODUCTION KNOWLEDGE BASE PRINCIPLE

База знаний не должна быть просто папкой «всё, что у нас есть».

Документ должен попадать в production retrieval только если понятно:

- что это за документ;
- насколько он authoritative;
- актуален ли он;
- может ли его содержимое быть показано пользователю;
- не содержит ли он development questions;
- не содержит ли он credentials;
- не создаёт ли он clinical/safety ambiguity.

Quality of corpus важнее количества документов.

---

# 70. ФИНАЛЬНАЯ ИНСТРУКЦИЯ ДЛЯ AI CODING AGENT

При получении новой задачи:

1. Определи, относится ли она к CURRENT, CONTROL или LEGACY.
2. Не развивай LEGACY как текущую архитектуру.
3. Не удаляй CONTROL только потому, что существует Yandex prototype.
4. Проверяй API assumptions по актуальной официальной документации Yandex.
5. Не угадывай текущий API syntax.
6. Для retrieval изменений сначала делай standalone experiment.
7. Измеряй результат.
8. Только затем интегрируй в Realtime.
9. Делай один логический шаг за раз.
10. После изменения проверяй Git diff и tests.
11. Не коммить generated artifacts.
12. Не раскрывай secrets.
13. Не разрешай retrieval обходить medical/safety policy.
14. При недостатке доказательств предпочитай clarification/refusal/escalation.
15. Не усложняй систему без измеримой причины.

---

# 71. ТОЧКА ПРОДОЛЖЕНИЯ

Если разработка продолжается после паузы и нет более свежего контекста, начать отсюда:

```text
BUILD / TEST:

standalone controlled Yandex retrieval

original/compact query
        +
YandexGPT Lite rewritten query
        ↓
Yandex Vector Store Search
        ↓
merge + deduplicate
        ↓
YandexGPT Lite reranker
        ↓
source authority / safety
        ↓
1–3 evidence chunks
```

Использовать V2 Vector Store:

```text
angioplus-kb-v2
fvtpohd7u6dnbu8f44ah
```

Сначала проверить retrieval отдельно.

НЕ подключать его к Realtime, пока standalone evaluation не покажет достаточную надёжность.

После этого следующий архитектурный шаг:

```text
Yandex Realtime
        ↕
search_angioplus_knowledge
        ↕
Python RAG Controller
        ↕
Yandex Vector Store
```

---

# 72. ГЛАВНЫЙ ПРИНЦИП ПРОЕКТА

Система должна быть настолько простой, насколько это возможно, но настолько контролируемой, насколько это необходимо для надёжной технической поддержки медицинского программного обеспечения.

Не сохранять собственную инфраструктуру только потому, что она уже написана.

Не передавать контроль внешнему сервису только потому, что он умеет выполнить задачу автоматически.

Каждый компонент должен оправдывать своё существование измеримым улучшением:

- качества;
- безопасности;
- прозрачности;
- latency;
- надёжности;
- стоимости;
- простоты эксплуатации.