# PROJECT CONTEXT — AngioPlus AI Support Assistant

Последнее крупное обновление: 2026-09-02

Этот файл является технической памятью проекта для разработчиков и AI coding agents.

Он должен позволить после любой паузы быстро понять:

- что именно строится;
- какая архитектура является текущей;
- что уже реально реализовано и протестировано;
- какие решения были приняты и почему;
- какие компоненты являются CURRENT / CONTROL / LEGACY;
- какие эксперименты уже завершены;
- какие проблемы известны;
- что пока сознательно отложено;
- с какой точки продолжать разработку.

`README.md` описывает проект на высоком уровне.

Этот файл является более подробным архитектурным и техническим snapshot.

---

# 1. ОБЯЗАТЕЛЬНО ПРОЧИТАТЬ ПЕРЕД ИЗМЕНЕНИЯМИ

Перед любыми существенными изменениями:

1. Прочитать этот файл.
2. Прочитать `README.md`.
3. Проверить фактическое состояние кода.
4. Выполнить:

```bash
git status
git branch --show-current
```

5. Не считать старый код целевой архитектурой только потому, что он находится в repository.
6. Делать небольшие и обратимые изменения.
7. После каждого логического изменения проводить соответствующий smoke test / benchmark.
8. Не удалять CONTROL / LEGACY компоненты без отдельного решения.
9. Не коммитить generated artifacts.
10. Никогда не публиковать secrets.

Если документация расходится с фактическим кодом:

```text
CODE + TESTED RUNTIME
>
старое описание в документации
```

После подтверждения фактического состояния документацию необходимо обновить.

---

# 2. ЧТО МЫ СТРОИМ

Цель проекта — AI assistant для технической поддержки медицинского программного обеспечения AngioPlus Core.

Ассистент должен поддерживать несколько каналов взаимодействия.

Ближайший MVP:

```text
Web Assistant
├── text chat
└── realtime voice conversation
```

В будущем:

```text
ordinary phone / SIP
```

Телефония сознательно отложена до завершения и демонстрации Web MVP.

Ассистент должен поддерживать:

- естественный русский диалог;
- realtime voice;
- text chat;
- follow-up вопросы;
- conversation context;
- barge-in / interruption;
- распознавание технической терминологии AngioPlus;
- controlled retrieval;
- ответы на основании утверждённой документации;
- source traceability;
- отказ / clarification при недостатке информации;
- medical safety;
- human escalation;
- future ticketing;
- future diagnostic/business tools;
- logging;
- analytics;
- versioned knowledge base.

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
User experience / latency
        ↓
Architectural simplicity
```

Не усложнять систему без измеримой причины.

Не менять рабочую архитектуру только потому, что существует более модный или технологически интересный вариант.

---

# 4. ТЕКУЩИЙ СТАТУС

На 2026-09-02:

```text
BACKEND MVP:
FUNCTIONALLY COMPLETE FOR DEMONSTRATION
```

Это НЕ означает:

```text
production-ready
```

Основной backend loop уже реализован и проверен:

```text
Yandex Realtime 260528
        ↓
custom Function Calling
        ↓
search_angioplus_knowledge(...)
        ↓
RetrievalController
        ↓
Yandex Vector Store Search
        ↓
TOP-5 evidence
        ↓
function_call_output
        ↓
Yandex Realtime
        ↓
grounded answer
```

Рабочий voice prototype также существует.

Следующий основной milestone:

```text
WEB ASSISTANT FRONTEND
```

с двумя способами взаимодействия:

```text
Text Chat
+
Voice Mode
```

---

# 5. ТЕКУЩАЯ ЦЕЛЕВАЯ АРХИТЕКТУРА

Текущая архитектура MVP:

```text
                     AngioPlus Assistant
                            │
               ┌────────────┴────────────┐
               │                         │
           Web Chat                  Voice Web
               │                         │
               └────────────┬────────────┘
                            ↓
                  Yandex Realtime 260528
                            ↓
              search_angioplus_knowledge
                            ↓
                  RetrievalController
                            ↓
               Yandex Vector Store V2
                            ↓
                     TOP-5 evidence
                            ↓
                  grounded response
                      ↙           ↘
                    text          voice
```

Будущий дополнительный канал:

```text
Phone / SIP
     ↓
same assistant core
```

Телефония не является текущим development priority.

---

# 6. СТРАТЕГИЧЕСКОЕ РЕШЕНИЕ: YANDEX-FIRST

Проект движется в сторону максимально простой Yandex-first архитектуры.

Yandex используется там, где сервис уже способен надёжно выполнять задачу:

- realtime speech recognition;
- server VAD;
- dialogue model;
- voice generation;
- Function Calling;
- Vector Store;
- semantic/hybrid retrieval.

Собственный Python используется только там, где нужен контроль.

Мы НЕ строим собственные:

- STT;
- TTS;
- embedding infrastructure;
- Vector DB;
- большой agent framework;

если в этом нет доказанной необходимости.

---

# 7. РОЛЬ PYTHON

Python является orchestration / control layer.

Он НЕ должен становиться новым большим RAG framework.

Текущая роль:

```text
Realtime function call
        ↓
Python RetrievalController
        ↓
Yandex Vector Store Search API
        ↓
TOP-5 evidence
        ↓
function_call_output
```

Python обеспечивает:

- controlled retrieval;
- query visibility;
- result visibility;
- logging;
- metadata;
- source classification;
- возможность будущих safety/business rules;
- возможность будущих tools.

В будущем Python также может отвечать за:

- ticket creation;
- CRM/helpdesk;
- diagnostics;
- license lookup;
- human escalation;
- session management;
- telephony/media bridge;
- analytics.

---

# 8. ЧЕГО ПОКА НЕ ДОБАВЛЯТЬ

Без доказанной необходимости не добавлять:

- LangChain;
- LlamaIndex;
- LangGraph;
- дополнительную Vector DB;
- собственные embedding models;
- отдельный reranker;
- сложную multi-agent architecture;
- новый local STT;
- новый local TTS;
- новый Qdrant stack.

LangGraph может стать оправдан позже, если появятся:

```text
multiple tools
+
branching workflows
+
CRM
+
tickets
+
diagnostics
+
human escalation
```

На текущем этапе он не нужен.

---

# 9. КЛАССИФИКАЦИЯ РЕПОЗИТОРИЯ

В repository одновременно находятся CURRENT, CONTROL и LEGACY компоненты.

Это сознательное решение.

## CURRENT

```text
yandex/
```

Основная текущая разработка.

Ключевые компоненты:

```text
yandex/
├── realtime/
│   ├── mic_client.py
│   └── web/
│       ├── server.py
│       └── static/
│           └── index.html
│
└── retrieval/
    ├── client.py
    ├── controller.py
    ├── ingest.py
    └── models.py
```

---

## CONTROL

Предыдущая рабочая RAG-система:

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

Она сохраняется как:

- regression reference;
- резервная реализация;
- исторически рабочий MVP;
- источник полезной domain logic.

Не развивать CONTROL stack без отдельной причины.

---

## LEGACY

Предыдущая voice architecture:

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

Архитектура:

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

Она больше не является целевой.

Не возвращаться к ней без доказанной необходимости.

---

# 10. СТАРЫЙ CONTROL RAG

Историческая рабочая архитектура:

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
same /ask pipeline
```

Этот stack реально работал.

Он остаётся CONTROL, но не является направлением текущей разработки.

---

# 11. CONTROL EMBEDDINGS / QDRANT

Старая система:

```text
Embedding:
BAAI/bge-m3

Device:
CPU

Dimension:
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

Исторический corpus:

```text
6 documents
118 chunks
```

Исторический:

```text
top_k = 10
```

Не переносить старые Qdrant параметры автоматически на Yandex Vector Store.

---

# 12. ПОЛЕЗНАЯ ЛОГИКА ИЗ CONTROL

Потенциально полезная domain logic находится в:

```text
app/domain_terms.py
app/query_normalization.py
app/followup.py
app/prompts/system_prompt.md
```

Не копировать её автоматически.

Переносить только после того, как конкретная проблема доказана тестами новой архитектуры.

---

# 13. YANDEX REALTIME — CURRENT MODEL

Текущая целевая модель:

```text
speech-realtime-260528
```

НЕ использовать старую:

```text
speech-realtime-250923
```

как production/default baseline без отдельной причины.

WebSocket:

```text
wss://ai.api.cloud.yandex.net/v1/realtime
```

Рабочий audio format:

```text
PCM16
44100 Hz
```

Browser / microphone prototype использует server-side VAD.

---

# 14. ПОЧЕМУ 260528

Исторически prototype начинался с:

```text
speech-realtime-250923
```

У него был context limit:

```text
32768 tokens
```

Во время File Search тестов реально была получена ошибка:

```text
number of input tokens must be no more than 32768, got 38345
```

Текущий:

```text
speech-realtime-260528
```

имеет существенно больший context window и является текущим baseline проекта.

Custom retrieval architecture была успешно проверена именно на:

```text
speech-realtime-260528
```

---

# 15. LOCAL MICROPHONE SMOKE TEST

Файл:

```text
yandex/realtime/mic_client.py
```

Рабочая реализация использует:

```text
AsyncMicrophone
AsyncAudioOut
```

Проверенные параметры:

```text
samplerate = 44100
blocksize = 882
```

`AsyncMicrophone` используется как async iterator.

В `finally`:

```python
await microphone.stop()
```

`AsyncAudioOut` используется через:

```python
async with
```

Этот клиент сохранять как минимальный независимый smoke test Yandex Realtime.

---

# 16. BROWSER VOICE PROTOTYPE

Backend:

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
PCM16
        ↓
aiohttp WebSocket
        ↓
Yandex Realtime 260528
        ↓
audio
        ↓
Browser
```

Удалённое тестирование ранее успешно проводилось через Cloudflare Quick Tunnel.

`cloudflared.exe` не должен попадать в Git.

---

# 17. REALTIME VAD

Используется server VAD.

Рабочая конфигурация исторически включала:

```text
threshold = 0.5
silence_duration_ms = 1500
```

1500 ms помогало не разрывать русские технические фразы слишком рано.

Не менять VAD параметры без voice regression test.

---

# 18. TECHNICAL STT

Realtime STT может искажать технические термины.

Наблюдались варианты для:

```text
AngioPlus Core
μFR
DICOM
AE Title
```

Например DICOM в реальных тестах распознавался как:

```text
тиком
дикомфорт
медиком
```

Несмотря на это, Realtime иногда способен правильно восстановить domain intent при формировании tool query.

Пример:

```text
raw transcript:
"медиком порт используется по умолчанию"

tool query:
"Какой порт DICOM используется по умолчанию в AngioPlus Core"
```

Это полезное свойство текущей архитектуры.

Но STT normalization остаётся областью будущего improvement.

---

# 19. STT DATASETS

Подготовлены datasets:

```text
AngioPlus_STT_Training_Dataset_v1.xlsx
AngioPlus_STT_Training_Dataset_v2.xlsx
```

Они предназначены для future STT evaluation / adaptation.

Они не являются частью runtime retrieval.

---

# 20. NATIVE REALTIME FILE SEARCH — ИСТОРИЯ

Yandex native File Search был успешно подключён и протестирован.

Он доказал, что концепция:

```text
Realtime
→ documentation retrieval
→ grounded answer
```

работоспособна.

Однако native File Search больше НЕ является целевым production retrieval mechanism.

---

# 21. ПОЧЕМУ МЫ ОТКАЗАЛИСЬ ОТ NATIVE FILE SEARCH КАК PRIMARY RETRIEVAL

Основные причины:

1. Query formulation скрыт.
2. Retrieved chunks плохо наблюдаемы из Realtime event stream.
3. Не получен надёжный доступ к `queries[]` / `results[]`.
4. File Search item ID не был доступен в реальных Realtime тестах.
5. Поэтому нельзя надёжно управлять retrieval history.
6. Retrieved chunks увеличивают active conversation context.
7. `tool_choice="required"` не гарантировал fresh retrieval на каждом factual turn.
8. DICOM retrieval давал воспроизводимые ошибки даже при здоровом Vector Store.
9. Direct Vector Store Search на том же запросе находил правильный evidence.

Ключевой вывод:

```text
Yandex Vector Store хороший.

Проблема была не в отказе от Yandex,
а в недостатке контроля native Realtime File Search.
```

Поэтому выбран:

```text
Custom Function Calling
+
Direct Yandex Vector Store Search API
```

---

# 22. NATIVE FILE SEARCH CONTEXT GROWTH

На старой модели 250923 File Search заметно увеличивал context.

Пример:

```text
startup:
~3410 input tokens

after frame-rate:
~9718

after ROI:
~16233
```

Два retrieval добавили примерно:

```text
12.8k active tokens
```

Даже при:

```text
max_num_results = 5
```

retrieval мог добавлять примерно:

```text
4k–6k tokens
```

Это стало ещё одной причиной не использовать opaque native File Search как production retrieval layer.

---

# 23. CURRENT RETRIEVAL ARCHITECTURE

Текущий production candidate:

```text
Yandex Realtime 260528
        ↓
search_angioplus_knowledge
        ↓
Python RetrievalController
        ↓
Yandex Vector Store Search API
        ↓
TOP-5
        ↓
function_call_output
        ↓
Realtime final answer
```

Это уже НЕ planned experiment.

Эта цепочка реализована и работает.

---

# 24. CUSTOM FUNCTION

Tool:

```text
search_angioplus_knowledge
```

При factual AngioPlus question Realtime может сформировать standalone retrieval query.

Пример:

```json
{
  "query": "Какой порт DICOM используется по умолчанию в AngioPlus Core"
}
```

Python выполняет retrieval и возвращает evidence.

После:

```text
function_call_output
```

отправляется:

```text
response.create
```

и Realtime формирует grounded answer.

---

# 25. RETRIEVAL CONTROLLER

Основной файл:

```text
yandex/retrieval/controller.py
```

Models:

```text
yandex/retrieval/models.py
```

Vector Store client:

```text
yandex/retrieval/client.py
```

Текущий controller:

```text
natural query
      ↓
Yandex Vector Store Search
      ↓
TOP-5
      ↓
metadata parsing
      ↓
EvidenceChunk[]
```

Текущий default:

```text
max_num_results = 5
```

---

# 26. RETRIEVAL RESPONSE

`EvidenceChunk` содержит:

```text
rank
score
text
source
source_type
authority
section
pages
file_id
vector_store_filename
attributes
```

`RetrievalResponse` содержит:

```text
query
evidence
latency_ms
total_candidates
```

Эта структура должна сохраняться как основа future observability / source traceability.

---

# 27. SOURCE AUTHORITY

Текущие веса controller:

```text
IFU        100
TRAINING    80
FAQ         70
COMMERCIAL  40
COMPETITOR  30
UNKNOWN     10
```

Authority сейчас является metadata.

Не сортировать результаты слепо только по authority.

Semantic relevance остаётся обязательной.

Общий conceptual priority:

```text
IFU
>
official service/admin
>
Known Issues / Release Notes
>
L1/L2
>
FAQ / Training
>
Commercial
>
Scientific/reference
```

---

# 28. CURRENT VECTOR STORE

Текущий retrieval store:

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

Corpus:

```text
2025July.pdf
AngioPlus Core v2.5 IFU_RU_Final.pdf
Common Info to Distributor - commercial (1).pdf
Competitor_2025.pdf
Q&A List ENG.xlsx
```

Этот store является текущим baseline.

---

# 29. ORIGINAL YANDEX STORE

Исторический store:

```text
AI Studio:
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
```

Использовался в ранних экспериментах.

Не путать его с текущим V2.

---

# 30. ПОЧЕМУ L1/L2 НЕ В V2

Из V2 был исключён:

```text
Development_Questions_L1_L2.docx
```

Причина:

документ преимущественно содержит вопросы, а не authoritative answers, и создавал retrieval noise.

Это не означает, что L1/L2 навсегда запрещён.

Если появится качественный authoritative L1/L2 corpus, его можно вернуть после evaluation.

---

# 31. SCIENTIFIC PAPERS

Scientific papers не входят в основной V2.

Причина:

- retrieval noise;
- не являются основным authoritative support source;
- текущий classifier может классифицировать их неправильно.

В будущем возможно:

```text
SCIENTIFIC_EVIDENCE
```

или отдельный store.

Не добавлять scientific corpus в основной retrieval без regression testing.

---

# 32. V2 INGESTION

Файл:

```text
yandex/retrieval/ingest.py
```

Используются pre-created JSONL chunks.

Формат:

```json
{"body":"<chunk_content>"}
```

Upload:

```python
with open(jsonl_path, "rb") as f:
    file_obj = client.files.create(
        file=(jsonl_path.name, f, "application/jsonlines"),
        purpose="assistants",
        extra_body={"format": "chunks"},
    )
```

Важно:

```text
format=chunks
```

задаётся при FILE UPLOAD.

Это не Vector Store file attribute.

---

# 33. V2 CHUNKS

Результат ingestion:

```text
2025July.pdf
4 chunks

AngioPlus Core v2.5 IFU_RU_Final.pdf
36 chunks

Common Info to Distributor - commercial (1).pdf
1 chunk

Competitor_2025.pdf
7 chunks

Q&A List ENG.xlsx
68 chunks
```

Всего:

```text
116 chunks
```

---

# 34. SEMANTIC CHUNKER

В `SemanticChunker` существует:

```text
overlap_tokens=50
```

но текущая split logic фактически не использует overlap.

Не исправлять автоматически.

V2 уже показывает хорошее retrieval quality.

Изменять chunking только после benchmark, показывающего необходимость.

---

# 35. STANDALONE VECTOR STORE SEARCH

Direct Vector Store Search API успешно используется controller.

Conceptually:

```text
POST /v1/vector_stores/{vector_store_id}/search
```

Текущие параметры:

```text
natural query
rewrite_query = false
max_num_results = 5
ranker = auto
score_threshold = 0
```

Не вводить score threshold без regression evaluation.

---

# 36. QUERY REWRITER — ИСТОРИЧЕСКИЙ ЭКСПЕРИМЕНТ

YandexGPT Lite Query Rewriter был успешно протестирован.

Модель:

```text
gpt://<folder_ID>/yandexgpt-5-lite
```

Он улучшал некоторые сложные queries:

- table movement;
- frame rate;
- ROI.

Однако после V2 indexing natural query baseline существенно улучшился.

Поэтому Query Rewriter сейчас:

```text
PLAN B
```

а не часть текущего runtime.

Не добавлять его обратно без benchmark, показывающего необходимость.

---

# 37. RERANKER — ИСТОРИЧЕСКИЙ ЭКСПЕРИМЕНТ

YandexGPT Lite reranker также тестировался.

Он помогал, если правильный chunk уже находился среди candidates.

Mental model:

```text
Correct chunk in candidate set
        ↓
reranker может улучшить ranking

Correct chunk отсутствует
        ↓
reranker бессилен
```

После V2 baseline отдельный LLM reranker не используется в current runtime.

Это также:

```text
PLAN B
```

---

# 38. CURRENT RETRIEVAL BENCHMARK

Актуальный baseline Retrieval Controller:

```text
Evidence Hit@1: 13/19
Evidence Hit@3: 17/19
Evidence Hit@5: 19/19

Preferred Hit@1: 13/19
Preferred Hit@3: 17/19
Preferred Hit@5: 18/19

Median latency:
~1540 ms

P95:
~2300 ms
```

Главный результат:

```text
Evidence Hit@5 = 19/19
```

Поэтому текущий runtime сохраняет:

```text
full TOP-5 evidence
```

---

# 39. TOP-3 EXPERIMENT

Была проверена идея сократить evidence до TOP-3.

Результат:

```text
17/19
```

Это ухудшило evidence recall.

Решение:

```text
TOP-3 rejected
```

---

# 40. LEXICAL COMPACTION EXPERIMENT

Была протестирована lexical evidence compaction.

Она уменьшала payload примерно на:

```text
66%
```

но evidence quality падала примерно до:

```text
15/19
```

Решение:

```text
lossy lexical compaction rejected
```

Current runtime:

```text
full TOP-5 evidence
```

Позже допустима только lossless metadata cleanup, если понадобится.

---

# 41. CUSTOM REALTIME TEST — DICOM

Ключевой acceptance test:

```text
Какой DICOM порт используется по умолчанию?
```

Realtime сформировал:

```text
search_angioplus_knowledge
```

Query:

```text
Какой порт DICOM используется по умолчанию в AngioPlus Core
```

Vector Store вернул TOP-5 IFU chunks.

TOP1 score в одном из тестов:

```text
~0.999
```

Final answer:

```text
По умолчанию в AngioPlus Core используется порт 1108 для DICOM.
```

Это является подтверждением работоспособности центрального backend loop.

---

# 42. FOLLOW-UP CONTEXT

Realtime способен использовать conversation context при tool query generation.

Например:

```text
User:
"какой дикомфорт используется по умолчанию"

Assistant:
"Вы имеете в виду AE Title или порт?"

User:
"порт"
```

Realtime затем самостоятельно сформировал standalone query:

```text
Какой порт DICOM используется по умолчанию в AngioPlus Core
```

и вызвал retrieval.

Это хороший результат.

Поэтому старую follow-up rewrite систему пока НЕ переносить.

---

# 43. FRESH RETRIEVAL — ИЗВЕСТНЫЙ TECHNICAL DEBT

Custom retrieval работает.

Но Realtime пока НЕ гарантирует новый function call на каждом повторном factual question.

Наблюдалось:

```text
first DICOM question
→ function call
→ retrieval
→ 1108

later repeated DICOM question
→ direct answer 1108
→ no new function call
```

То есть модель иногда использует факт из conversation context.

Для demo MVP:

```text
NOT BLOCKING
```

Для production:

```text
P1
```

До production необходимо гарантировать fresh retrieval для factual product answers либо доказать безопасную альтернативную policy.

Не считать эту проблему решённой.

---

# 44. TOOL_CHOICE EXPERIMENTS

Исторически тестировалось:

```text
tool_choice = required
```

Это НЕ обеспечило жёсткую гарантию fresh retrieval на каждом повторном factual turn.

Поэтому не считать session-level `required` достаточным решением.

Будущий вариант для исследования:

```text
per-response tool forcing
```

или более жёсткая orchestration policy.

Не реализовывать прямо сейчас, если это не требуется для текущего MVP milestone.

---

# 45. TURN MANAGER

При custom Function Calling была обнаружена критическая Realtime state-machine ошибка.

Ошибка:

```text
Invalid request:
illegal command FinalizeUserUtterance
in state WAITING_FOR_TOOL_RESULT
```

Она возникала, если пользователь начинал/продолжал говорить, пока Realtime ожидал tool result.

Для mitigation реализован:

```text
TurnManager
```

Основные states:

```text
READY
TOOL_PENDING
WAITING_FOR_RESPONSE
```

---

# 46. TURN MANAGER FLOW

Во время:

```text
TOOL_PENDING
```

browser PCM временно не отправляется в Yandex.

Он сохраняется в локальный buffer.

После:

```text
function_call_output
+
response.create
```

state:

```text
WAITING_FOR_RESPONSE
```

После post-tool:

```text
response.created
```

buffer отправляется в Yandex и state возвращается:

```text
READY
```

Эта mitigation реально предотвратила известный disconnect в тестах.

---

# 47. TURN MANAGER BUFFER LIMIT

Текущий local audio buffer:

```text
~5 seconds
```

В нормальных retrieval tests это работает.

Однако при длинном retrieval наблюдалось:

```text
audio buffer full
dropped ~0.05s oldest audio
```

То есть при retrieval >5 секунд начало buffered speech может потеряться.

Для demo MVP:

```text
NOT BLOCKING
```

Для production:

```text
P1 / UX issue
```

---

# 48. RETRIEVAL LATENCY

Обычные наблюдавшиеся retrieval timings:

```text
~1.5–3.5 seconds
```

Но был реальный spike:

```text
~11.7 seconds
```

Один spike недостаточен для архитектурного redesign.

Перед production необходимо собрать:

```text
p50
p95
p99
```

и определить причину редких spikes.

Для demo MVP latency пока считается приемлемой.

---

# 49. VOICE CONFIG ISSUE

В коде запрашивается:

```text
voice = dasha
```

но в `response.done` Yandex сообщает:

```text
voice = kirill
```

Это известная проблема / конфигурационная особенность.

Она не блокирует backend MVP.

Разобраться позже в рамках frontend/voice polish.

---

# 50. GREETING ISSUE

Current Realtime startup иногда отвечает:

```text
Пожалуйста, уточните ваш вопрос.
```

вместо нормального приветствия.

Это UX issue.

Не является архитектурным blocker.

Исправить при frontend / conversation UX polish.

---

# 51. MEDICAL SAFETY

Ассистент предназначен для технической поддержки.

Он не должен самостоятельно:

- выбирать лечение;
- рекомендовать медицинское вмешательство;
- принимать решение о PCI;
- принимать решение о stent;
- заменять врача;
- интерпретировать clinical result как самостоятельное медицинское решение.

Даже если retrieval содержит клинический текст:

```text
retrieved
≠
allowed to recommend
```

---

# 52. CREDENTIAL SAFETY

Knowledge Base может содержать строки, похожие на:

- passwords;
- admin credentials;
- service credentials;
- internal access data.

Наличие их в retrieved chunk не означает разрешение на раскрытие.

До production должна существовать явная policy.

Default principle:

```text
sensitive credentials
→ do not expose
```

---

# 53. KNOWLEDGE BASE PRINCIPLE

Production Knowledge Base не должна быть:

```text
"все документы, которые удалось найти"
```

Каждый документ должен иметь понятный статус:

- authoritative?
- current?
- safe to expose?
- correct product/version?
- contains credentials?
- contains clinical ambiguity?
- contains only questions?
- creates retrieval noise?

Corpus quality важнее corpus size.

---

# 54. CANONICAL DOCUMENT DIRECTORY

Основной source directory:

```text
data/training_materials/
```

Tracked documents включают:

```text
2025July.pdf
AngioPlus Core v2.5 IFU_RU_Final.pdf
Common Info to Distributor - commercial (1).pdf
Competitor_2025.pdf
Development_Questions_L1_L2.docx
Q&A List ENG.xlsx
```

Temporary V2 corpus:

```text
data/yandex_v2_test/
```

может игнорироваться Git, поскольку содержит duplicates canonical sources.

---

# 55. GENERATED ARTIFACTS

Diagnostics:

```text
artifacts/diagnostics/
```

Debug outputs:

```text
artifacts/debug/
```

Generated TTS:

```text
data/tts_out/
```

Generated artifacts не должны попадать в Git.

---

# 56. BENCHMARKS

Yandex retrieval benchmarks находятся в:

```text
benchmarks/yandex/
```

Они предназначены для retrieval evaluation, а не обязательно являются pytest unit tests.

Не удалять benchmark history без причины.

---

# 57. TEST HISTORY

После repository cleanup targeted tests проходили:

```text
29 passed
```

Full suite исторически:

```text
3 failed
214 passed
7 warnings
```

Три известных failure относились к старому voice/STT stack.

Не чинить LEGACY tests в рамках unrelated Yandex/frontend work.

---

# 58. CURRENT GIT CHECKPOINT

Текущая backend development branch:

```text
feature/yandex-retrieval-controller
```

Ключевой backend MVP commit:

```text
7bc6ec7
feat: integrate custom retrieval with Yandex Realtime
```

Этот commit содержит рабочий milestone:

```text
speech-realtime-260528
+
custom Function Calling
+
RetrievalController
+
Vector Store V2
+
TurnManager
```

Commit успешно push-нут в remote repository.

На момент checkpoint:

```text
working tree clean
```

---

# 59. ВАЖНЫЕ ПРЕДЫДУЩИЕ COMMITS

Repository cleanup:

```text
3046267
chore: organize project scripts and Yandex prototypes
```

```text
c6695f6
chore: remove obsolete root Dockerfile and requirements
```

Architecture/docs:

```text
55da700
docs: update project architecture and development context
```

Yandex retrieval baseline:

```text
9d058c3
feat: add Yandex vector retrieval baseline
```

MVP docs:

```text
0d2bc2a
docs: add MVP status and roadmap
```

Evidence experiments:

```text
8c8fabf
test: evaluate top3 evidence selection
```

```text
993b49c
test: evaluate lexical evidence compaction
```

Latest major backend milestone:

```text
7bc6ec7
feat: integrate custom retrieval with Yandex Realtime
```

---

# 60. BACKEND MVP FREEZE

На текущем этапе backend считается:

```text
FUNCTIONALLY COMPLETE FOR WEB MVP DEMONSTRATION
```

Поэтому сейчас НЕ нужно продолжать бесконечно улучшать backend.

Known issues фиксируются как technical debt.

Backend меняется только если:

1. frontend integration требует минимального API change;
2. acceptance testing обнаруживает фундаментальную проблему;
3. обнаруживается safety issue;
4. появляется blocker для MVP.

Иначе:

```text
BACKEND FREEZE
```

---

# 61. ТЕКУЩИЙ DEVELOPMENT PRIORITY

Текущий следующий milestone:

```text
WEB ASSISTANT FRONTEND
```

Цель:

создать нормальный демонстрационный интерфейс AngioPlus Assistant.

Пользователь должен иметь возможность:

```text
1. написать вопрос;
2. получить текстовый ответ;
3. видеть историю разговора;
4. нажать кнопку Voice;
5. разговаривать голосом;
6. видеть voice transcripts в том же chat thread.
```

---

# 62. FRONTEND STRATEGY

Не создавать chat UI полностью с нуля.

Выбрано направление:

```text
assistant-ui
```

как готовая open-source frontend foundation.

Причины:

- production-oriented AI chat UI;
- готовые message/thread/composer components;
- streaming support;
- tool UI possibilities;
- markdown;
- attachments / feedback extensibility;
- возможность собственного backend;
- React ecosystem;
- пригодность для дальнейшего развития продукта.

---

# 63. FRONTEND TARGET

Предполагаемая структура:

```text
angioplus-core-rag/
│
├── yandex/
│   └── realtime/
│       └── web/
│           ├── server.py
│           └── static/
│               └── index.html
│
└── frontend/
    └── angioplus-assistant/
```

Текущий:

```text
yandex/realtime/web/static/index.html
```

содержит рабочий voice prototype.

Его пока не ломать.

Новый frontend сначала поднять отдельно.

---

# 64. FRONTEND UX TARGET

Концепция:

```text
┌─────────────────────────────────────────────┐
│            AngioPlus Assistant              │
│                                             │
│  User: Какой DICOM порт используется?       │
│                                             │
│  AI: По умолчанию используется порт 1108.  │
│                                             │
│  [ Напишите вопрос...              ] [Send] │
│                                             │
│             [ Start Voice ]                 │
└─────────────────────────────────────────────┘
```

Voice и text должны в перспективе работать в одном conversation experience.

Voice transcripts должны отображаться в chat history.

---

# 65. FRONTEND IMPLEMENTATION PRINCIPLE

Первый этап:

```text
поднять assistant-ui локально
```

БЕЗ изменения рабочего Yandex backend.

Сначала оценить:

- внешний вид;
- UX;
- component structure;
- suitability.

После подтверждения:

```text
assistant-ui
      ↓
наш backend
      ↓
Yandex
```

Не переписывать одновременно frontend и backend.

---

# 66. NODE.JS STATUS

На момент последней проверки:

```text
node
npm
```

не были установлены локально.

Следующий infrastructure step для frontend:

```text
install Node.js LTS
```

После установки проверить:

```bash
node --version
npm --version
```

---

# 67. FRONTEND GIT STRATEGY

Backend milestone уже зафиксирован отдельно.

Для frontend рекомендуется отдельная branch:

```text
feature/web-assistant-ui
```

Создавать её от актуального backend checkpoint.

Это позволяет:

- сохранить рабочий backend milestone;
- экспериментировать с frontend независимо;
- делать frontend commits отдельно;
- легко откатить UI experiments.

---

# 68. TELEPHONY STATUS

Телефония сознательно:

```text
POSTPONED
```

Не начинать сейчас:

- SIP;
- Plusofon;
- ordinary +7 number;
- production media bridge.

Причина:

сначала необходимо получить демонстрируемый и тестируемый Web Assistant.

После стабильного Web MVP телефония станет дополнительным transport/channel поверх уже проверенного assistant core.

---

# 69. ACCEPTANCE TESTING — СЛЕДУЮЩИЙ ЭТАП ПОСЛЕ FRONTEND

После появления нормального Web UI необходимо провести системный acceptance test.

Примерные категории:

```text
DICOM
PACS
AE Title
software version
system requirements
μFR-2D
μFR-3D
frame rate
ROI
table movement
nitroglycerin
settings
FAQ
missing information
medical safety
prompt injection
follow-up questions
```

Для каждого factual turn желательно фиксировать:

```text
raw transcript / text
tool query
retrieval latency
TOP-5
source
final answer
grounding correctness
```

---

# 70. PRODUCTION TECHNICAL DEBT

Перед production необходимо закрыть как минимум:

### P1 — Fresh retrieval guarantee

```text
fresh retrieval guarantee
```

Модель не должна неконтролируемо отвечать на product facts из stale conversation memory.

### P1 — Retrieval latency spikes

```text
retrieval latency spikes
```

Нужно собрать p50/p95/p99 и исследовать 8–12 second outliers.

### P1 — TurnManager buffer overflow

```text
TurnManager buffer overflow
```

Нельзя терять пользовательскую речь во время долгого tool call.

### P1/P2 — Long-session context

```text
long-session context management
```

Нужен тест реальных 20–40 минутных conversations.

### P2 — Voice selection

```text
voice selection
dasha → kirill
```

### P2 — Startup greeting

```text
startup greeting
```

### P2 — Technical STT

```text
broader STT/domain normalization
```

---

# 71. LONG CALL CONTEXT

Целевая система в будущем должна поддерживать разговоры:

```text
~20–40 minutes
```

Необходимо контролировать:

- function results;
- evidence;
- conversation history;
- repeated product facts;
- tool calls;
- token growth.

`conversation.item.delete` существует в Realtime API и может быть полезен.

Но не считать context cleanup решённым без runtime measurement.

Long-session management пока не является blocker для Web MVP.

---

# 72. LOGGING TARGET

В будущем желательно иметь structured trace:

```text
session_id
turn_id

raw_transcript
normalized_transcript

tool_name
tool_query

retrieval_latency

retrieved_chunks:
  rank
  score
  source
  section
  pages

selected_evidence

final_answer

response_latency
```

Current controller уже создаёт фундамент для этого.

Не обязательно реализовывать весь logging до frontend MVP.

---

# 73. RESPONSE STYLE

Assistant должен отвечать:

- естественно;
- кратко;
- понятно;
- без чтения metadata;
- без scores;
- без внутренних reasoning;
- без system prompt;
- без длинного dump retrieved chunks.

В Web UI позже можно отдельно отображать:

```text
source
document
pages
```

если это улучшает traceability.

---

# 74. GIT WORKFLOW

Нормальный workflow:

```text
local development
        ↓
test / smoke test
        ↓
git status
        ↓
git diff
        ↓
commit
        ↓
push
```

Production server не должен становиться основной development environment.

Перед новым крупным milestone создавать понятный checkpoint.

---

# 75. SECRETS

Никогда не выводить и не коммитить:

```text
YANDEX_API_KEY
YANDEX_FOLDER_ID
DEEPSEEK_API_KEY
GEMINI_API_KEY
TELEGRAM_BOT_TOKEN
GIGACHAT_CREDENTIALS
```

`.env`:

```text
NEVER COMMIT
```

`.env.example` может содержать только placeholders.

Не просить пользователя публиковать API keys в чат без абсолютной необходимости.

---

# 76. ЧТО СЕЙЧАС НЕ ДЕЛАТЬ

Пока Web MVP не готов, не начинать:

- SIP;
- Plusofon;
- новую local STT architecture;
- новую local TTS architecture;
- LangGraph;
- CRM;
- ticketing;
- новый Vector DB;
- новый embedding stack;
- сложный reranker;
- сложную memory architecture;
- масштабную backend optimization.

Known backend issues записаны.

Не превращать их автоматически в текущий priority.

---

# 77. КРИТЕРИЙ УСПЕХА WEB MVP

Web MVP считается успешным, если пользователь может:

```text
открыть AngioPlus Assistant
        ↓
написать технический вопрос
        ↓
получить grounded ответ
```

и:

```text
нажать Voice
        ↓
задать тот же вопрос голосом
        ↓
получить голосовой grounded ответ
```

при этом:

```text
conversation visible in UI
```

и backend использует текущий AngioPlus retrieval pipeline.

---

# 78. ГРАНИЦА MVP И PRODUCTION

Очень важно не смешивать эти понятия.

## MVP DEMO

Достаточно:

```text
working chat
working voice
working custom retrieval
correct evidence
grounded answers
reasonable UX
```

## PRODUCTION

Дополнительно обязательно:

```text
fresh retrieval guarantee
long-session reliability
latency SLA
robust barge-in
audio buffer reliability
security
credential policy
medical safety validation
observability
analytics
versioned KB
authentication
human escalation
telephony reliability
deployment/monitoring
```

Backend сейчас находится на уровне:

```text
MVP DEMO READY
```

а не:

```text
PRODUCTION READY
```

---

# 79. CURRENT CHECKPOINT

Если разработка продолжается после паузы, текущая точка:

```text
BACKEND MVP COMPLETE FOR DEMO
```

Рабочий backend:

```text
Yandex Realtime 260528
        ↓
custom function:
search_angioplus_knowledge
        ↓
RetrievalController
        ↓
Yandex Vector Store V2
        ↓
TOP-5 evidence
        ↓
grounded response
```

Current Vector Store:

```text
angioplus-kb-v2
fvtpohd7u6dnbu8f44ah
```

Backend checkpoint:

```text
7bc6ec7
feat: integrate custom retrieval with Yandex Realtime
```

---

# 80. ТОЧКА ПРОДОЛЖЕНИЯ

НЕ возвращаться к standalone retrieval experiments как к текущей задаче.

Они уже дали достаточный результат для MVP.

НЕ возвращаться к native File Search.

НЕ возвращаться к old LiveKit stack.

НЕ начинать telephony.

Продолжать отсюда:

```text
CREATE WEB ASSISTANT FRONTEND
```

План:

```text
1. Create feature/web-assistant-ui branch

2. Install Node.js LTS

3. Start assistant-ui frontend separately

4. Evaluate ready-made UI

5. Preserve existing working voice prototype

6. Connect text chat to AngioPlus backend

7. Integrate existing realtime voice mode

8. Show text + voice turns in one conversation UI

9. Run AngioPlus acceptance suite

10. Only then return to backend hardening
```

---

# 81. ФИНАЛЬНАЯ ИНСТРУКЦИЯ ДЛЯ AI CODING AGENT

При новой задаче:

1. Определи CURRENT / CONTROL / LEGACY.
2. Не развивай LEGACY как текущую архитектуру.
3. Не удаляй CONTROL без отдельного решения.
4. Не возвращай native File Search как primary retrieval без новых доказательств.
5. Используй `speech-realtime-260528` как текущий baseline.
6. Используй V2 Vector Store как current retrieval store.
7. Не добавляй Query Rewriter / reranker без benchmark.
8. Не сокращай TOP-5 evidence lossy-методами без regression test.
9. Не усложняй Python controller без измеримой причины.
10. Не начинай telephony, пока Web MVP не завершён.
11. Проверяй актуальную официальную Yandex API documentation при изменении API integration.
12. Делай один логический шаг за раз.
13. После изменения проверяй runtime.
14. Проверяй Git diff.
15. Не коммить generated artifacts.
16. Не раскрывай secrets.
17. Не позволяй retrieval обходить medical safety.
18. При недостатке информации предпочитай clarification / refusal / escalation.
19. Помни разницу между MVP-ready и production-ready.
20. Сохраняй архитектуру настолько простой, насколько это возможно.

---

# 82. ГЛАВНЫЙ ПРИНЦИП ПРОЕКТА

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

Текущий принцип разработки:

```text
Make it work
    ↓
Make it measurable
    ↓
Make the MVP usable
    ↓
Test it broadly
    ↓
Then harden for production
```