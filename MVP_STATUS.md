# AngioPlus AI Assistant — MVP Status & Roadmap

**Последнее обновление:** 2 сентября 2026  
**Текущий этап:** Web MVP / Frontend  
**Backend status:** MVP Demo Ready

---

# 1. Цель проекта

Цель проекта — создать AI-ассистента технической поддержки для медицинского программного обеспечения **AngioPlus Core**.

Ассистент должен позволять пользователю взаимодействовать с системой:

```text
текстом
+
голосом
```

и получать ответы на технические вопросы на основании утверждённой документации AngioPlus.

Ближайший продуктовый milestone:

```text
AngioPlus Web Assistant
```

с двумя режимами:

```text
Text Chat
+
Realtime Voice
```

В дальнейшем тот же assistant core должен поддерживать:

```text
Phone / SIP
```

Телефония сознательно отложена до завершения Web MVP.

---

# 2. Текущий статус проекта

На текущем этапе главный технический риск MVP уже снят.

Доказано, что основной backend loop работает end-to-end:

```text
User
 ↓
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

Поэтому текущий статус backend:

```text
FUNCTIONALLY COMPLETE FOR MVP DEMONSTRATION
```

Это НЕ означает:

```text
PRODUCTION READY
```

Следующий основной этап:

```text
WEB ASSISTANT FRONTEND
```

---

# 3. Что означает MVP на текущем этапе

Текущий MVP — это демонстрационная версия продукта, позволяющая показать полный пользовательский сценарий:

```text
пользователь открывает Web Assistant
        ↓
задаёт вопрос текстом или голосом
        ↓
assistant понимает вопрос
        ↓
при необходимости выполняет retrieval
        ↓
получает evidence из AngioPlus Knowledge Base
        ↓
формирует grounded answer
        ↓
отвечает текстом или голосом
```

Для Web MVP пока НЕ требуется полностью закрыть:

- production telephony;
- CRM;
- ticketing;
- admin panel;
- billing;
- multi-tenant architecture;
- production analytics;
- long-session optimization;
- полноценный monitoring;
- automatic human escalation;
- production SLA.

Эти задачи относятся к следующим этапам.

---

# 4. MVP vs Production

Очень важно разделять два уровня готовности.

## MVP Demo

Необходимо:

```text
working text chat
working realtime voice
working custom retrieval
correct evidence
grounded answers
follow-up context
reasonable latency
usable frontend
```

## Production

Дополнительно необходимо:

```text
fresh retrieval guarantee
long-session reliability
latency SLA
robust barge-in
audio buffering reliability
security hardening
credential policy
medical safety validation
structured observability
analytics
versioned knowledge base
authentication
human escalation
production deployment
telephony reliability
```

Текущий backend находится на уровне:

```text
MVP DEMO READY
```

---

# 5. Research / Architecture

## Status

```text
DONE
```

В ходе проекта были исследованы:

- собственный RAG;
- Qdrant;
- BAAI/bge-m3;
- DeepSeek;
- Telegram Bot;
- LiveKit;
- Silero VAD;
- faster-whisper;
- Qwen3-TTS;
- Yandex Realtime;
- Yandex native File Search;
- Yandex Vector Store;
- pre-created chunks;
- Query Rewriter;
- LLM reranking;
- custom Function Calling;
- direct Vector Store Search.

Основной архитектурный вопрос:

> Можно ли построить качественного voice/text technical support assistant на Yandex-first архитектуре?

считается положительно подтверждённым.

---

# 6. Предыдущий собственный RAG

## Status

```text
CONTROL / COMPLETE
```

Была создана собственная RAG-система:

```text
User
 ↓
Python RAG
 ↓
BAAI/bge-m3
 ↓
Qdrant
 ↓
DeepSeek
 ↓
Answer
```

Дополнительно существовали:

- query normalization;
- follow-up rewriting;
- system prompt;
- Telegram Bot;
- voice integrations.

Этот stack работал и остаётся в repository как:

```text
CONTROL
```

Он больше не является основной архитектурой.

---

# 7. Legacy Voice Stack

## Status

```text
LEGACY
```

Предыдущая voice architecture:

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

Она была полезна для исследования, но оказалась значительно сложнее текущего Yandex Realtime подхода.

Она больше не является target architecture.

---

# 8. Yandex Realtime

## Status

```text
MVP READY
```

Текущая модель:

```text
speech-realtime-260528
```

Работают:

- realtime audio input;
- realtime speech recognition;
- voice generation;
- server-side VAD;
- natural Russian dialogue;
- follow-up context;
- interruption / barge-in;
- browser WebSocket gateway;
- custom Function Calling.

Рабочий audio format:

```text
PCM16
44100 Hz
```

---

# 9. Browser Voice Prototype

## Status

```text
MVP READY
```

Рабочая архитектура:

```text
Browser
  ↕
WebSocket
  ↕
Python gateway
  ↕
Yandex Realtime 260528
```

Backend:

```text
yandex/realtime/web/server.py
```

Текущий простой frontend:

```text
yandex/realtime/web/static/index.html
```

Этот prototype уже позволяет проводить реальные voice tests без SIP/телефонии.

---

# 10. Native Yandex File Search

## Status

```text
TESTED
NOT SELECTED AS PRIMARY RETRIEVAL
```

Built-in Realtime File Search был успешно подключён и протестирован.

Он доказал жизнеспособность:

```text
Realtime
→ retrieval
→ grounded answer
```

Но выявил ограничения:

- недостаточная наблюдаемость search query;
- недостаточная наблюдаемость retrieved chunks;
- ограниченный контроль ranking;
- сложный source audit;
- retrieval results увеличивают active conversation context;
- повторные searches увеличивают token usage;
- `tool_choice="required"` не дал гарантии fresh retrieval;
- наблюдались воспроизводимые retrieval failures.

Поэтому native File Search больше не является основным production candidate.

---

# 11. Текущая Retrieval Architecture

## Status

```text
MVP READY
```

Выбрана и реализована архитектура:

```text
Yandex Realtime 260528
        ↓
search_angioplus_knowledge(...)
        ↓
Python RetrievalController
        ↓
Yandex Vector Store Search API
        ↓
TOP-5 evidence
        ↓
function_call_output
        ↓
Yandex Realtime
```

Это больше не эксперимент.

Эта цепочка реально работает end-to-end.

---

# 12. Knowledge Base V2

## Status

```text
MVP READY
```

Текущий Vector Store:

```text
AI Studio:
angioplus-kb-v2

Vector Store ID:
fvtpohd7u6dnbu8f44ah
```

Содержит:

```text
5 documents
116 chunks
```

Corpus:

```text
2025July.pdf

AngioPlus Core v2.5 IFU_RU_Final.pdf

Common Info to Distributor - commercial (1).pdf

Competitor_2025.pdf

Q&A List ENG.xlsx
```

---

# 13. Исключённые документы

`Development_Questions_L1_L2.docx` сознательно не входит в текущий V2 corpus.

Причина:

```text
в основном вопросы
+
мало authoritative answers
+
retrieval noise
```

Это не означает, что L1/L2 documentation запрещена навсегда.

Качественный authoritative L1/L2 corpus можно добавить позже.

Scientific papers также пока не входят в основной Vector Store.

---

# 14. Pre-created Chunking

## Status

```text
DONE
```

Используется загрузка заранее подготовленных chunks через JSONL:

```json
{"body":"<chunk_content>"}
```

с:

```text
application/jsonlines
format=chunks
```

Это позволяет контролировать preprocessing и chunk boundaries до загрузки в Vector Store.

Текущий V2:

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

# 15. Retrieval Controller

## Status

```text
MVP READY
```

Основные файлы:

```text
yandex/retrieval/client.py

yandex/retrieval/controller.py

yandex/retrieval/models.py
```

Текущий pipeline:

```text
natural query
      ↓
Yandex Vector Store Search
      ↓
TOP-5
      ↓
metadata parsing
      ↓
source classification
      ↓
structured evidence
```

Default:

```text
max_num_results = 5
```

Controller не использует обязательный дополнительный LLM call.

---

# 16. Evidence Model

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

Эта структура является основой будущих:

- logs;
- source traceability;
- diagnostics;
- analytics;
- retrieval evaluation.

---

# 17. Source Authority

Текущие metadata weights:

```text
IFU        100
TRAINING    80
FAQ         70
COMMERCIAL  40
COMPETITOR  30
UNKNOWN     10
```

Authority не заменяет semantic relevance.

Текущий принцип:

```text
semantic relevance
+
source authority
+
safety policy
```

а не:

```text
highest authority always wins
```

---

# 18. Retrieval Benchmark

## Status

```text
MVP READY
```

Текущий benchmark:

```text
Evidence Hit@1: 13/19
Evidence Hit@3: 17/19
Evidence Hit@5: 19/19

Preferred Hit@1: 13/19
Preferred Hit@3: 17/19
Preferred Hit@5: 18/19
```

Latency:

```text
Median:
~1540 ms

P95:
~2300 ms
```

Главный результат:

```text
Evidence Hit@5 = 19/19
```

То есть правильный evidence присутствует в TOP-5 для всех текущих retrieval test cases.

---

# 19. TOP-3 Experiment

## Status

```text
REJECTED
```

Была проверена передача только TOP-3 evidence.

Результат:

```text
17/19
```

Это ухудшило recall.

Поэтому current runtime использует:

```text
TOP-5
```

---

# 20. Lexical Evidence Compaction

## Status

```text
REJECTED
```

Lossy lexical compaction уменьшала payload примерно на:

```text
66%
```

но evidence quality падала примерно до:

```text
15/19
```

Поэтому текущий runtime передаёт:

```text
full TOP-5 evidence
```

---

# 21. Query Rewriter

## Status

```text
TESTED
DEFERRED / PLAN B
```

YandexGPT Lite Query Rewriter был успешно протестирован.

Он помогал на некоторых сложных queries.

Однако после создания V2 corpus natural query baseline стал достаточно сильным.

Добавление обязательного Query Rewriter сейчас:

- увеличит latency;
- увеличит стоимость;
- усложнит architecture;
- добавит дополнительный failure point.

Поэтому он не используется в current runtime.

---

# 22. LLM Reranker

## Status

```text
TESTED
DEFERRED / PLAN B
```

YandexGPT Lite reranking также был протестирован.

Главный вывод:

```text
correct evidence already in candidate set
→ reranker может помочь

correct evidence отсутствует
→ reranker не поможет
```

При текущем:

```text
Evidence Hit@5 = 19/19
```

добавлять отдельный LLM reranker в runtime пока нет необходимости.

---

# 23. Custom Function Calling

## Status

```text
MVP READY
```

Realtime использует function:

```text
search_angioplus_knowledge
```

Пример:

```text
User:
Какой DICOM порт используется по умолчанию?
```

Realtime формирует retrieval query:

```text
Какой порт DICOM используется по умолчанию в AngioPlus Core
```

После этого:

```text
Function Call
      ↓
RetrievalController
      ↓
Vector Store
      ↓
TOP-5
      ↓
function_call_output
      ↓
Realtime answer
```

---

# 24. End-to-End DICOM Test

## Status

```text
PASS
```

В реальном voice test:

```text
User:
Какой DICOM порт используется по умолчанию?
```

Retrieval:

```text
FUNCTION CALL
→ search_angioplus_knowledge
→ TOP-5 evidence
```

TOP result был из IFU.

Final answer:

```text
По умолчанию в AngioPlus Core используется порт 1108 для DICOM.
```

Это подтвердило работоспособность центрального backend architecture.

---

# 25. Follow-up Context

## Status

```text
PASS
```

Пример реального поведения:

```text
User:
какой дикомфорт используется по умолчанию

Assistant:
Вы имеете в виду AE Title или порт?

User:
порт
```

Realtime самостоятельно сформировал standalone retrieval query:

```text
Какой порт DICOM используется по умолчанию в AngioPlus Core
```

Это показывает, что current Realtime model способна использовать conversational context для tool query generation.

Поэтому старый отдельный follow-up rewriting layer пока не требуется.

---

# 26. TurnManager

## Status

```text
MVP READY
```

Во время custom Function Calling была обнаружена ошибка Realtime state machine:

```text
Invalid request:
illegal command FinalizeUserUtterance
in state WAITING_FOR_TOOL_RESULT
```

Для mitigation реализован:

```text
TurnManager
```

States:

```text
READY

TOOL_PENDING

WAITING_FOR_RESPONSE
```

Во время tool execution browser audio временно сохраняется локально.

После получения tool result и создания нового response audio снова передаётся в Yandex.

Эта mitigation успешно прошла live testing.

---

# 27. Fresh Retrieval Guarantee

## Status

```text
PRODUCTION P1
NOT BLOCKING WEB MVP
```

Известная проблема:

после первого успешного retrieval Realtime иногда отвечает на повторный factual question из conversation context без нового function call.

Пример:

```text
Question #1
→ retrieval
→ answer 1108

Question #2
→ no retrieval
→ answer 1108
```

Для демонстрационного MVP это допустимо.

Для production необходимо гарантировать fresh retrieval для product facts либо доказать безопасную альтернативную policy.

---

# 28. Retrieval Latency

## Status

```text
ACCEPTABLE FOR MVP
PRODUCTION INVESTIGATION REQUIRED
```

Current benchmark:

```text
Median:
~1.54 sec

P95:
~2.30 sec
```

В live test также наблюдался отдельный spike:

```text
~11.7 sec
```

Перед production необходимо собрать более серьёзную статистику:

```text
p50
p95
p99
```

Один latency spike не является причиной менять архитектуру на текущем этапе.

---

# 29. Audio Buffer During Tool Calls

## Status

```text
MVP ACCEPTABLE
PRODUCTION P1
```

Текущий `TurnManager` хранит примерно:

```text
5 seconds
```

входящего audio во время tool execution.

При retrieval spike ~11.7 seconds наблюдалось:

```text
audio buffer full
dropped oldest audio
```

Поэтому пользовательская речь потенциально может частично потеряться во время очень долгого retrieval.

Для Web MVP это не blocker.

Для production необходимо решить.

---

# 30. Voice Selection

## Status

```text
KNOWN ISSUE
```

В конфигурации запрашивается:

```text
dasha
```

но runtime сообщает:

```text
kirill
```

Не является blocker для MVP.

Исправить в рамках дальнейшего voice UX polish.

---

# 31. Startup Greeting

## Status

```text
KNOWN UX ISSUE
```

Вместо нормального приветствия assistant иногда говорит:

```text
Пожалуйста, уточните ваш вопрос.
```

Не является architecture issue.

Исправить при frontend / conversation UX polish.

---

# 32. Technical STT

## Status

```text
WORKING
IMPROVEMENT REQUIRED
```

Realtime STT иногда искажает:

```text
AngioPlus
μFR
DICOM
AE Title
```

Например DICOM распознавался как:

```text
тиком
дикомфорт
медиком
```

При этом Realtime model иногда успешно нормализует термин при формировании retrieval query.

Это достаточно для текущего MVP, но требует более широкого acceptance testing.

---

# 33. Medical Safety

## Status

```text
BASE POLICY EXISTS
PRODUCTION VALIDATION REQUIRED
```

Assistant предназначен для:

```text
technical support
```

Он не должен самостоятельно:

- рекомендовать лечение;
- принимать решение о PCI;
- выбирать stent;
- принимать clinical decisions;
- заменять врача;
- превращать retrieved clinical information в самостоятельную рекомендацию.

До production нужен отдельный safety acceptance suite.

---

# 34. Credential Safety

## Status

```text
PRODUCTION TODO
```

Knowledge Base потенциально может содержать:

- passwords;
- service credentials;
- admin credentials;
- internal access information.

Наличие такого текста в retrieved chunk не означает право его раскрыть.

Default policy:

```text
sensitive credentials
→ do not expose
```

До production необходимо формализовать и протестировать эту policy.

---

# 35. Backend MVP Freeze

## Status

```text
ACTIVE
```

Текущий backend считается:

```text
FUNCTIONALLY COMPLETE FOR WEB MVP
```

Поэтому backend не следует продолжать бесконечно оптимизировать перед frontend.

Backend меняется сейчас только если:

1. frontend integration требует небольшого API change;
2. acceptance testing обнаруживает fundamental issue;
3. обнаруживается safety problem;
4. появляется blocker для MVP.

Known production issues записываются в technical debt.

---

# 36. Текущий этап — Web Assistant Frontend

## Status

```text
NEXT / IN PROGRESS
```

Это текущий главный milestone.

Необходимо создать полноценный Web Assistant.

Целевой UX:

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

Пользователь должен иметь возможность:

- писать сообщения;
- получать text answers;
- видеть conversation history;
- запускать Voice Mode;
- разговаривать голосом;
- видеть voice transcripts;
- продолжать разговор текстом после voice;
- использовать единый conversation experience.

---

# 37. Frontend Technology

## Status

```text
SELECTED
IMPLEMENTATION NOT STARTED
```

В качестве frontend foundation выбрано:

```text
assistant-ui
```

Причины:

- готовый AI chat UX;
- React / TypeScript;
- message/thread components;
- streaming support;
- markdown;
- tool UI extensibility;
- возможность собственного backend;
- возможность дальнейшего развития продукта.

Планируемая директория:

```text
frontend/
└── angioplus-assistant/
```

Существующий:

```text
yandex/realtime/web/static/index.html
```

пока сохраняется как рабочий voice prototype.

---

# 38. Frontend Implementation Plan

Первый этап:

```text
создать assistant-ui отдельно
```

не изменяя рабочий backend.

Порядок:

```text
1. Create frontend branch

2. Install Node.js LTS

3. Create assistant-ui frontend

4. Run frontend locally

5. Evaluate UI

6. Connect text chat to backend

7. Integrate existing voice WebSocket

8. Display voice transcripts in chat

9. Unify text + voice conversation

10. Run acceptance tests
```

---

# 39. Frontend Git Branch

Backend checkpoint уже сохранён отдельно.

Текущая backend branch:

```text
feature/yandex-retrieval-controller
```

Ключевой commit:

```text
7bc6ec7
feat: integrate custom retrieval with Yandex Realtime
```

Для frontend используется отдельная branch:

```text
feature/web-assistant-ui
```

Это позволяет сохранить рабочий backend milestone независимо от frontend experiments.

---

# 40. Node.js

## Status

```text
NOT INSTALLED
```

На момент последней проверки:

```text
node
npm
```

не были доступны локально.

Перед началом frontend необходимо установить:

```text
Node.js LTS
```

и проверить:

```bash
node --version
npm --version
```

---

# 41. Telephony

## Status

```text
DEFERRED
```

Телефония больше НЕ является ближайшим MVP milestone.

Не начинать сейчас:

- SIP;
- Plusofon;
- +7 phone number;
- production media bridge.

Новая последовательность:

```text
Backend
    ↓
Web Assistant
    ↓
Acceptance Testing
    ↓
Backend Hardening
    ↓
Production Readiness
    ↓
Telephony
```

Причина:

Web Assistant позволяет намного быстрее:

- демонстрировать продукт;
- тестировать пользователей;
- отлаживать retrieval;
- тестировать voice;
- проверять UX;
- собирать feedback.

---

# 42. Acceptance Testing

## Status

```text
PARTIAL
EXPANSION REQUIRED AFTER FRONTEND
```

После появления Web UI необходимо провести системный acceptance test.

Минимальные категории:

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

Для factual turns желательно фиксировать:

```text
user text / transcript
tool query
retrieval latency
TOP-5 evidence
sources
final answer
grounding correctness
```

---

# 43. Long Conversation Testing

## Status

```text
PRODUCTION TODO
```

Целевой будущий conversation duration:

```text
20–40 minutes
```

Необходимо проверить:

- context growth;
- tool results;
- repeated retrieval;
- evidence accumulation;
- function call history;
- token usage;
- response latency.

`conversation.item.delete` может использоваться как один из механизмов context management.

Но конкретная стратегия должна основываться на runtime measurements.

---

# 44. Logging & Observability

## Status

```text
PARTIAL
PRODUCTION EXPANSION REQUIRED
```

Current backend уже логирует основные retrieval events.

Future structured trace должен содержать:

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
errors
```

Полноценная admin dashboard не требуется для текущего Web MVP.

---

# 45. Human Escalation

## Status

```text
FUTURE
```

При недостатке evidence будущая production system должна поддерживать:

```text
clarification
      ↓
refusal
      ↓
human escalation
```

На этапе Web MVP достаточно корректного clarification/refusal behavior.

---

# 46. Ticketing / CRM / Diagnostic Tools

## Status

```text
FUTURE
```

В будущем custom Function Calling позволит добавить:

```text
create_support_ticket(...)

check_license(...)

get_system_status(...)

run_diagnostic(...)

transfer_to_human(...)
```

Эти функции не должны добавляться до проверки основного Web Assistant.

---

# 47. Production Deployment

## Status

```text
FUTURE
```

Production deployment должен включать:

```text
services / containers
HTTPS / WSS
secret management
automatic restart
structured logging
monitoring
domain
health checks
error alerts
backup configuration
authentication
```

Это отдельный этап после Web MVP и acceptance testing.

---

# 48. Текущая карта готовности

```text
Research / architecture         DONE

Legacy/control evaluation       DONE

Yandex Realtime                 MVP READY

Knowledge Base V2               MVP READY

Vector Store retrieval          MVP READY

Retrieval Controller            MVP READY

Custom Function Calling         MVP READY

Realtime + Retrieval E2E        MVP READY

TurnManager                     MVP READY

Browser voice prototype         MVP READY

Web Assistant frontend          NEXT / IN PROGRESS

Text chat integration           TODO

Unified text + voice UX         TODO

Expanded acceptance testing     TODO

Fresh retrieval guarantee       PRODUCTION P1

Latency hardening               PRODUCTION P1

Long-session management         PRODUCTION TODO

Safety validation               PRODUCTION TODO

Logging / observability         PARTIAL

Human escalation                FUTURE

Ticketing / CRM                 FUTURE

Production deployment           FUTURE

Telephony / SIP                 DEFERRED
```

---

# 49. Что уже доказано

Проект больше не находится в стадии проверки фундаментальной feasibility.

Уже доказано:

```text
Yandex Realtime works
```

```text
speech-realtime-260528 works
```

```text
browser realtime voice works
```

```text
Yandex Vector Store works
```

```text
natural-language retrieval works
```

```text
Evidence Hit@5 = 19/19
```

```text
custom Function Calling works
```

```text
RetrievalController works
```

```text
Realtime → Retrieval → Realtime works
```

```text
grounded voice answer works
```

```text
follow-up context works
```

```text
TurnManager prevents known WAITING_FOR_TOOL_RESULT crash
```

Поэтому текущая стадия:

```text
PRODUCTIZATION
+
WEB MVP
```

а не:

```text
R&D
```

---

# 50. Что пока НЕ доказано

Перед production необходимо доказать:

```text
fresh retrieval on every required factual turn
```

```text
stable latency under sustained usage
```

```text
20–40 minute conversation reliability
```

```text
no speech loss during long tool calls
```

```text
robust technical STT
```

```text
medical safety under adversarial cases
```

```text
credential protection
```

```text
production monitoring / recovery
```

```text
telephony reliability
```

Это не блокирует Web MVP, но блокирует утверждение:

```text
PRODUCTION READY
```

---

# 51. Ближайший milestone

Текущая непосредственная задача:

```text
AngioPlus Web Assistant
```

Цель milestone:

```text
Text Chat
+
Voice Mode
+
Conversation History
+
Current AngioPlus Backend
```

Definition of Done:

```text
User opens web application
        ↓
asks AngioPlus question by text
        ↓
gets grounded text answer
        ↓
starts Voice Mode
        ↓
asks AngioPlus question by voice
        ↓
gets grounded voice answer
        ↓
conversation is visible in the same UI
```

---

# 52. После Web MVP

После завершения frontend:

```text
Web MVP
   ↓
Expanded AngioPlus Acceptance Suite
   ↓
Fix demonstrated blockers
   ↓
Fresh Retrieval P1
   ↓
Latency / Buffer P1
   ↓
Long-session testing
   ↓
Safety validation
   ↓
Production hardening
   ↓
Deployment
   ↓
Telephony / SIP
```

Порядок может меняться только если testing обнаружит более критичный blocker.

---

# 53. Чего сейчас НЕ делать

До завершения Web MVP не начинать без отдельной причины:

```text
new Vector DB
new embedding stack
new local STT
new local TTS
LangChain
LangGraph
multi-agent architecture
mandatory Query Rewriter
mandatory LLM reranker
SIP
Plusofon
CRM
ticketing
admin panel
billing
```

Главная опасность текущего этапа:

```text
продолжать улучшать backend,
который уже достаточен для MVP,
вместо создания самого продукта.
```

---

# 54. Текущий Git Checkpoint

Рабочий backend milestone:

```text
branch:
feature/yandex-retrieval-controller

commit:
7bc6ec7

message:
feat: integrate custom retrieval with Yandex Realtime
```

Этот checkpoint содержит:

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

Backend checkpoint сохранён в remote repository.

---

# 55. Главный принцип следующего этапа

Сейчас приоритет меняется с:

```text
"можем ли мы это построить?"
```

на:

```text
"может ли пользователь этим нормально пользоваться?"
```

Поэтому текущий главный focus:

```text
USER EXPERIENCE
+
WEB PRODUCTIZATION
```

при сохранении уже работающей backend architecture.

---

# 56. Основной принцип проекта

Приоритет:

```text
Correctness
    >
Grounding
    >
Safety
    >
Traceability
    >
User Experience
    >
Latency
    >
Architectural Simplicity
    >
Additional Features
```

Новые компоненты добавляются только тогда, когда testing или benchmark показывает реальную необходимость.

Главный принцип:

> **Система должна быть настолько простой, насколько это возможно, но настолько контролируемой, насколько это необходимо для надёжной технической поддержки медицинского программного обеспечения.**

Текущая стратегия:

```text
Make it work
    ↓
Make it measurable
    ↓
Make the MVP usable
    ↓
Test it broadly
    ↓
Harden what testing proves necessary
    ↓
Then move to production
```