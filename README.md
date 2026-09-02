# AngioPlus AI Support Assistant

AI-ассистент для технической поддержки медицинского программного обеспечения **AngioPlus Core**.

Цель проекта — создать надёжного технического ассистента, который позволяет пользователю общаться с системой текстом или голосом, отвечает на вопросы на основании утверждённой документации AngioPlus и при недостатке информации запрашивает уточнение, отказывается от неподтверждённого ответа или передаёт обращение человеку.

Текущий этап проекта — **Web MVP**.

---

## Возможности системы

Целевая система должна поддерживать:

- текстовый AI-чат;
- realtime voice conversation;
- естественный русский диалог;
- follow-up вопросы и контекст разговора;
- barge-in / interruption;
- распознавание технической терминологии AngioPlus;
- controlled retrieval по документации;
- source traceability;
- отказ от неподтверждённых ответов;
- medical safety;
- human escalation;
- будущую систему тикетов;
- будущие diagnostic/business tools;
- журналирование и аналитику;
- versioned knowledge base;
- в дальнейшем — обычную телефонию / SIP.

Ассистент предназначен для **технической поддержки AngioPlus Core**.

Он не должен самостоятельно принимать клинические решения или придумывать медицинские рекомендации.

---

## Текущий статус

Backend MVP функционально готов для демонстрации.

Рабочий основной цикл:

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
grounded response
```

Проверены:

- realtime voice conversation;
- Yandex Realtime `speech-realtime-260528`;
- custom Function Calling;
- прямой поиск через Yandex Vector Store Search API;
- собственный Retrieval Controller;
- передача TOP-5 evidence обратно в Realtime;
- grounded ответы по документации;
- follow-up context;
- защита Realtime state machine во время tool execution через `TurnManager`.

Backend считается:

```text
MVP DEMO READY
```

но пока не:

```text
PRODUCTION READY
```

---

## Текущая архитектура

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

Телефония сознательно отложена до завершения Web MVP.

---

## Yandex-first подход

Проект использует Yandex там, где сервис уже способен надёжно выполнять необходимую функцию:

- realtime speech recognition;
- server-side VAD;
- dialogue model;
- voice generation;
- Function Calling;
- Vector Store;
- semantic/hybrid retrieval.

Python остаётся тонким orchestration/control layer.

Мы не строим собственные STT, TTS, Vector DB, embedding infrastructure или большой agent framework без доказанной необходимости.

Основной принцип:

```text
Yandex делает то, что умеет делать надёжно.

Python контролирует то,
что требует прозрачности и бизнес-логики.
```

---

## Retrieval

Основной production candidate для retrieval:

```text
Yandex Realtime
        ↓
search_angioplus_knowledge
        ↓
Python RetrievalController
        ↓
Yandex Vector Store Search
        ↓
TOP-5 evidence
        ↓
Yandex Realtime
```

Встроенный Realtime File Search был протестирован, но больше не является основным retrieval mechanism.

Основные причины:

- скрытое формирование query;
- недостаточная наблюдаемость retrieved chunks;
- ограниченный контроль над retrieval;
- рост active Realtime context;
- сложности с source audit;
- воспроизводимые retrieval failures для некоторых вопросов;
- отсутствие гарантии fresh retrieval на каждом factual turn.

Direct Vector Store Search для тех же вопросов показал существенно лучший контроль и предсказуемость.

---

## Retrieval Controller

Основные компоненты:

```text
yandex/retrieval/
├── client.py
├── controller.py
├── ingest.py
└── models.py
```

Controller получает query от Realtime и возвращает структурированный evidence.

Текущий runtime:

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

Query Rewriter и отдельный LLM reranker тестировались, но сейчас не используются в runtime.

Они остаются возможными Plan B, если будущие regression tests покажут необходимость.

---

## Retrieval benchmark

Текущий baseline:

```text
Evidence Hit@1: 13/19
Evidence Hit@3: 17/19
Evidence Hit@5: 19/19

Preferred Hit@1: 13/19
Preferred Hit@3: 17/19
Preferred Hit@5: 18/19

Median latency: ~1540 ms
P95 latency:   ~2300 ms
```

Главный результат:

```text
Evidence Hit@5 = 19/19
```

Поэтому runtime передаёт полный TOP-5 evidence.

Эксперименты с TOP-3 и lossy lexical compaction ухудшили recall и были отклонены.

---

## База знаний

Текущий Yandex Vector Store:

```text
AI Studio:
angioplus-kb-v2

Vector Store ID:
fvtpohd7u6dnbu8f44ah
```

Состояние:

```text
5 documents
116 chunks
completed
```

Текущий corpus:

```text
2025July.pdf
AngioPlus Core v2.5 IFU_RU_Final.pdf
Common Info to Distributor - commercial (1).pdf
Competitor_2025.pdf
Q&A List ENG.xlsx
```

`Development_Questions_L1_L2.docx` сознательно исключён из V2, поскольку преимущественно содержит вопросы для разработки и создавал retrieval noise.

Scientific papers также пока не входят в основной production retrieval corpus.

---

## Приоритет источников

Общий порядок авторитетности:

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

Source authority не заменяет semantic relevance.

Нерелевантный IFU chunk не должен автоматически становиться правильным evidence только из-за высокого authority.

---

## Grounding и безопасность

Ассистент должен:

- отвечать на продуктовые вопросы на основании документации;
- не придумывать отсутствующие факты;
- запрашивать уточнение при неоднозначном вопросе;
- не раскрывать credentials;
- не принимать самостоятельные клинические решения;
- не рекомендовать лечение;
- не принимать решение о PCI или stent;
- отказываться от неподтверждённого ответа;
- позволять будущую эскалацию человеку.

Принцип:

```text
retrieved
≠
automatically allowed to answer
```

Safety policy имеет приоритет над retrieval ranking.

---

## Voice prototype

Рабочий browser voice backend:

```text
yandex/realtime/web/server.py
```

Текущий простой frontend:

```text
yandex/realtime/web/static/index.html
```

Flow:

```text
Browser microphone
        ↓
PCM16 44100 Hz
        ↓
Python WebSocket
        ↓
Yandex Realtime 260528
        ↓
custom retrieval
        ↓
voice response
        ↓
Browser
```

Отдельный microphone smoke test:

```text
yandex/realtime/mic_client.py
```

---

## TurnManager

Во время интеграции custom Function Calling была обнаружена Realtime state-machine проблема:

```text
illegal command FinalizeUserUtterance
in state WAITING_FOR_TOOL_RESULT
```

Для защиты реализован `TurnManager`.

Основные states:

```text
READY
TOOL_PENDING
WAITING_FOR_RESPONSE
```

Пока выполняется retrieval, входящее browser audio временно буферизуется локально и отправляется после перехода Realtime обратно в безопасное состояние.

Это решение уже проверено в live voice tests.

---

## Известные ограничения

Backend достаточен для Web MVP, но перед production необходимо закрыть несколько вопросов.

### Fresh retrieval

Realtime иногда отвечает на повторный factual question из conversation context без нового вызова retrieval.

Это является production P1.

### Retrieval latency

Обычный retrieval занимает примерно:

```text
1.5–3.5 s
```

но наблюдался отдельный spike примерно:

```text
11.7 s
```

Перед production необходимо собрать p50/p95/p99.

### Audio buffering

Текущий `TurnManager` хранит примерно 5 секунд входящего аудио во время tool execution.

При очень долгом retrieval старое audio может начать отбрасываться.

### Voice selection

Запрашивается:

```text
dasha
```

но Yandex в runtime сообщает:

```text
kirill
```

### Greeting

Startup greeting требует дополнительной UX-настройки.

### Long sessions

Необходимо отдельно протестировать реальные разговоры продолжительностью примерно 20–40 минут.

Эти проблемы не блокируют текущий Web MVP.

---

## Текущий frontend milestone

Следующий этап разработки:

```text
WEB ASSISTANT FRONTEND
```

Цель — создать нормальный демонстрационный интерфейс, где пользователь сможет:

1. написать вопрос;
2. получить текстовый grounded ответ;
3. видеть историю разговора;
4. запустить Voice Mode;
5. разговаривать с тем же ассистентом голосом;
6. видеть voice transcripts в общей истории разговора.

В качестве frontend foundation выбрано:

```text
assistant-ui
```

Новый frontend планируется создавать отдельно:

```text
frontend/
└── angioplus-assistant/
```

Существующий рабочий voice prototype сначала сохраняется без изменений.

---

## Структура репозитория

```text
yandex/
    Текущая Yandex-first архитектура.

app/
    Предыдущая RAG-реализация:
    BGE-M3 + Qdrant + DeepSeek.
    CONTROL / reference.

bot/
    Telegram-интерфейс старого RAG.
    CONTROL / reference.

voice/
    Старый LiveKit voice stack.
    LEGACY.

frontend/
    Новый Web Assistant frontend.
    Создаётся на текущем этапе.

benchmarks/
    Retrieval benchmarks и архитектурные эксперименты.

scripts/
    Служебные и экспериментальные scripts.

legacy/
    Исторические компоненты.

data/
    Документы и локальные данные проекта.

tests/
    Автоматические тесты.

artifacts/
    Generated diagnostics/debug outputs.
    Не отслеживаются Git.
```

Наличие старого компонента в repository не означает, что он является частью текущей production architecture.

---

## CONTROL RAG

Предыдущая система:

```text
Telegram
   ↓
FastAPI /ask
   ↓
query normalization
   ↓
BAAI/bge-m3
   ↓
Qdrant
   ↓
DeepSeek
   ↓
Telegram
```

Она сохраняется как:

- regression reference;
- контрольная реализация;
- источник проверенной domain logic;
- резервный исторический MVP.

Новые возможности без необходимости туда не добавлять.

---

## LEGACY Voice Stack

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

Она больше не является целевой архитектурой.

Код сохраняется только для:

- истории;
- benchmarks;
- возможного повторного использования отдельных решений.

---

## Git checkpoint

Текущая backend branch:

```text
feature/yandex-retrieval-controller
```

Ключевой backend MVP commit:

```text
7bc6ec7
feat: integrate custom retrieval with Yandex Realtime
```

Этот checkpoint содержит рабочую интеграцию:

```text
Yandex Realtime 260528
+
custom Function Calling
+
RetrievalController
+
Vector Store V2
+
TurnManager
```

Backend milestone сохранён в remote Git repository.

Для frontend рекомендуется отдельная branch:

```text
feature/web-assistant-ui
```

---

## Git workflow

Обычный процесс:

```text
local development
        ↓
tests / smoke tests
        ↓
git status
        ↓
git diff
        ↓
commit
        ↓
push
```

Production server не должен быть основной development environment.

Секреты хранятся в `.env` и никогда не должны попадать в Git.

---

## Текущая точка разработки

Backend MVP:

```text
COMPLETE FOR DEMONSTRATION
```

Текущая задача:

```text
CREATE WEB ASSISTANT FRONTEND
```

Порядок дальнейшей работы:

```text
Backend MVP
    ↓
Web Assistant UI
    ↓
Text integration
    ↓
Voice integration
    ↓
Unified conversation experience
    ↓
AngioPlus acceptance testing
    ↓
Backend hardening
    ↓
Production readiness
    ↓
Telephony / SIP
```

Сейчас не нужно возвращаться к:

- native Realtime File Search;
- standalone Query Rewriter experiments;
- standalone reranker experiments;
- old LiveKit architecture;
- новой local STT/TTS;
- новому Vector DB;
- SIP / Plusofon;
- сложной agent architecture.

если frontend или acceptance testing не покажут реальную необходимость.

---

## Подробный технический контекст

Подробная история архитектурных решений, benchmarks, известных проблем, экспериментов и текущих технических ограничений находится в:

```text
PROJECT_CONTEXT.md
```

Перед существенными архитектурными изменениями необходимо сначала прочитать этот файл.

---

## Главный принцип проекта

> **Система должна быть настолько простой, насколько это возможно, но настолько контролируемой, насколько это необходимо для надёжной технической поддержки медицинского программного обеспечения.**

Текущий подход:

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

Если возникает выбор между красивой архитектурой и измеримо более надёжным решением, приоритет имеет надёжность.