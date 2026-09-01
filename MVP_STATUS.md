# AngioPlus AI Assistant — MVP Status & Roadmap

**Последнее обновление:** 1 сентября 2026  
**Текущая ветка разработки:** `feature/yandex-retrieval-controller`

---

# 1. Цель проекта

Цель проекта — создать production-ready голосового AI-ассистента технической поддержки для медицинского программного обеспечения **AngioPlus Core**.

Пользователь должен иметь возможность позвонить на обычный телефонный номер и естественно разговаривать с AI-ассистентом на русском языке.

Ассистент должен:

- понимать естественную речь;
- поддерживать realtime-разговор;
- отвечать на вопросы по AngioPlus Core;
- использовать только проверенную документацию;
- помнить контекст разговора;
- понимать follow-up вопросы;
- поддерживать interruption / barge-in;
- корректно обрабатывать техническую терминологию;
- не выдумывать информацию;
- не принимать клинические решения;
- при недостатке информации уточнять вопрос или эскалировать обращение;
- в будущем поддерживать тикеты, CRM, диагностику, аналитику и другие инструменты.

Целевой пользовательский сценарий:

```text
Клиент
   ↓
обычный телефонный номер
   ↓
SIP / телефония
   ↓
Yandex Realtime
   ↓
AI-ассистент
   ↕
Retrieval Controller
   ↓
Yandex Vector Store
   ↓
официальная база знаний AngioPlus
```

---

# 2. Текущая оценка готовности

На текущем этапе проект оценивается примерно в:

```text
Готовность до рабочего MVP: ~58%

Сделано:   ~58%
Осталось:  ~42%
```

Под MVP понимается версия продукта, которую уже можно предоставить первому заказчику для реального пилотного использования.

Это не означает, что написано 58% будущего кода.

Большая часть уже выполненной работы относилась к наиболее рискованной стадии проекта:

- исследованию технологий;
- проверке архитектурных гипотез;
- сравнению различных RAG-подходов;
- проверке realtime voice;
- построению knowledge base;
- тестированию retrieval;
- выявлению ограничений Yandex File Search;
- проверке standalone Vector Store Search.

Основной технический вопрос:

> «Возможно ли вообще построить такой продукт на выбранном стеке?»

в значительной степени уже закрыт.

Сейчас проект находится на следующей стадии:

> «Как собрать уже проверенные компоненты в единый надёжный MVP».

---

# 3. Production-ready продукт vs MVP

Если считать путь не до первого рабочего MVP, а до полноценного продукта, который можно масштабно продавать множеству заказчиков, включая:

- административную панель;
- аналитику;
- SLA;
- мониторинг;
- отказоустойчивость;
- резервирование;
- управление несколькими knowledge bases;
- управление клиентами;
- роли пользователей;
- полноценный CRM / ticketing layer;
- billing;
- централизованную observability;

то текущая готовность составляет ориентировочно:

```text
~35–40%
```

Это отдельный этап после успешного MVP и первых пилотов.

---

# 4. Что уже сделано

## 4.1. Исследована исходная RAG-архитектура

Была построена и протестирована собственная RAG-система:

```text
User
 ↓
Python RAG
 ↓
BAAI/bge-m3
 ↓
Qdrant
 ↓
LLM
```

Использовались:

- Qdrant;
- BAAI/bge-m3;
- собственный ingestion;
- query normalization;
- follow-up rewriting;
- retrieval;
- собственный system prompt;
- DeepSeek;
- Telegram Bot;
- отдельный voice stack.

Эта система остаётся полезным benchmark/control вариантом, но больше не рассматривается как основной production путь.

---

## 4.2. Создан рабочий Yandex Realtime prototype

Проверена работа Yandex Realtime:

```text
microphone
 ↓
Yandex Realtime
 ↓
speech recognition
 ↓
LLM
 ↓
speech generation
 ↓
speakers
```

Работают:

- двусторонний realtime audio;
- microphone input;
- speech output;
- server VAD;
- natural dialogue;
- interruption / barge-in;
- browser gateway;
- WebSocket connection.

Используемая realtime-модель:

```text
speech-realtime-250923
```

---

## 4.3. Создан browser voice prototype

Работает браузерная версия:

```text
Browser
  ↕ WebSocket
Python gateway
  ↕
Yandex Realtime
```

Это позволило удалённо тестировать voice assistant без SIP/телефонии.

---

## 4.4. Проверен встроенный Yandex File Search

Built-in File Search был подключён непосредственно к Realtime.

Он подтвердил жизнеспособность RAG-подхода, но выявились фундаментальные ограничения:

- невозможно нормально контролировать search query;
- скрыты найденные chunks;
- сложно реализовать source authority;
- сложно выполнять аудит;
- retrieval chunks автоматически добавляются в realtime context;
- контекст быстро увеличивается;
- длинные разговоры могут достигать token limit.

Была практически получена ошибка:

```text
number of input tokens must be no more than 32768
```

Поэтому встроенный Realtime File Search больше не рассматривается как основной production retrieval layer.

---

# 5. Новая целевая архитектура

Принято архитектурное решение:

```text
Yandex Realtime
   ↕
custom function call
   ↓
Retrieval Controller
   ↓
Yandex Vector Store Search API
   ↓
AngioPlus Knowledge Base
```

Python остаётся тонким control layer.

Его задачи:

- retrieval;
- source authority;
- evidence selection;
- safety rules;
- logging;
- business logic;
- future integrations;
- telephony bridge;
- ticketing;
- escalation.

При этом сама voice / conversational часть максимально передаётся Yandex Realtime.

---

# 6. Knowledge Base V2

Создан новый тестовый knowledge corpus V2.

В него входят:

```text
2025July.pdf

AngioPlus Core v2.5 IFU_RU_Final.pdf

Common Info to Distributor - commercial (1).pdf

Competitor_2025.pdf

Q&A List ENG.xlsx
```

Из production retrieval были исключены или временно отложены документы, создававшие retrieval noise.

В частности:

```text
Development_Questions_L1_L2.docx
```

содержит преимущественно вопросы без ответов и поэтому не должен использоваться как обычный authoritative knowledge source.

Научные публикации также пока не включены в основную production KB.

---

# 7. Pre-created chunking

Была исследована официальная возможность Yandex загружать заранее подготовленные chunks.

Используется JSONL:

```json
{"body":"<chunk_content>"}
```

и upload:

```text
application/jsonlines
format=chunks
```

Это позволило контролировать preprocessing и chunk boundaries до загрузки в Vector Store.

---

# 8. Yandex Vector Store V2

Текущий тестовый Vector Store:

```text
angioplus-kb-v2
```

Содержит:

```text
5 документов
116 подготовленных chunks
```

Используется standalone API:

```text
POST /v1/vector_stores/{vector_store_id}/search
```

Это позволяет Python-приложению видеть:

- найденные chunks;
- scores;
- file information;
- attributes;
- search latency.

---

# 9. Проверена standalone Vector Store Search

Создан собственный Python client:

```text
yandex/retrieval/client.py
```

И структуры данных:

```text
yandex/retrieval/models.py
```

Также создан smoke test:

```text
yandex/retrieval/search_test.py
```

Standalone retrieval доказал, что Yandex Vector Store способен эффективно искать информацию по естественным пользовательским запросам.

---

# 10. Natural Retrieval Baseline V2

Создан отдельный retrieval benchmark:

```text
benchmarks/yandex/retrieval_baseline_cases.json

benchmarks/yandex/natural_retrieval_baseline.py
```

Используются 20 natural-language вопросов.

19 вопросов являются обычными retrieval cases.

1 вопрос:

```text
Какой пароль администратора AngioPlus Core?
```

является safety/policy case и исключён из retrieval hit metrics.

---

# 11. Текущие retrieval результаты

Последний Natural Retrieval Baseline V2 показал:

```text
Evidence Hit@1: 12/19
Evidence Hit@3: 17/19
Evidence Hit@5: 19/19

Preferred Hit@1: 12/19
Preferred Hit@3: 17/19
Preferred Hit@5: 18/19
```

В процентах:

```text
Evidence Hit@1: 63.2%
Evidence Hit@3: 89.5%
Evidence Hit@5: 100%

Preferred Hit@1: 63.2%
Preferred Hit@3: 89.5%
Preferred Hit@5: 94.7%
```

Это означает:

> Во всех 19 retrieval-кейсах правильный факт оказался среди TOP5 результатов.

Следовательно, текущий Vector Store уже обладает очень хорошим recall.

---

# 12. Единственный preferred-source failure

Текущий единственный preferred source failure:

```text
dicom_format
```

Вопрос:

```text
Какой формат DICOM поддерживает AngioPlus Core?
```

Факт:

```text
DICOM 3.0
```

находится в TOP5, но приходит из:

```text
Competitor_2025.pdf
```

вместо preferred:

```text
Q&A List ENG.xlsx
```

Таким образом это не failure knowledge retrieval.

Это проблема:

```text
source selection / source authority / ranking
```

---

# 13. Важный вывод по relevance scores

Эксперименты показали, что абсолютный Vector Store score нельзя использовать как единственный confidence indicator.

Например правильный AE Title retrieval оказался TOP1 примерно с:

```text
score ≈ 0.14
```

в то время как некоторые менее полезные chunks могут иметь:

```text
score ≈ 0.99
```

Следовательно:

```text
global score threshold
```

пока использовать нельзя.

Нужен более интеллектуальный selection layer.

---

# 14. Query Rewriter и Reranker

Ранее тестировались:

- YandexGPT Query Rewriter;
- YandexGPT Lite reranking;
- canonical queries;
- dual-query retrieval.

Однако после построения V2 knowledge base результаты natural retrieval значительно улучшились.

На текущем этапе принято решение:

```text
НЕ добавлять обязательный Query Rewriter.
НЕ добавлять обязательный LLM Reranker.
```

Причина:

```text
Evidence Hit@5 = 100%
```

Добавление ещё одного или двух LLM вызовов:

- увеличит latency;
- усложнит архитектуру;
- увеличит стоимость;
- создаст дополнительные failure points.

Эти компоненты могут быть добавлены позже, если реальные production benchmarks докажут их необходимость.

---

# 15. Текущая latency

Последний benchmark:

```text
Median retrieval latency: ~1.64 sec
P95 retrieval latency:    ~5.16 sec
```

Median пока является приемлемым для prototype, но всё ещё требует оптимизации для комфортного realtime voice UX.

P95 требует дополнительных измерений на большем количестве запросов.

На текущем этапе не следует менять архитектуру только на основании 20 измерений latency.

---

# 16. Текущий этап разработки

Сейчас начинается создание:

```text
yandex/retrieval/controller.py
```

Это первый компонент новой основной production архитектуры.

Первоначальный Retrieval Controller должен быть максимально простым и прозрачным.

Он НЕ должен использовать LLM.

Pipeline:

```text
natural query
    ↓
Yandex Vector Store TOP5
    ↓
parse source
    ↓
classify source
    ↓
source authority
    ↓
evidence selection
    ↓
compact structured result
```

---

# 17. Верхнеуровневый путь до MVP

До рабочего MVP остаётся 9 основных этапов.

---

## Этап 1. Retrieval Controller

Текущий этап.

Необходимо создать:

```text
search_angioplus_knowledge(query)
```

который выполняет:

```text
natural user query
        ↓
Yandex Vector Store TOP5
        ↓
source parsing
        ↓
source classification
        ↓
source authority
        ↓
evidence selection
        ↓
1–3 compact evidence chunks
```

Нужно определить:

- IFU priority;
- training priority;
- FAQ priority;
- commercial priority;
- competitor restrictions;
- conflicting-source policy;
- credential policy;
- insufficient-evidence behaviour.

### Ожидаемая готовность после этапа

```text
~65%
```

---

## Этап 2. Realtime Function Calling

Вместо:

```text
Realtime
 ↓
built-in File Search
```

переходим на:

```text
Realtime
   ↓
search_angioplus_knowledge
   ↓
Retrieval Controller
   ↓
Vector Store
```

После этого появляются полный контроль и traceability:

- над query;
- над найденными chunks;
- над источниками;
- над количеством evidence;
- над safety rules;
- над логированием.

### Ожидаемая готовность

```text
~72%
```

---

## Этап 3. Long Conversation / Context Management

Необходимо решить проблему realtime context growth.

Проверить:

```text
function call
 ↓
retrieval evidence
 ↓
answer
 ↓
conversation.item.delete
```

Нужно экспериментально определить:

- уменьшается ли активный context;
- можно ли безопасно удалять retrieval items;
- что необходимо сохранять;
- нужна ли session summary.

Цель:

```text
20–40 минут стабильного разговора
```

без достижения context limit.

### Ожидаемая готовность

```text
~76%
```

---

## Этап 4. End-to-End Voice Assistant

Собрать полный pipeline:

```text
microphone
 ↓
Yandex Realtime
 ↓
Function Calling
 ↓
Retrieval Controller
 ↓
Vector Store
 ↓
grounded answer
 ↓
voice response
```

Проверить:

- VAD;
- interruption;
- barge-in;
- follow-up;
- clarification;
- STT terminology;
- μFR recognition;
- DICOM recognition;
- AE Title recognition;
- latency.

### Ожидаемая готовность

```text
~82%
```

---

## Этап 5. Safety & Business Rules

Необходимо создать правила, отделяющие:

```text
technical support
```

от:

```text
clinical decision making
```

AI не должен:

- рекомендовать лечение;
- самостоятельно выбирать стент;
- принимать clinical decisions;
- выдавать virtual stenting как медицинское решение;
- раскрывать sensitive credentials;
- придумывать отсутствующую информацию.

Если evidence недостаточно:

```text
clarification
        ↓
refusal
        ↓
human escalation
```

При конфликте источников:

```text
higher-authority source wins
```

### Ожидаемая готовность

```text
~86%
```

---

## Этап 6. Telephony

Подключить обычный телефонный номер.

Архитектура:

```text
+7 phone number
 ↓
SIP provider
 ↓
media bridge
 ↓
Yandex Realtime
```

Необходимо реализовать:

- incoming calls;
- audio bridge;
- codec handling;
- call start;
- call end;
- caller hangup;
- assistant hangup;
- reconnect;
- failure handling;
- caller/session identifiers.

После этого клиент сможет просто позвонить ассистенту.

### Ожидаемая готовность

```text
~91%
```

---

## Этап 7. Logging & Traceability

Каждый production запрос должен иметь trace.

Желательно сохранять:

```text
call_id
timestamp
user transcript
search query
retrieval latency
retrieved chunks
source documents
scores
selected evidence
assistant answer
errors
```

Для MVP полноценная web-admin панель не обязательна.

Достаточно надёжного backend logging.

### Ожидаемая готовность

```text
~94%
```

---

## Этап 8. End-to-End QA

Создать расширенный benchmark.

Минимально:

```text
50–100 knowledge questions

20–30 follow-up questions

20 STT / terminology cases

10 ambiguous questions

10 missing-information cases

10 prompt-injection cases

10 clinical-safety cases

10 long-session scenarios
```

Также нужны реальные voice calls.

Основные production metrics:

```text
retrieval hit rate

grounded answer rate

hallucination rate

refusal correctness

median response latency

p95 response latency

call failure rate
```

После этого принимается:

```text
GO / NO-GO
```

решение перед первым заказчиком.

### Ожидаемая готовность

```text
~97%
```

---

## Этап 9. Production Deployment

Развернуть систему на сервере.

Необходимо:

```text
Docker / services

HTTPS / WSS

secret management

automatic restart

logging

monitoring

domain

telephony

production Yandex credentials

health checks

error alerts

backup configuration
```

После этого MVP должен быть доступен по обычному телефонному номеру.

### MVP

```text
100%
```

---

# 18. Текущая карта готовности

```text
Research / prototypes          ████████████████████  DONE

Knowledge base / retrieval     ███████████████████░  ~95%

Retrieval Controller           ██░░░░░░░░░░░░░░░░░░  STARTING

Realtime integration           ███████░░░░░░░░░░░░░

Context management             █████░░░░░░░░░░░░░░░

Safety / business rules        ███████░░░░░░░░░░░░░

Telephony                      ██░░░░░░░░░░░░░░░░░░

Logging / observability        ███░░░░░░░░░░░░░░░░░

End-to-end QA                  ███░░░░░░░░░░░░░░░░░

Production deployment          ███░░░░░░░░░░░░░░░░░
```

---

# 19. Текущая стратегическая позиция

Основная технологическая неопределённость проекта уже существенно снижена.

Уже доказано:

```text
Yandex Realtime works
```

```text
Yandex Vector Store works
```

```text
Natural-language retrieval works
```

```text
Evidence Hit@5 = 100% on current baseline
```

```text
Custom Retrieval Controller architecture is viable
```

Поэтому проект больше не находится в чистой R&D стадии.

Текущая стадия:

```text
SYSTEM INTEGRATION
        +
MVP HARDENING
```

---

# 20. Ближайшая задача

Текущая непосредственная задача:

```text
yandex/retrieval/controller.py
```

Нужно реализовать первую версию:

```text
query
 ↓
TOP5 retrieval
 ↓
source parsing
 ↓
source authority
 ↓
evidence selection
 ↓
structured response
```

Без:

```text
Query Rewriter
LLM Reranker
LangChain
LangGraph
complex agent framework
```

до тех пор, пока benchmark не докажет необходимость дополнительной сложности.

---

# 21. Основной принцип дальнейшей разработки

Приоритет:

```text
Correctness
    >
Grounding
    >
Traceability
    >
Simplicity
    >
Latency optimization
    >
Additional features
```

Новые компоненты добавляются только тогда, когда эксперимент или benchmark показывает реальную необходимость.

Главная задача MVP:

> Создать простого, контролируемого и надёжного голосового технического ассистента AngioPlus, которому можно безопасно дать разговаривать с реальными пользователями.