import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YANDEX_API_KEY")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

VECTOR_STORE_ID = "fvtpohd7u6dnbu8f44ah"

SEARCH_URL = (
    f"https://ai.api.cloud.yandex.net/v1/"
    f"vector_stores/{VECTOR_STORE_ID}/search"
)

RESPONSES_URL = "https://ai.api.cloud.yandex.net/v1/responses"

REWRITER_MODEL = f"gpt://{FOLDER_ID}/yandexgpt-5-lite"

PREVIEW_LENGTH = 700

REWRITER_INSTRUCTIONS = """
Ты преобразуешь вопрос пользователя в поисковый запрос
для технической базы знаний AngioPlus Core.

Ты НЕ отвечаешь на вопрос.
Ты только формируешь хороший запрос для vector search.

Правила:

1. Строго сохраняй исходный смысл вопроса.
2. Добавляй технические термины, синонимы и английские названия,
   если это помогает поиску.
3. Используй AngioPlus Core, μFR, μFR-2D, DICOM, PACS и другие
   термины только если они относятся к вопросу.
4. Не придумывай факты, значения, требования, пароли, номера,
   версии или ответы.
5. Не вставляй предполагаемый ответ в поисковый запрос.
6. Не делай медицинских или клинических выводов.
7. Не преобразовывай вопрос в диагноз, рекомендацию,
   противопоказание или решение о лечении.
8. Если пользователь спрашивает о клиническом применении,
   сохраняй нейтральную формулировку и ищи intended use,
   limitations, warnings, clinical interpretation или документацию
   по указанному параметру.
9. Можно использовать русский и английский одновременно.
10. Запрос должен быть коротким — обычно одна строка ключевых терминов.
11. Верни ТОЛЬКО поисковый запрос без пояснений,
    кавычек и markdown.

Примеры:

Вопрос:
Можно ли двигать стол во время съёмки?

Поисковый запрос:
AngioPlus μFR angiographic acquisition table movement move table table height during angiography

Вопрос:
Сколько кадров в секунду нужно для анализа?

Поисковый запрос:
AngioPlus μFR μFR-2D angiographic imaging frame rate frames per second fps acquisition requirements

Вопрос:
Какой длины должен быть участок сосуда для анализа?

Поисковый запрос:
AngioPlus μFR μFR-2D vessel segment ROI length minimum ROI length mm analysis

Вопрос:
Какой AE Title используется по умолчанию?

Поисковый запрос:
AngioPlus Core DICOM AE Title default AE Title configuration

Вопрос:
Можно ли поставить стент пациенту если μFR меньше 0.80?

Поисковый запрос:
AngioPlus Core μFR 0.80 clinical interpretation intended use limitations clinical decision
""".strip()


RERANKER_INSTRUCTIONS = """
Ты — reranker технической базы знаний AngioPlus Core.

Тебе будут переданы:
- исходный вопрос пользователя;
- поисковый запрос;
- до 5 кандидатов, найденных Vector Store.

Ты НЕ отвечаешь на вопрос.

Твоя задача — выбрать и упорядочить кандидаты,
которые лучше всего содержат доказательства для ответа.

Правила:

1. Оценивай прежде всего прямую релевантность содержимого вопросу.
2. Не выбирай chunk только потому, что там встречаются похожие слова.
3. Предпочитай chunk, который содержит конкретный ответ,
   значение, правило, ограничение, процедуру или описание,
   непосредственно относящееся к вопросу.
4. Если два кандидата одинаково релевантны, используй приоритет:
   IFU > FAQ > TRAINING > COMMERCIAL.
5. Но не выбирай IFU только из-за приоритета,
   если другой источник содержит прямой ответ, а IFU — нет.
6. Для safety, intended use, limitations и clinical-use вопросов
   предпочитай официальные ограничения IFU.
7. Не делай медицинских выводов.
8. Не используй знания вне переданных кандидатов.
9. Не придумывай факты.
10. Верни ТОЛЬКО номера кандидатов через запятую
    в порядке полезности.
11. Максимум 3 номера.

Пример ответа:

2,1,4
""".strip()


TESTS = [
    {
        "question": "Какая сейчас версия AngioPlus Core?",
        "expected": ["version"],
    },
    {
        "question": "Какие системные требования у AngioPlus Core?",
        "expected": ["system"],
    },
    {
        "question": "Какой AE Title используется по умолчанию?",
        "expected": ["ae", "title"],
    },
    {
        "question": "Какой порт используется для DICOM?",
        "expected": ["dicom", "port"],
    },
    {
        "question": "Поддерживает ли AngioPlus Core PACS Query/Retrieve?",
        "expected": ["query", "retrieve"],
    },
    {
        "question": "Какие требования к ангиографическому изображению?",
        "expected": ["angiograph"],
    },
    {
        "question": "Сколько кадров в секунду нужно для анализа?",
        "expected": ["15", "frame"],
    },
    {
        "question": "Можно ли использовать 7.5 кадров в секунду?",
        "expected": ["7.5"],
    },
    {
        "question": "Нужно ли вводить нитроглицерин перед ангиографией?",
        "expected": ["nitroglycer"],
    },
    {
        "question": "Можно ли двигать стол во время съёмки?",
        "expected": ["table"],
    },
    {
        "question": "Можно ли менять высоту стола во время съёмки?",
        "expected": ["table", "height"],
    },
    {
        "question": "Какой длины должен быть участок сосуда для анализа?",
        "expected": ["20", "mm"],
    },
    {
        "question": "Где должны находиться начало и конец исследуемого сегмента?",
        "expected": ["segment"],
    },
    {
        "question": "Как выбирается key frame?",
        "expected": ["key", "frame"],
    },
    {
        "question": "Можно ли вручную исправить контур сосуда?",
        "expected": ["contour"],
    },
    {
        "question": "Для чего используется fixed flow?",
        "expected": ["fixed", "flow"],
    },
    {
        "question": "Какая минимальная разница углов нужна для 3D анализа?",
        "expected": ["angle"],
    },
    {
        "question": "Можно ли использовать AngioPlus для принятия решения какой стент ставить?",
        "expected": ["stent"],
    },
    {
        "question": "Какой пароль администратора установлен по умолчанию?",
        "expected": ["password"],
    },
    {
        "question": "Можно ли поставить стент пациенту если μFR меньше 0.80?",
        "expected": ["0.80"],
    },
]


def rewrite_query(question: str) -> str:
    response = requests.post(
        RESPONSES_URL,
        headers={
            "Authorization": f"Api-Key {API_KEY}",
            "x-project": FOLDER_ID,
            "Content-Type": "application/json",
        },
        json={
            "model": REWRITER_MODEL,
            "instructions": REWRITER_INSTRUCTIONS,
            "input": question,
            "temperature": 0.1,
            "max_output_tokens": 150,
        },
        timeout=60,
    )

    if not response.ok:
        return f"[REWRITER ERROR {response.status_code}] {question}"

    data = response.json()

    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "").strip()
                if text:
                    return text

    return question


def get_text(item):
    parts = []

    for content in item.get("content", []):
        if content.get("type") == "text":
            text = content.get("text", "")
            if text:
                parts.append(text)

    return "\n".join(parts)


def extract_source_name(item):
    text = get_text(item)

    markers = [
        '"body": "Документ: ',
        '"body":"Документ: ',
        "Документ: ",
    ]

    for marker in markers:
        if marker in text:
            source = text.split(marker, 1)[1]
            source = source.split("\\n", 1)[0]
            source = source.split("\n", 1)[0]
            source = source.replace('"', "").strip()
            return source

    return item.get("filename", "UNKNOWN")


def clean_preview(text):
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = text.replace('\\"', '"')

    # Схлопываем пробелы, чтобы preview был компактным.
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > PREVIEW_LENGTH:
        return text[:PREVIEW_LENGTH] + "..."

    return text


def contains_expected(text, expected):
    normalized = text.lower()

    found = []
    missing = []

    for term in expected:
        if term.lower() in normalized:
            found.append(term)
        else:
            missing.append(term)

    return found, missing


def search(query: str):
    response = requests.post(
        SEARCH_URL,
        headers={
            "Authorization": f"Api-Key {API_KEY}",
            "x-project": FOLDER_ID,
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "rewrite_query": False,
            "max_num_results": 5,
        },
        timeout=60,
    )

    if not response.ok:
        return {
            "error": f"HTTP {response.status_code}: {response.text}",
            "results": [],
        }

    data = response.json()
    results = []

    for item in data.get("data", []):
        text = get_text(item)

        results.append(
            {
                "score": item.get("score"),
                "source": extract_source_name(item),
                "text": text,
                "preview": clean_preview(text),
            }
        )

    return {
        "error": None,
        "results": results,
    }

def rerank(question: str, rewritten_query: str, results: list[dict]) -> list[int]:
    if not results:
        return []

    candidate_blocks = []

    for idx, result in enumerate(results, start=1):
        candidate_blocks.append(
            "\n".join(
                [
                    f"CANDIDATE {idx}",
                    f"SOURCE: {result.get('source', 'UNKNOWN')}",
                    f"VECTOR SCORE: {result.get('score')}",
                    "TEXT:",
                    result.get("text", ""),
                ]
            )
        )

    reranker_input = (
        f"USER QUESTION:\n{question}\n\n"
        f"SEARCH QUERY:\n{rewritten_query}\n\n"
        f"CANDIDATES:\n\n"
        + "\n\n---\n\n".join(candidate_blocks)
    )

    response = requests.post(
        RESPONSES_URL,
        headers={
            "Authorization": f"Api-Key {API_KEY}",
            "x-project": FOLDER_ID,
            "Content-Type": "application/json",
        },
        json={
            "model": REWRITER_MODEL,
            "instructions": RERANKER_INSTRUCTIONS,
            "input": reranker_input,
            "temperature": 0.0,
            "max_output_tokens": 40,
        },
        timeout=60,
    )

    if not response.ok:
        print(
            f"RERANKER ERROR HTTP {response.status_code}: "
            f"{response.text}"
        )
        return []

    data = response.json()

    output_text = ""

    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                output_text += content.get("text", "")

    numbers = [
        int(number)
        for number in re.findall(r"\d+", output_text)
    ]

    selected = []
    seen = set()

    for number in numbers:
        if 1 <= number <= len(results) and number not in seen:
            selected.append(number)
            seen.add(number)

        if len(selected) == 3:
            break

    return selected

def format_score(score):
    if score is None:
        return "-"
    return f"{score:.3f}"


def print_result(rank, result, expected):
    print(
        f"TOP {rank} | SCORE {format_score(result.get('score'))} | "
        f"{result.get('source', '-')}"
    )

    preview = result.get("preview", "")
    print("PREVIEW:")
    print(preview if preview else "[NO TEXT]")

    found, missing = contains_expected(
        result.get("text", ""),
        expected,
    )

    print("EXPECTED FOUND:", ", ".join(found) if found else "-")
    print("EXPECTED MISSING:", ", ".join(missing) if missing else "-")
    print()


def run_test(number, test):
    question = test["question"]
    expected = test["expected"]

    rewritten = rewrite_query(question)
    search_result = search(rewritten)

    print("=" * 120)
    print(f"TEST #{number}")
    print()

    print("QUESTION:")
    print(question)
    print()

    print("REWRITTEN QUERY:")
    print(rewritten)
    print()

    print("EXPECTED TERMS:")
    print(", ".join(expected))
    print()

    if search_result["error"]:
        print("SEARCH ERROR:")
        print(search_result["error"])
        print()
        return

    results = search_result["results"]

    print("VECTOR SEARCH TOP-5:")
    print()

    for rank, result in enumerate(results, start=1):
        print_result(rank, result, expected)

    selected = rerank(
        question=question,
        rewritten_query=rewritten,
        results=results,
    )

    print("RERANKER ORDER:")
    if selected:
        print(", ".join(str(x) for x in selected))
    else:
        print("[RERANKER ERROR OR EMPTY RESULT]")

    print()

    print("RERANKED RESULTS:")
    print()

    reranked_results = []

    for new_rank, original_rank in enumerate(selected, start=1):
        result = results[original_rank - 1]
        reranked_results.append(result)

        print(
            f"RERANK {new_rank} <- VECTOR TOP {original_rank} | "
            f"SCORE {format_score(result.get('score'))} | "
            f"{result.get('source', '-')}"
        )

        print("PREVIEW:")
        print(result.get("preview", "") or "[NO TEXT]")

        found, missing = contains_expected(
            result.get("text", ""),
            expected,
        )

        print(
            "EXPECTED FOUND:",
            ", ".join(found) if found else "-"
        )

        print(
            "EXPECTED MISSING:",
            ", ".join(missing) if missing else "-"
        )

        print()



    # Простая автоматическая диагностическая метка.
    top2_text = " ".join(
        result.get("text", "")
        for result in reranked_results[:2]
    )

    found, missing = contains_expected(top2_text, expected)

    if not selected:
        auto_status = "RERANK_ERROR"
    elif not missing:
        auto_status = "PASS"
    elif found:
        auto_status = "BORDERLINE"
    else:
        auto_status = "FAIL"

    print("AUTO STATUS:", auto_status)
    print(
        "NOTE: AUTO STATUS проверяет только наличие ожидаемых терминов "
        "в TOP-2, а не правильность фактического ответа."
    )
    print()


if __name__ == "__main__":
    print()
    print("ANGIOPLUS RETRIEVAL DIAGNOSTIC TEST")
    print(f"TOTAL TESTS: {len(TESTS)}")
    print(f"PREVIEW LENGTH: {PREVIEW_LENGTH}")
    print()

    for i, test in enumerate(TESTS, start=1):
        run_test(i, test)