import os
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

QUERIES = [
    "2.5.0.0.0",
    "AngioPlus Core Номер версии ПО 2.5.0.0.0",
    "AngioPlus Core software version Номер версии ПО",
    "μFR 1108",
    "Параметры по умолчанию μFR 1108",
    "AngioPlus Core DICOM AE Title μFR port 1108",
]


def get_text(item):
    parts = []

    for content in item.get("content", []):
        if content.get("type") == "text":
            text = content.get("text", "")
            if text:
                parts.append(text)

    return "\n".join(parts)


for query in QUERIES:
    print("=" * 100)
    print("QUERY:")
    print(query)
    print()

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
        print("ERROR:", response.status_code, response.text)
        continue

    data = response.json()

    for rank, item in enumerate(data.get("data", []), start=1):
        text = get_text(item)

        print(
            f"TOP {rank} | SCORE {item.get('score')}"
        )

        if len(text) > 1200:
            text = text[:1200] + "..."

        print(text)
        print()