from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

from .client import YandexVectorStoreClient


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python -m yandex.retrieval.search_test "search query"'
        )

    query = " ".join(sys.argv[1:]).strip()

    client = YandexVectorStoreClient()

    response = client.search(
        query,
        max_num_results=5,
        rewrite_query=False,
    )

    print()
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(response.query)

    print()
    print("=" * 80)
    print("LATENCY")
    print("=" * 80)
    print(f"{response.latency_ms:.1f} ms")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    if not response.results:
        print("No results.")
        return

    for index, result in enumerate(response.results, start=1):
        print()
        print(f"[{index}]")
        print(f"score:    {result.score:.6f}")
        print(f"filename: {result.filename}")
        print(f"file_id:  {result.file_id}")
        print(f"attrs:    {json.dumps(result.attributes, ensure_ascii=False)}")
        print()
        print(result.text)
        print("-" * 80)


if __name__ == "__main__":
    main()