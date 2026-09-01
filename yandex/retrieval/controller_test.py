from __future__ import annotations

from dotenv import load_dotenv

from .controller import search_angioplus_knowledge


QUERIES = [
    "Какая версия программного обеспечения AngioPlus Core?",
    "Какой DICOM порт используется по умолчанию?",
    "Какой минимальный frame rate нужен для μFR?",
    "Можно ли двигать стол во время ангиографической съемки?",
    "Какой формат DICOM поддерживает AngioPlus Core?",
]


def main() -> None:
    load_dotenv()

    for query in QUERIES:
        response = search_angioplus_knowledge(query)

        print()
        print("=" * 100)
        print(f"QUERY: {response.query}")
        print(f"LATENCY: {response.latency_ms:.1f} ms")
        print("=" * 100)

        for item in response.evidence:
            print(
                f"#{item.rank} "
                f"score={item.score:.6f} "
                f"type={item.source_type:<10} "
                f"authority={item.authority:<3} "
                f"source={item.source!r} "
                f"section={item.section!r} "
                f"pages={item.pages!r}"
            )


if __name__ == "__main__":
    main()