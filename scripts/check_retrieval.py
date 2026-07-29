import os

from app import main as rag


QUESTION = "Как установить AngioPlus Core?"
TOP_K = 5


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")

    print("=" * 80)
    print("QUESTION:")
    print(QUESTION)
    print("=" * 80)

    vector = rag.embed_query(api_key, QUESTION)
    results = rag.search_qdrant(vector, top_k=TOP_K)

    if not results:
        print("No retrieval results.")
        return

    for i, point in enumerate(results, start=1):
        payload = point.payload or {}

        print()
        print("-" * 80)
        print(f"RANK     : {i}")
        score = f"{point.score:.6f}" if point.score is not None else "n/a"

        print(f"SCORE    : {score}")
        print(f"FILE     : {payload.get('file_name')}")
        print(f"TYPE     : {payload.get('document_type')}")
        print(f"SECTION  : {payload.get('section')}")
        print(f"PAGES    : {payload.get('page_start')} - {payload.get('page_end')}")
        print()
        print("TEXT:")
        print((payload.get("text") or "")[:700])


if __name__ == "__main__":
    main()