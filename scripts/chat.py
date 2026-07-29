# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "angioplus_documents"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "app/prompts/system_prompt.md")


def embed_query(api_key: str, question: str, model_name: str) -> list[float]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model=model_name,
        content=question,
        task_type="retrieval_query",
    )
    return response["embedding"]


def search_qdrant(client: QdrantClient, collection: str, vector: list[float], limit: int):
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def build_context(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload or {}
        text = payload.get("text", "")
        parts.append(f"[Chunk {i}]\n{text}")
    return "\n\n".join(parts)

def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError as exc:
        logger.warning(
            "Falling back to the built-in system prompt, %r could not be read: %s",
            SYSTEM_PROMPT_PATH,
            exc,
        )
        return "Ты — технический ассистент поддержки AngioPlus Core. Отвечай строго на основе контекста."


def generate_answer(api_key: str, question: str, context: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    system_prompt = load_system_prompt()

    prompt = f"""
{system_prompt}

--------------------
RETRIEVED CONTEXT / НАЙДЕННЫЙ КОНТЕКСТ
--------------------

{context}

--------------------
USER QUESTION / ВОПРОС ПОЛЬЗОВАТЕЛЯ
--------------------

{question}
"""

    model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
    response = model.generate_content(prompt)

    answer = (response.text or "").strip()

    if not answer:
        raise RuntimeError(
            f"Gemini returned no answer (prompt_feedback={response.prompt_feedback})"
        )

    return answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--question", "-q", dest="question_flag", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"))
    return parser.parse_args()


def resolve_question(args):
    if args.question_flag:
        return args.question_flag
    if args.question:
        return args.question
    print("No question provided", file=sys.stderr)
    sys.exit(2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    args = parse_args()
    question = resolve_question(args)

    api_key = (args.gemini_api_key or "").strip()
    if not api_key:
        print("GEMINI_API_KEY required", file=sys.stderr)
        sys.exit(2)

    vector = embed_query(api_key, question, DEFAULT_GEMINI_EMBEDDING_MODEL)

    client = QdrantClient(url=args.qdrant_url)

    chunks = search_qdrant(
        client=client,
        collection=args.collection,
        vector=vector,
        limit=args.top_k,
    )

    context = build_context(chunks)

    answer = generate_answer(api_key, question, context)

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===\n")

    seen = set()
    relevant_sources = []

    for chunk in chunks:
        p = chunk.payload or {}
        key = (
            p.get("file_name"),
            p.get("section"),
            p.get("page_start"),
            p.get("page_end"),
        )

        if key in seen:
            continue

        seen.add(key)
        relevant_sources.append(chunk)

        if len(relevant_sources) >= 3:
            break

    for chunk in relevant_sources:
        p = chunk.payload or {}
        score = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
        print(
            f"{p.get('file_name')} | "
            f"{p.get('section')} | "
            f"pages {p.get('page_start')}-{p.get('page_end')} | "
            f"score {score}"
        )


if __name__ == "__main__":
    main()