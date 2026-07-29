# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient

from rag_common import (
    DEFAULT_COLLECTION,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    build_context,
    build_context_block,
    iter_unique_chunks,
    load_system_prompt,
    search_qdrant,
)
from rag_common.gemini import embed_query

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def generate_answer(api_key: str, question: str, context: str):
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    system_prompt = load_system_prompt()

    prompt = f"{system_prompt}\n\n{build_context_block(context, question)}"

    model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
    response = model.generate_content(prompt)

    return response.text


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

    relevant_sources = list(islice(iter_unique_chunks(chunks), 3))

    for chunk in relevant_sources:
        p = chunk.payload or {}
        print(
            f"{p.get('file_name')} | "
            f"{p.get('section')} | "
            f"pages {p.get('page_start')}-{p.get('page_end')} | "
            f"score {chunk.score:.3f}"
        )


if __name__ == "__main__":
    main()