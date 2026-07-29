# -*- coding: utf-8 -*-
"""
STEP 2: retrieval-only QA: embed the question with Gemini and search Qdrant.
Does not call an LLM for a final answer.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient

from rag_common import DEFAULT_COLLECTION, DEFAULT_GEMINI_EMBEDDING_MODEL, search_qdrant
from rag_common.gemini import embed_query

# Keep defaults aligned with scripts/ingest_documents.py
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
PREVIEW_MAX_CHARS = 400


def format_preview(text: str, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top chunks from Qdrant for a question (Gemini query embedding; no LLM answer).",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question text. If omitted, use --question or read from stdin.",
    )
    parser.add_argument(
        "--question",
        "-q",
        dest="question_flag",
        default=None,
        help="Question text (alternative to positional arg).",
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL.")
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"), help="Qdrant API key.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key.")
    parser.add_argument(
        "--gemini-embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL),
        help="Must match ingest (e.g. models/gemini-embedding-001).",
    )
    return parser.parse_args()


def resolve_question(args: argparse.Namespace) -> str:
    if args.question_flag is not None:
        return args.question_flag.strip()
    if args.question is not None and str(args.question).strip():
        return str(args.question).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Error: provide a question as an argument, --question, or via stdin.", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    args = parse_args()
    question = resolve_question(args)
    if not question:
        print("Error: empty question.", file=sys.stderr)
        sys.exit(2)

    api_key = (args.gemini_api_key or "").strip()
    if not api_key:
        print(
            "Error: GEMINI_API_KEY is required (or pass --gemini-api-key) to embed the query.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        vector = embed_query(api_key, question, args.gemini_embedding_model)
    except Exception as exc:
        print(f"Error: failed to embed query: {exc}", file=sys.stderr)
        sys.exit(2)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    try:
        hits = search_qdrant(
            client=client,
            collection=args.collection,
            vector=vector,
            limit=args.top_k,
        )
    except Exception as exc:
        print(f"Error: Qdrant search failed: {exc}", file=sys.stderr)
        sys.exit(2)

    print("Question:")
    print(textwrap.indent(question.strip(), prefix="  "))
    print()

    print(f"Matched chunks: {len(hits)} (top_k={args.top_k})")
    print()

    if not hits:
        print("(No hits - check collection name, ingestion, and embedding model/size.)")

    for i, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text = payload.get("text") or ""

        print(f"--- Chunk {i} ---")
        print(f"score:       {hit.score}")
        print(f"file_name:   {payload.get('file_name', '')}")
        print(f"section:     {payload.get('section', '')}")
        print(f"page_start:  {payload.get('page_start', '')}")
        print(f"page_end:    {payload.get('page_end', '')}")
        print("text preview:")
        wrapped = format_preview(text)
        print(textwrap.indent(wrapped, prefix="  ") if wrapped else "  <empty>")
        print()


if __name__ == "__main__":
    main()
