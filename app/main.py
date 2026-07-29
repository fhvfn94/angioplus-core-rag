# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
from qdrant_client import QdrantClient

from rag_common import (
    DEFAULT_COLLECTION,
    DEFAULT_GEMINI_MODEL,
    build_context,
    build_context_block,
    iter_unique_chunks,
    load_system_prompt,
)
from rag_common.retrieval import search_qdrant as _search_qdrant

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
# New google-genai SDK expects the bare model id (no "models/" prefix).
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

app = FastAPI(title="AngioPlus Core RAG Assistant")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    file_name: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    score: float | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set",
        )

    return api_key


def create_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def embed_query(api_key: str, question: str) -> list[float]:
    client = create_gemini_client(api_key)

    response = client.models.embed_content(
        model=DEFAULT_GEMINI_EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no query embedding")

    embedding = response.embeddings[0]

    if not embedding.values:
        raise RuntimeError("Gemini returned an empty query embedding")

    return list(embedding.values)


def search_qdrant(vector: list[float], top_k: int):
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
    return _search_qdrant(client, DEFAULT_COLLECTION, vector, top_k)


def generate_answer(
    api_key: str,
    question: str,
    context: str,
) -> str:
    client = create_gemini_client(api_key)
    system_prompt = load_system_prompt()

    prompt = build_context_block(context, question)

    response = client.models.generate_content(
        model=DEFAULT_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )

    answer = response.text

    if not answer:
        return "Такой информации нет в имеющейся документации."

    return answer.strip()


def build_sources(chunks: list[Any]) -> list[Source]:
    sources: list[Source] = []

    relevant_chunks = [
        chunk
        for chunk in chunks
        if chunk.score is not None and chunk.score >= 0.80
    ]

    if not relevant_chunks and chunks:
        relevant_chunks = [chunks[0]]

    for chunk in iter_unique_chunks(relevant_chunks):
        payload = chunk.payload or {}

        sources.append(
            Source(
                file_name=payload.get("file_name"),
                section=payload.get("section"),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                score=(
                    round(float(chunk.score), 3)
                    if chunk.score is not None
                    else None
                ),
            )
        )

    return sources


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question is empty",
        )

    api_key = get_api_key()

    try:
        vector = embed_query(api_key, question)
        chunks = search_qdrant(vector, request.top_k)

        if not chunks:
            return AskResponse(
                answer="Not found in documentation.",
                sources=[],
            )

        context = build_context(chunks)
        answer = generate_answer(api_key, question, context)
        sources = build_sources(chunks)

        return AskResponse(
            answer=answer,
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG request failed: {exc}",
        ) from exc