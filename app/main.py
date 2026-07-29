# -*- coding: utf-8 -*-
from __future__ import annotations

import hmac
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

DEFAULT_COLLECTION = "angioplus_documents"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT_PATH = os.getenv(
    "SYSTEM_PROMPT_PATH",
    "app/prompts/system_prompt.md",
)
MAX_QUESTION_LENGTH = 2000
MAX_TOP_K = 20

logger = logging.getLogger(__name__)

app = FastAPI(title="AngioPlus Core RAG Assistant")


def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("RAG_API_KEY", "").strip()

    if not expected:
        logger.error("RAG_API_KEY is not set, refusing to serve /ask")
        raise HTTPException(
            status_code=503,
            detail="Service is not configured",
        )

    if not hmac.compare_digest(x_api_key.strip(), expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


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
    client = QdrantClient(
        url=DEFAULT_QDRANT_URL,
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    response = client.query_points(
        collection_name=DEFAULT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    return list(response.points)


def build_context(chunks: list[Any]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload or {}
        text = payload.get("text", "")
        parts.append(f"[Chunk {i}]\n{text}")

    return "\n\n".join(parts)


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return (
            "Ты — технический ассистент поддержки AngioPlus Core. "
            "Отвечай строго на основе контекста."
        )


def generate_answer(
    api_key: str,
    question: str,
    context: str,
) -> str:
    client = create_gemini_client(api_key)
    system_prompt = load_system_prompt()

    prompt = f"""
--------------------
RETRIEVED CONTEXT / НАЙДЕННЫЙ КОНТЕКСТ
--------------------

{context}

--------------------
USER QUESTION / ВОПРОС ПОЛЬЗОВАТЕЛЯ
--------------------

{question}
""".strip()

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
    seen = set()
    sources: list[Source] = []

    relevant_chunks = [
        chunk
        for chunk in chunks
        if chunk.score is not None and chunk.score >= 0.80
    ]

    if not relevant_chunks and chunks:
        relevant_chunks = [chunks[0]]

    for chunk in relevant_chunks:
        payload = chunk.payload or {}

        key = (
            payload.get("file_name"),
            payload.get("section"),
            payload.get("page_start"),
            payload.get("page_end"),
        )

        if key in seen:
            continue

        seen.add(key)

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


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key)],
)
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
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="RAG request failed",
        ) from exc