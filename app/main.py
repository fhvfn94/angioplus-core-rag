# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "angioplus_documents"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT_PATH = os.getenv(
    "SYSTEM_PROMPT_PATH",
    "app/prompts/system_prompt.md",
)
FALLBACK_SYSTEM_PROMPT = (
    "Ты — технический ассистент поддержки AngioPlus Core. "
    "Отвечай строго на основе контекста."
)
NOT_FOUND_ANSWER = "Такой информации нет в имеющейся документации."

app = FastAPI(title="AngioPlus Core RAG Assistant")


class EmbeddingError(RuntimeError):
    """Query embedding could not be produced."""


class RetrievalError(RuntimeError):
    """Vector search failed."""


class GenerationError(RuntimeError):
    """Answer generation failed or was blocked."""


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)


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
        logger.error("GEMINI_API_KEY is not set; cannot serve requests")
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set",
        )

    return api_key


def create_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def embed_query(api_key: str, question: str) -> list[float]:
    client = create_gemini_client(api_key)

    try:
        response = client.models.embed_content(
            model=DEFAULT_GEMINI_EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Gemini embedding request failed: {exc}"
        ) from exc

    if not response.embeddings:
        raise EmbeddingError("Gemini returned no query embedding")

    embedding = response.embeddings[0]

    if not embedding.values:
        raise EmbeddingError("Gemini returned an empty query embedding")

    return list(embedding.values)


def search_qdrant(vector: list[float], top_k: int):
    client = QdrantClient(url=DEFAULT_QDRANT_URL)

    try:
        response = client.query_points(
            collection_name=DEFAULT_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Qdrant search in collection "
            f"{DEFAULT_COLLECTION!r} failed: {exc}"
        ) from exc
    finally:
        client.close()

    return list(response.points)


def build_context(chunks: list[Any]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload or {}
        text = (payload.get("text") or "").strip()

        if not text:
            logger.warning("Chunk %s has no text payload, skipping it", i)
            continue

        parts.append(f"[Chunk {i}]\n{text}")

    return "\n\n".join(parts)


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as file:
            prompt = file.read().strip()
    except OSError as exc:
        logger.warning(
            "Falling back to the built-in system prompt, "
            "%r could not be read: %s",
            SYSTEM_PROMPT_PATH,
            exc,
        )
        return FALLBACK_SYSTEM_PROMPT

    if not prompt:
        logger.warning(
            "Falling back to the built-in system prompt, %r is empty",
            SYSTEM_PROMPT_PATH,
        )
        return FALLBACK_SYSTEM_PROMPT

    return prompt


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

    try:
        response = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
    except Exception as exc:
        raise GenerationError(
            f"Gemini generation request failed: {exc}"
        ) from exc

    answer = (response.text or "").strip()

    if answer:
        return answer

    block_reason = (
        response.prompt_feedback.block_reason
        if response.prompt_feedback is not None
        else None
    )
    finish_reasons = [
        candidate.finish_reason for candidate in response.candidates or []
    ]

    if block_reason is not None or any(
        reason is not None and reason != types.FinishReason.STOP
        for reason in finish_reasons
    ):
        raise GenerationError(
            f"Gemini returned no answer "
            f"(block_reason={block_reason}, "
            f"finish_reasons={finish_reasons})"
        )

    logger.warning(
        "Gemini returned an empty answer without a block reason, "
        "responding with the not-found message"
    )

    return NOT_FOUND_ANSWER


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
    except EmbeddingError as exc:
        logger.exception("Query embedding failed")
        raise HTTPException(
            status_code=502,
            detail="Embedding provider is unavailable",
        ) from exc

    try:
        chunks = search_qdrant(vector, request.top_k)
    except RetrievalError as exc:
        logger.exception("Document retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Document store is unavailable",
        ) from exc

    if not chunks:
        return AskResponse(
            answer=NOT_FOUND_ANSWER,
            sources=[],
        )

    context = build_context(chunks)

    if not context.strip():
        logger.warning(
            "Retrieved %s chunks without any text payload; "
            "answering with the not-found message",
            len(chunks),
        )
        return AskResponse(
            answer=NOT_FOUND_ANSWER,
            sources=build_sources(chunks),
        )

    try:
        answer = generate_answer(api_key, question, context)
    except GenerationError as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed",
        ) from exc

    return AskResponse(
        answer=answer,
        sources=build_sources(chunks),
    )