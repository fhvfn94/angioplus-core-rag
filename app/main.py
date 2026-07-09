# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

DEFAULT_COLLECTION = "angioplus_documents"
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "app/prompts/system_prompt.md")

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
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
    return api_key


def embed_query(api_key: str, question: str) -> list[float]:
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model=DEFAULT_GEMINI_EMBEDDING_MODEL,
        content=question,
        task_type="retrieval_query",
    )
    return response["embedding"]


def search_qdrant(vector: list[float], top_k: int):
    client = QdrantClient(url=DEFAULT_QDRANT_URL)
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
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Ты — технический ассистент поддержки AngioPlus Core. Отвечай строго на основе контекста."

def generate_answer(api_key: str, question: str, context: str) -> str:
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
    return response.text or "Такой информации нет в имеющейся документации."


def build_sources(chunks: list[Any]) -> list[Source]:
    seen = set()
    sources: list[Source] = []

    relevant_chunks = [chunk for chunk in chunks if chunk.score >= 0.80]

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
                score=round(float(chunk.score), 3) if chunk.score is not None else None,
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
        raise HTTPException(status_code=400, detail="Question is empty")

    api_key = get_api_key()

    vector = embed_query(api_key, question)
    chunks = search_qdrant(vector, request.top_k)

    if not chunks:
        return AskResponse(answer="Not found in documentation.", sources=[])

    context = build_context(chunks)
    answer = generate_answer(api_key, question, context)
    sources = build_sources(chunks)

    return AskResponse(answer=answer, sources=sources)