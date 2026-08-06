# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Sequence

EXPECTED_EMBEDDING_DIMENSION = 1024  # BAAI/bge-m3
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE = 4


class SentenceTransformerEmbedder:
    """Local embedding provider backed by sentence-transformers.

    Configuration is read from the environment:
      - LOCAL_EMBEDDING_MODEL  (default: BAAI/bge-m3)
      - EMBEDDING_DEVICE       (default: cpu)
      - EMBEDDING_BATCH_SIZE   (default: 4)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("LOCAL_EMBEDDING_MODEL")
            or DEFAULT_LOCAL_EMBEDDING_MODEL
        )
        self.device = (
            device
            or os.getenv("EMBEDDING_DEVICE")
            or DEFAULT_EMBEDDING_DEVICE
        )

        raw_batch = batch_size if batch_size is not None else os.getenv("EMBEDDING_BATCH_SIZE")
        if raw_batch is None:
            self.batch_size = DEFAULT_EMBEDDING_BATCH_SIZE
        else:
            try:
                self.batch_size = int(raw_batch)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "EMBEDDING_BATCH_SIZE must be an integer, "
                    f"got {raw_batch!r}"
                ) from exc

        if self.batch_size <= 0:
            raise RuntimeError(
                f"EMBEDDING_BATCH_SIZE must be positive, got {self.batch_size}"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: sentence-transformers. "
                "Install it with `pip install sentence-transformers` "
                "to use the local embedder."
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local embedding model {self.model_name!r} "
                f"on device {self.device!r}: {exc}"
            ) from exc

        try:
            dim = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read embedding dimension for {self.model_name!r}: {exc}"
            ) from exc

        if dim != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Unexpected embedding dimension for model {self.model_name!r}: "
                f"got {dim}, expected {EXPECTED_EMBEDDING_DIMENSION}. "
                "Make sure LOCAL_EMBEDDING_MODEL is BAAI/bge-m3."
            )

    @property
    def model_label(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("embed_query received an empty or whitespace-only input")
        vectors = self._encode([text])
        return vectors[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: numpy. "
                "Install it with `pip install numpy` "
                "to use the local embedder."
            ) from exc

        try:
            encoded = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Local embedding failed for model {self.model_name!r}: {exc}"
            ) from exc

        if encoded is None:
            raise RuntimeError(
                f"Local embedding returned no output for model {self.model_name!r}"
            )

        arr = np.asarray(encoded, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return [list(map(float, row)) for row in arr]
