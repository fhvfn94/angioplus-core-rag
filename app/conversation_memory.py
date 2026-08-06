# -*- coding: utf-8 -*-
"""In-memory short conversation store (MVP, no Redis).

State is kept per (conversation_id, user_id) pair and never mixed.

Constraints:
- maximum CONVERSATION_MAX_TURNS turns per conversation;
- TTL = CONVERSATION_TTL_SECONDS;
- each stored question is capped at a maximum length;
- in-memory only; lost on restart (acceptable for MVP).

Lock usage is strictly limited to:
- TTL cleanup,
- reading/copying a snapshot of a state,
- appending a successfully processed turn.
The lock is NEVER held during embedding / Qdrant / gate / generation or any
network operation.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    """A single processed user turn (never stores secrets).

    was_secret is intentionally absent: secret requests are NEVER persisted.
    """

    normalized_question: str
    standalone_question: str
    timestamp: float


@dataclass
class ConversationState:
    conversation_id: str
    user_id: str
    turns: deque[Turn] = field(default_factory=deque)
    updated_at: float = field(default_factory=time.time)


class ConversationMemory:
    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_turns: int = 3,
        max_question_len: int = 300,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_question_len = max_question_len
        self._store: dict[tuple[str, str], ConversationState] = {}
        self._lock = threading.Lock()

    def _truncate(self, text: str) -> str:
        if not text:
            return text
        if len(text) > self.max_question_len:
            return text[: self.max_question_len]
        return text

    def _prune_expired(self, now: float) -> None:
        """Drop entries whose updated_at is older than the TTL.

        Called while holding the lock (internal).
        """
        cutoff = now - self.ttl_seconds
        stale = [
            key
            for key, state in self._store.items()
            if state.updated_at < cutoff
        ]
        for key in stale:
            del self._store[key]

    def get_snapshot(self, conversation_id: str, user_id: str) -> ConversationState:
        """Return a Copy of the state (or an empty state) without holding
        the lock beyond the read. Safe to use while doing network calls."""
        now = time.time()
        key = (conversation_id, user_id)
        with self._lock:
            self._prune_expired(now)
            existing = self._store.get(key)
            if existing is None:
                # Return a snapshot so the caller never mutates store directly.
                snapshot = ConversationState(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    turns=deque(),
                    updated_at=now,
                )
                return snapshot
            return ConversationState(
                conversation_id=existing.conversation_id,
                user_id=existing.user_id,
                turns=deque(existing.turns),
                updated_at=existing.updated_at,
            )

    def append_turn(
        self,
        conversation_id: str,
        user_id: str,
        normalized_question: str,
        standalone_question: str,
    ) -> None:
        """Append a successfully processed turn.

        Must be called only with a safe standalone question (never a secret,
        never empty). The lock is held only around the append (no I/O).
        """
        normalized_question = self._truncate(normalized_question)
        standalone_question = self._truncate(standalone_question)
        if not normalized_question and not standalone_question:
            return
        now = time.time()
        key = (conversation_id, user_id)
        turn = Turn(
            normalized_question=normalized_question,
            standalone_question=standalone_question,
            timestamp=now,
        )
        with self._lock:
            self._prune_expired(now)
            state = self._store.get(key)
            if state is None:
                state = ConversationState(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    turns=deque(maxlen=self.max_turns),
                    updated_at=now,
                )
                self._store[key] = state
            state.turns.append(turn)
            state.updated_at = now
