from __future__ import annotations

from typing import Any


_ALLOWED_ROLES = {"user", "assistant", "operator"}


def build_conversation_history(messages: list[Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []

    for message in messages:
        role = str(message.role).strip().lower()
        text = str(message.text).strip()

        if role not in _ALLOWED_ROLES:
            continue

        if not text:
            continue

        history.append(
            {
                "role": role,
                "text": text,
            }
        )

    return history