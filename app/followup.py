# -*- coding: utf-8 -*-
"""Deterministic follow-up detection and constrained rewrite (MVP).

Follow-up detection uses a combined heuristic (context hints, continuation
starters, brevity, and explicit-domain-entity override). Rewriting only
supports a small set of safe, first-phase templates. Ambiguity (two or more
distinct canonical entities in the last standalone question) disables
rewriting. Unknown or unsupported constructions are returned unchanged.
"""

from __future__ import annotations

import re

from app.domain_terms import distinct_entities
from app.query_normalization import normalize_user_query

# Context signals indicating a follow-up question.
FOLLOWUP_HINTS: tuple[str, ...] = (
    "его",
    "её",
    "их",
    "это",
    "там",
    "так",
    "после этого",
    "а как",
    "а где",
    "а что",
    "что дальше",
    "и дальше",
    "потом",
    "после",
    "а если",
    "где это",
    "у него",
    "с ним",
    "для него",
)

CONTINUATION_STARTERS: tuple[str, ...] = (
    "а",
    "и",
    "тогда",
    "после",
    "потом",
)

# Templates that are safe to rewrite in phase one.
# Verb templates capture a verb token that may not contain a trailing '?'.
_VERB_TEMPLATES: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"^а?\s*как\s+(?:его|её|их)\s+(?P<verb>[^\s?]+)\??$"),
        "Как {verb} {entity}?",
    ),
    (
        re.compile(r"^как\s+(?:его|её|их)\s+(?P<verb>[^\s?]+)\??$"),
        "Как {verb} {entity}?",
    ),
    (
        re.compile(r"^а?\s*где\s+(?:его|её|их)\s+(?P<verb>[^\s?]+)\??$"),
        "Где {verb} {entity}?",
    ),
    (
        re.compile(r"^где\s+(?:его|её|их)\s+(?P<verb>[^\s?]+)\??$"),
        "Где {verb} {entity}?",
    ),
    (
        re.compile(r"^(?:а\s+)?кто\s+может\s+(?:его|её|их)\s+(?P<verb>[^\s?]+)\??$"),
        "Кто может {verb} {entity}?",
    ),
)

# No-verb templates (only the entity is inserted).
_NO_VERB_TEMPLATES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^а?\s*что\s+это\??$"), "Что такое {entity}?"),
    (re.compile(r"^что\s+это\??$"), "Что такое {entity}?"),
)


def _has_any_entity(question: str) -> bool:
    return len(distinct_entities(question)) > 0


def is_follow_up_candidate(
    current_norm: str,
    last_standalone: str | None,
) -> bool:
    """Return True only if there is prior context and a clear follow-up signal.

    A question with an explicit domain entity is treated as standalone even if
    short. Brevity alone is NOT sufficient.
    """
    if not last_standalone:
        return False
    if not current_norm:
        return False
    c = current_norm.casefold()

    has_hint = any(h in c for h in FOLLOWUP_HINTS)
    starts_continuation = any(
        c.startswith(w + " ") or c == w for w in CONTINUATION_STARTERS
    )
    explicit_entity = _has_any_entity(current_norm)

    if explicit_entity:
        return False

    return has_hint or starts_continuation


def rewrite_followup_question(
    current_norm: str,
    last_standalone: str | None,
) -> tuple[str, bool]:
    """Return (standalone_question, used_history).

    used_history is True only when a safe template matched AND the prior
    context provides exactly one distinct entity. On ambiguity, no-history,
    or unsupported construction, the normalized current question is returned.
    """
    if not last_standalone:
        return current_norm, False
    if not is_follow_up_candidate(current_norm, last_standalone):
        return current_norm, False

    # Ambiguity: more than one distinct entity in the last standalone question.
    entities = distinct_entities(last_standalone)
    if len(entities) != 1:
        return current_norm, False

    entity = entities[0]

    # Try safe templates on the normalized current question (case-insensitive).
    c_norm = normalize_user_query(current_norm)
    c_lower = c_norm.casefold().strip()

    for pattern, template in _VERB_TEMPLATES:
        match = pattern.match(c_lower)
        if not match:
            continue
        verb = match.group("verb")
        if not verb:
            continue
        standalone = template.format(verb=verb, entity=entity)
        return normalize_user_query(standalone), True

    for pattern, template in _NO_VERB_TEMPLATES:
        match = pattern.match(c_lower)
        if not match:
            continue
        standalone = template.format(entity=entity)
        return normalize_user_query(standalone), True

    # Unsupported construction: do not guess, do not permute words.
    return current_norm, False
