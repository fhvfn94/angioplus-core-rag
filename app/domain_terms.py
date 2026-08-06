# -*- coding: utf-8 -*-
"""Domain term dictionary for AngioPlus RAG.

Contains:
- canonical domain entities,
- exact STT word-gap fixes,
- exact term variants -> canonical form.

Only verified, narrow replacements are allowed here. Do NOT add broad
fuzzy/LCS/Levenshtein matching. Expand TERM_MAP manually only with
confirmed examples.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical domain entities of the first phase.
CANONICAL_ENTITIES: tuple[str, ...] = (
    "AngioPlus Core",
    "Pulse Medical",
    "PStation",
    "DICOM",
    "\u00b5FR",  # µFR
)

# Exact STT word-gap fixes (applied BEFORE domain terms).
# Order matters: more specific / multi-word gaps first.
WORD_GAP_FIXES: tuple[tuple[str, str], ...] = (
    ("у становить", "установить"),
)

# Exact term variants -> canonical form (applied AFTER gap fixes).
# Put longer/more specific phrases before shorter ones.
TERM_MAP: tuple[tuple[str, str], ...] = (
    ("пульс мьюзикл", "Pulse Medical"),
    ("пульс медикал", "Pulse Medical"),
    ("пьюлс мьюзикл", "Pulse Medical"),
    ("ангио плюс кор", "AngioPlus Core"),
    ("ангиоплюс кор", "AngioPlus Core"),
    ("пи стейшн", "PStation"),
    ("диком", "DICOM"),
)


@dataclass(frozen=True)
class EntityMatch:
    """A canonical entity occurrence found in a question."""

    entity: str
    start: int
    end: int


def _entity_patterns() -> list[tuple[str, str]]:
    """Return (lowercased canonical entity, canonical entity) pairs."""
    return [(e.casefold(), e) for e in CANONICAL_ENTITIES]


def extract_entities(question: str) -> list[EntityMatch]:
    """Return canonical entity matches sorted by position in the text.

    A single repeated entity yields multiple matches; downstream logic
    deduplicates by entity name.
    """
    if not question:
        return []
    q_lower = question.casefold()
    matches: list[EntityMatch] = []
    for e_lower, e in _entity_patterns():
        start = 0
        while True:
            idx = q_lower.find(e_lower, start)
            if idx == -1:
                break
            matches.append(EntityMatch(entity=e, start=idx, end=idx + len(e_lower)))
            start = idx + len(e_lower)
    matches.sort(key=lambda m: (m.start, m.end))
    return matches


def distinct_entities(question: str) -> list[str]:
    """Return unique canonical entities present, in first-appearance order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in extract_entities(question):
        if m.entity not in seen_set:
            seen_set.add(m.entity)
            seen.append(m.entity)
    return seen
