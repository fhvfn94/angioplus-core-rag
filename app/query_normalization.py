# -*- coding: utf-8 -*-
"""Deterministic query normalization (no external LLM, no fuzzy matching).

Applies, in order:
1. whitespace collapse + trim;
2. exact STT word-gap fixes;
3. exact domain term variants -> canonical form.

Only verified, narrow replacements from app.domain_terms are used. Unknown
words are never modified.
"""

from __future__ import annotations

import re

from app.domain_terms import TERM_MAP, WORD_GAP_FIXES

_WS_RE = re.compile(r"\s+")


def normalize_user_query(text: str) -> str:
    if not text:
        return text

    s = _WS_RE.sub(" ", text).strip()

    # Word-gap fixes first (so that, e.g., "у становить" -> "установить").
    for gap, repl in WORD_GAP_FIXES:
        s = re.sub(rf"(?i)\b{re.escape(gap)}\b", repl, s)

    # Then domain terms (exact phrases, word-boundary, case-insensitive).
    for variant, canonical in TERM_MAP:
        s = re.sub(rf"(?i)\b{re.escape(variant)}\b", canonical, s)

    return s
