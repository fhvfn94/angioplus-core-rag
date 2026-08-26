# -*- coding: utf-8 -*-
"""Tests for the Phase D1.5 STT evaluation phrase set (voice/stt_phrases.py)."""
from __future__ import annotations

from voice.stt_phrases import (
    STT_PHRASES,
    get_minimum_phrases,
    phrase_texts,
    phrases_by_category,
)

REQUIRED_MINIMUM = [
    "Как выполнить калибровку катетера?",
    "Что такое μFR?",
    "Как измерить остаточный μFR?",
    "Где посмотреть версию программного обеспечения?",
    "Что делать, если программа не реагирует на действия пользователя?",
]


def test_phrase_set_size_is_in_range():
    assert 15 <= len(STT_PHRASES) <= 25


def test_required_minimum_phrases_present():
    texts = phrase_texts()
    for phrase in REQUIRED_MINIMUM:
        assert phrase in texts, f"missing required phrase: {phrase!r}"


def test_minimum_phrases_flagged():
    minimum_ids = {p["id"] for p in get_minimum_phrases()}
    assert len(minimum_ids) >= 5
    for phrase in REQUIRED_MINIMUM:
        assert any(
            p["text"] == phrase and p["is_min"] for p in get_minimum_phrases()
        )


def test_entries_are_valid():
    for p in STT_PHRASES:
        assert p["id"]
        assert p["text"]
        assert p["category"]
        assert isinstance(p["is_min"], bool)
    ids = [p["id"] for p in STT_PHRASES]
    assert len(ids) == len(set(ids)), "phrase ids must be unique"


def test_categories_present():
    cats = {p["category"] for p in STT_PHRASES}
    assert "catheter/calibration" in cats
    assert "angiography" in cats
    assert "mufr" in cats
    assert "software/interface" in cats
    assert "short" in cats
