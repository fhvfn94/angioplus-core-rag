# -*- coding: utf-8 -*-
"""Tests for the STT benchmark helper normalization + CER/WER (Phase D1.5).

These tests cover only the deterministic pure functions in
scripts/stt_benchmark.py (no model, no audio, no wall-clock timing).
"""
from __future__ import annotations

from scripts.stt_benchmark import _cer, _wer, normalize


def test_normalize_conservative():
    assert normalize("  Как  выполнить  калибровку катетера?  ") == (
        "как выполнить калибровку катетера"
    )
    assert normalize("Что такое μFR?") == "что такое μfr"


def test_normalize_empty_and_punct():
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_cer_perfect_and_shift():
    ref = "Как выполнить калибровку катетера?"
    # identical
    assert _cer(ref, ref) == 0.0
    # one character change
    assert 0.0 < _cer(ref, "Как выполнить калибровку катерра?") < 0.2


def test_wer_perfect():
    ref = "Как выполнить калибровку катетера?"
    assert _wer(ref, ref) == 0.0


def test_cer_wer_empty_reference():
    # A single word hypothesis against empty reference = fully wrong (1.0)
    assert _cer("", "катетер") == 1.0
    assert _wer("", "катетер") == 1.0
    # both empty = perfect
    assert _cer("", "") == 0.0
    assert _wer("", "") == 0.0
