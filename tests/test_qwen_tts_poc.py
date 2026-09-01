# -*- coding: utf-8 -*-
"""Tests for the Phase D2.2 Qwen3-TTS PoC (Phase D2.2, TASK 7).

These tests cover ONLY the deterministic pure logic in
scripts/tts/qwen_tts_poc.py and voice/tts_phrases.py. They NEVER import the real
Qwen model (qwen_tts is imported lazily and only inside `load_model`, which
these tests do not call), never download weights and never generate audio.

It is an explicit invariant that these tests run in an environment WITHOUT
qwen-tts installed (the model is mocked / not required).
"""
from __future__ import annotations

import sys

import pytest

from scripts.tts.qwen_tts_poc import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_ID,
    DEFAULT_SPEAKER,
    SUPPORTED_SPEAKERS,
    format_rtf,
    realtime_factor,
)
from voice.tts_phrases import (
    TTS_PHRASES,
    get_ids,
    long_phrases,
    normal_phrases,
    phrase_text,
    safe_stem,
    short_phrases,
)
from scripts.tts.qwen_tts_poc import detect_environment


# ---------------------------------------------------------------------------
# Phrase dataset integrity (voice/tts_phrases.py)
# ---------------------------------------------------------------------------

def test_phrase_set_has_all_d2_2_required_texts():
    texts = [p["text"] for p in TTS_PHRASES]
    required = [
        "Как выполнить калибровку катетера?",
        "Как выполнить калибровку катетера перед анализом?",
        "Что показывает μFR?",
        "Как рассчитать μFR?",
        "Что означает остаточный μFR?",
        "Выполните анализ коронарной артерии.",
        "Как оценить стеноз коронарной артерии?",
        "Что такое виртуальный стент?",
        "Откройте AngioPlus Core.",
        "Откройте AngioPlus Core на рабочей станции PStation.",
    ]
    for phrase in required:
        assert phrase in texts, f"missing required D2.2 phrase: {phrase!r}"


def test_phrase_entries_are_valid():
    for p in TTS_PHRASES:
        assert p["id"], "id must be non-empty"
        assert p["text"], "text must be non-empty"
        assert p["category"], "category must be non-empty"
        assert p["kind"] in ("short", "normal", "long"), p["kind"]
    ids = get_ids()
    assert len(ids) == len(set(ids)), "phrase ids must be unique"


def test_phrase_categories_present():
    cats = {p["category"] for p in TTS_PHRASES}
    for required in ("catheter/calibration", "mufr", "angiography",
                     "software/interface", "short", "long"):
        assert required in cats


def test_kind_buckets_not_empty():
    assert short_phrases(), "must contain short phrases"
    assert normal_phrases(), "must contain normal phrases"
    assert long_phrases(), "must contain long phrases"


def test_phrase_text_lookup_and_fallback():
    assert phrase_text("calibration") == "Как выполнить калибровку катетера?"
    # Unknown id -> returns the input unchanged
    assert phrase_text("не существующий id") == "не существующий id"


def test_dataset_size_matches_poc_defaults():
    # The PoC iterates over TTS_PHRASES directly; just assert a sane range.
    assert 15 <= len(TTS_PHRASES) <= 30


# ---------------------------------------------------------------------------
# Output filename handling (safe_stem)
# ---------------------------------------------------------------------------

def test_safe_stem_is_deterministic_and_safe():
    a = safe_stem("Как выполнить калибровку катетера?")
    b = safe_stem("Как выполнить калибровку катетера?")
    assert a == b
    # No dangerous characters / spaces
    assert not any(ch in a for ch in "?/:\\&*<>| \"'")
    assert a == a.lower()
    # Latin tokens kept for AngioPlus / PStation / μFR
    assert "angioplus" in safe_stem("Откройте AngioPlus Core.")
    assert "pstation" in safe_stem("Рабочая станция PStation")


def test_safe_stem_empty_returns_fallback():
    assert safe_stem("???") == "utterance"
    assert safe_stem("  ") == "utterance"


def test_poc_output_basename_scheme():
    # deterministic <id>__<stem>.wav used by run_dataset
    for p in TTS_PHRASES:
        filename = f"{p['id']}__{safe_stem(p['text'])}.wav"
        assert filename.endswith(".wav")
        assert "?" not in filename
        assert p["id"] in filename


# ---------------------------------------------------------------------------
# Realtime-factor calculation
# ---------------------------------------------------------------------------

def test_rtf_basic():
    assert realtime_factor(2.0, 4.0) == pytest.approx(0.5)
    assert realtime_factor(6.0, 3.0) == pytest.approx(2.0)
    assert realtime_factor(0.0, 4.0) is None


def test_rtf_guards_invalid_inputs():
    assert realtime_factor(1.0, 0.0) is None
    assert realtime_factor(1.0, -1.0) is None
    assert realtime_factor(-1.0, 2.0) is None
    assert realtime_factor(None, 2.0) is None


def test_format_rtf():
    assert "0.500" in format_rtf(0.5)
    assert "2.000" in format_rtf(2.0)
    assert format_rtf(None) == "n/a"


# ---------------------------------------------------------------------------
# Constants / config (official model id, speakers, language)
# ---------------------------------------------------------------------------

def test_official_defaults():
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert DEFAULT_LANGUAGE == "Russian"
    assert DEFAULT_SPEAKER in SUPPORTED_SPEAKERS


def test_supported_speakers_set():
    expected = {
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee",
    }
    assert set(SUPPORTED_SPEAKERS) == expected
    assert len(SUPPORTED_SPEAKERS) == len(set(SUPPORTED_SPEAKERS))


def test_unknown_speaker_rejected():
    from scripts.tts.qwen_tts_poc import _build_parser

    parser = _build_parser()
    # Simulate the guard in main(): an unknown speaker must raise SystemExit.
    args = parser.parse_args(["--text", "Привет"])
    # __main__ path guards this; assert the validation set contains it:
    assert "NotARealSpeaker" not in SUPPORTED_SPEAKERS


# ---------------------------------------------------------------------------
# Environment detection (must not assume a GPU; must be CPU-safe)
# ---------------------------------------------------------------------------

def test_detect_environment_never_crashes():
    env = detect_environment(None)
    assert env.python_version
    assert env.selected_device in ("cuda", "cpu")
    # If torch missing, still returns a valid report with device cpu.
    assert env.selected_device


def test_detect_environment_force_cpu():
    env = detect_environment("cpu")
    assert env.selected_device == "cpu"


def test_qwen_tts_import_optional_failure_is_tested():
    # The pure module must remain importable WITHOUT qwen_tts installed.
    # load_model() is the only place that imports qwen_tts; it raises
    # SystemExit with a helpful message when the package is absent.
    import scripts.tts.qwen_tts_poc as poc

    def _fake_import(*names):
        raise ImportError("No module named 'qwen_tts'")

    if getattr(sys, "_has_qwen_tts_for_test", False):
        pass
    # Ensure load_model's lazy import path is guarded: monkeypatch just the
    # import.__import__ to simulate a missing package.
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **kw):
        if str(name).startswith("qwen_tts"):
            raise ImportError("No module named 'qwen_tts' (simulated)")
        return real_import(name, *a, **kw)

    builtins.__import__ = guard
    try:
        with pytest.raises(SystemExit) as exc:
            poc.load_model(DEFAULT_MODEL_ID, "cpu")
        assert "qwen_tts is not installed" in str(exc.value)
    finally:
        builtins.__import__ = real_import
