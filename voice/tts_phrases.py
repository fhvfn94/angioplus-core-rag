# -*- coding: utf-8 -*-
"""AngioPlus Russian TTS evaluation phrase set (Phase D2.2, TASK 3).

A small, deterministic corpus of Russian medical-domain utterances for
measuring Qwen3-TTS 0.6B output quality (pronunciation / prosody) and
latency/resource behaviour. The purpose is to expose pronunciation problems
for AngioPlus terminology (AngioPlus, AngioPlus Core, PStation, μFR,
катетер, калибровка, ангиография, коронарная артерия, стеноз,
виртуальный стент), NOT to add any text-rewrite / pronunciation hacks.

Note: these are *synthesis* test utterances (text-to-speech). Several overlap
with the STT corpus (voice/stt_phrases.py) on purpose so the same AngioPlus
phrases can be measured both ways; the "answers" here are the target Russian
sentences to be spoken, not factual medical claims.

Fields
------
  id        stable short identifier (also used for a deterministic output basename)
  text      the exact text to synthesize (Russian, canonical spellings)
  category  functional grouping (for reporting)
  kind      short | normal | long  (utterance length bucket for latency analysis)
"""
from __future__ import annotations

from typing import List, TypedDict


class TtsPhrase(TypedDict):
    id: str
    text: str
    category: str
    kind: str  # short | normal | long


TTS_PHRASES: List[TtsPhrase] = [
    # --- required minimum set from the D2.2 brief ---------------------------
    {
        "id": "calibration",
        "text": "Как выполнить калибровку катетера?",
        "category": "catheter/calibration",
        "kind": "normal",
    },
    {
        "id": "calibration_before_analysis",
        "text": "Как выполнить калибровку катетера перед анализом?",
        "category": "catheter/calibration",
        "kind": "normal",
    },
    {
        "id": "mufr_shows",
        "text": "Что показывает μFR?",
        "category": "mufr",
        "kind": "short",
    },
    {
        "id": "mufr_calculate",
        "text": "Как рассчитать μFR?",
        "category": "mufr",
        "kind": "short",
    },
    {
        "id": "residual_mufr",
        "text": "Что означает остаточный μFR?",
        "category": "mufr",
        "kind": "normal",
    },
    {
        "id": "analyze_coronary",
        "text": "Выполните анализ коронарной артерии.",
        "category": "angiography",
        "kind": "normal",
    },
    {
        "id": "stenosis",
        "text": "Как оценить стеноз коронарной артерии?",
        "category": "angiography",
        "kind": "normal",
    },
    {
        "id": "virtual_stent",
        "text": "Что такое виртуальный стент?",
        "category": "mufr",
        "kind": "short",
    },
    {
        "id": "open_angioplus",
        "text": "Откройте AngioPlus Core.",
        "category": "software/interface",
        "kind": "short",
    },
    {
        "id": "open_pstation",
        "text": "Откройте AngioPlus Core на рабочей станции PStation.",
        "category": "software/interface",
        "kind": "normal",
    },
    # --- short phrases -------------------------------------------------------
    {
        "id": "short_yes",
        "text": "Да",
        "category": "short",
        "kind": "short",
    },
    {
        "id": "short_no",
        "text": "Нет",
        "category": "short",
        "kind": "short",
    },
    {
        "id": "short_calibrate_catheter",
        "text": "Калибруй катетер",
        "category": "short",
        "kind": "short",
    },
    {
        "id": "short_show_version",
        "text": "Покажи версию",
        "category": "short",
        "kind": "short",
    },
    # --- normal conversational sentences --------------------------------------
    {
        "id": "what_is_angioplus",
        "text": "Что такое AngioPlus Core?",
        "category": "software/interface",
        "kind": "normal",
    },
    {
        "id": "when_calibrate",
        "text": "Когда нужна калибровка катетера?",
        "category": "catheter/calibration",
        "kind": "normal",
    },
    {
        "id": "what_is_angiography",
        "text": "Что такое ангиография?",
        "category": "angiography",
        "kind": "normal",
    },
    {
        "id": "begin_calibration",
        "text": "Как начать калибровку?",
        "category": "catheter/calibration",
        "kind": "normal",
    },
    # --- longer AngioPlus answers ----------------------------------------------
    {
        "id": "long_software_version",
        "text": "Версию программного обеспечения можно посмотреть в разделе help и about на панели инструментов рабочей станции PStation. Откройте раздел справки и выберите пункт о программе.",
        "category": "long",
        "kind": "long",
    },
    {
        "id": "long_residual_mufr",
        "text": "Остаточный μFR показывает функциональную значимость стеноза коронарной артерии после процедуры. Сравните полученное значение с нормой, чтобы оценить результат.",
        "category": "long",
        "kind": "long",
    },
]


def get_ids() -> List[str]:
    return [p["id"] for p in TTS_PHRASES]


def phrase_text(id_or_text: str) -> str:
    """Return the canonical text for a phrase id, or the input if it is not a known id."""
    for p in TTS_PHRASES:
        if p["id"] == id_or_text:
            return p["text"]
    return id_or_text


def phrases_by_category(category: str) -> List[TtsPhrase]:
    return [p for p in TTS_PHRASES if p["category"] == category]


def short_phrases() -> List[TtsPhrase]:
    return [p for p in TTS_PHRASES if p["kind"] == "short"]


def normal_phrases() -> List[TtsPhrase]:
    return [p for p in TTS_PHRASES if p["kind"] == "normal"]


def long_phrases() -> List[TtsPhrase]:
    return [p for p in TTS_PHRASES if p["kind"] == "long"]


def safe_stem(text: str) -> str:
    """Deterministic, filesystem-safe filename stem from a phrase text."""
    stem = "".join(
        ch.lower() if ch.isalnum() or ch in "-_" else " "
        for ch in text.strip()
    )
    parts = [p for p in stem.split() if p]
    return "_".join(parts) if parts else "utterance"
