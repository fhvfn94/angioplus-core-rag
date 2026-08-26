# -*- coding: utf-8 -*-
"""AngioPlus Russian STT evaluation phrase set (Phase D1.5, TASK 3).

A small, deterministic benchmark corpus for measuring faster-whisper accuracy
and latency for the AngioPlus Russian voice assistant. Each entry is a
*vocabulary/STT test utterance*, NOT a factual medical claim and NOT an answer.

Format choice
-------------
A Python module (not JSON) because:
  - it imports directly in pytest and in the benchmark helper with zero loader
    code / no path handling;
  - inline per-phrase notes keep the corpus self-explanatory for reviewers;
  - it is easy for a human to extend without a JSON edit round-trip.
JSON would be preferred once a large, externally-authored corpus is needed.

Fields
------
  id       stable short identifier
  text     the exact expected transcript (Russian, canonical spellings)
  category functional grouping (for reporting)
  is_min   True for the required minimum set listed in the D1.5 brief
"""
from __future__ import annotations

from typing import List, TypedDict


class SttPhrase(TypedDict):
    id: str
    text: str
    category: str
    is_min: bool


STT_PHRASES: List[SttPhrase] = [
    # --- required minimum set from the D1.5 brief (is_min=True) -------------
    {
        "id": "min_calibration",
        "text": "Как выполнить калибровку катетера?",
        "category": "catheter/calibration",
        "is_min": True,
    },
    {
        "id": "min_mufr",
        "text": "Что такое μFR?",
        "category": "mufr",
        "is_min": True,
    },
    {
        "id": "min_residual_mufr",
        "text": "Как измерить остаточный μFR?",
        "category": "mufr",
        "is_min": True,
    },
    {
        "id": "min_software_version",
        "text": "Где посмотреть версию программного обеспечения?",
        "category": "software/interface",
        "is_min": True,
    },
    {
        "id": "min_no_reaction",
        "text": "Что делать, если программа не реагирует на действия пользователя?",
        "category": "software/interface",
        "is_min": True,
    },
    # --- everyday AngioPlus questions ---------------------------------------
    {
        "id": "what_is_angioplus",
        "text": "Что такое AngioPlus Core?",
        "category": "everyday",
        "is_min": False,
    },
    {
        "id": "how_start_calibration",
        "text": "Как начать калибровку?",
        "category": "catheter/calibration",
        "is_min": False,
    },
    # --- catheter / calibration terminology ----------------------------------
    {
        "id": "catheter_what_is",
        "text": "Что такое катетер?",
        "category": "catheter/calibration",
        "is_min": False,
    },
    {
        "id": "calibrate_catheter",
        "text": "Когда нужна калибровка катетера?",
        "category": "catheter/calibration",
        "is_min": False,
    },
    # --- angiography terminology ---------------------------------------------
    {
        "id": "what_is_angiography",
        "text": "Что такое ангиография?",
        "category": "angiography",
        "is_min": False,
    },
    {
        "id": "coronary_artery_stenosis",
        "text": "Что такое стеноз коронарной артерии?",
        "category": "angiography",
        "is_min": False,
    },
    {
        "id": "coronary_artery",
        "text": "Как называется артерия?",
        "category": "angiography",
        "is_min": False,
    },
    # --- μFR terminology -----------------------------------------------------
    {
        "id": "residual_mufr_measure",
        "text": "Что показывает остаточный μFR?",
        "category": "mufr",
        "is_min": False,
    },
    {
        "id": "virtual_stent",
        "text": "Что такое виртуальный стент?",
        "category": "mufr",
        "is_min": False,
    },
    # --- software / interface terminology -------------------------------------
    {
        "id": "pstation_help",
        "text": "Где в PStation вкладка справка?",
        "category": "software/interface",
        "is_min": False,
    },
    {
        "id": "program_freezes",
        "text": "Программа зависла, что мне делать?",
        "category": "software/interface",
        "is_min": False,
    },
    {
        "id": "software_version_where",
        "text": "Версия программы где указана?",
        "category": "software/interface",
        "is_min": False,
    },
    # --- short phrases --------------------------------------------------------
    {
        "id": "short_yes",
        "text": "Да",
        "category": "short",
        "is_min": False,
    },
    {
        "id": "short_calibrate",
        "text": "Калибруй катетер",
        "category": "short",
        "is_min": False,
    },
    {
        "id": "short_show_version",
        "text": "Покажи версию",
        "category": "short",
        "is_min": False,
    },
]


def get_minimum_phrases() -> List[SttPhrase]:
    """The required minimum set (at least the five listed in the D1.5 brief)."""
    return [p for p in STT_PHRASES if p["is_min"]]


def phrases_by_category(category: str) -> List[SttPhrase]:
    return [p for p in STT_PHRASES if p["category"] == category]


def phrase_texts() -> List[str]:
    return [p["text"] for p in STT_PHRASES]
