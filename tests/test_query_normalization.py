# -*- coding: utf-8 -*-
from app.query_normalization import normalize_user_query


def test_pulse_musicl_variant_to_canonical():
    assert normalize_user_query("Кто такие Пульс Мьюзикл?") == "Кто такие Pulse Medical?"


def test_pulse_medical_variant_to_canonical():
    assert normalize_user_query("Кто такие Пульс Медикал?") == "Кто такие Pulse Medical?"


def test_pulse_musicl_lowercased_to_canonical():
    assert normalize_user_query("пьюлс мьюзикл") == "Pulse Medical"


def test_angio_plus_core_variants():
    assert normalize_user_query("Ангио плюс кор") == "AngioPlus Core"
    assert normalize_user_query("Ангиоплюс кор") == "AngioPlus Core"


def test_pstation_and_dicom():
    assert normalize_user_query("Пи стейшн") == "PStation"
    assert normalize_user_query("Диком") == "DICOM"


def test_word_gap_fix_ustanovit():
    assert normalize_user_query("А как его у становить?") == "А как его установить?"


def test_canonical_already_is_idempotent():
    # A canonical form must not be corrupted.
    assert normalize_user_query("Pulse Medical") == "Pulse Medical"
    assert normalize_user_query("AngioPlus Core") == "AngioPlus Core"


def test_unknown_words_unchanged():
    assert normalize_user_query("Расскажи про валидацию устройства") == (
        "Расскажи про валидацию устройства"
    )


def test_whitespace_collapse_and_trim():
    assert normalize_user_query("  Пульс   Мьюзикл  ") == "Pulse Medical"


def test_empty_input():
    assert normalize_user_query("") == ""
    assert normalize_user_query(None) is None
