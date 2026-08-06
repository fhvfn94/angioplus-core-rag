# -*- coding: utf-8 -*-
from app.domain_terms import distinct_entities, extract_entities
from app.followup import (
    is_follow_up_candidate,
    rewrite_followup_question,
)


# --- entity extraction: position-based, not dictionary order ---

def test_entity_selected_by_position_in_text():
    # "AngioPlus Core" appears after "Pulse Medical" in the string, so it must
    # be the last positioned match regardless of dictionary order.
    q = "Расскажи про Pulse Medical и AngioPlus Core"
    entities = distinct_entities(q)
    assert entities == ["Pulse Medical", "AngioPlus Core"]


def test_extract_entities_sorted_by_position():
    q = "Про AngioPlus Core и Pulse Medical"
    matches = extract_entities(q)
    positions = [m.start for m in matches]
    assert positions == sorted(positions)
    assert [m.entity for m in matches] == ["AngioPlus Core", "Pulse Medical"]


def test_repeated_entity_is_not_ambiguity():
    q = "AngioPlus Core AngioPlus Core"
    entities = distinct_entities(q)
    assert entities == ["AngioPlus Core"]
    assert len(entities) == 1


def test_two_distinct_entities_is_ambiguity():
    q = "Расскажи про AngioPlus Core и PStation"
    entities = distinct_entities(q)
    assert len(entities) == 2


# --- canonical required target case ---

def test_required_follow_up_rewrite():
    last = "Какие системные требования у AngioPlus Core?"
    current = "А как его у становить?"  # raw user input with STT gap
    normalized = __import__("app.query_normalization", fromlist=["normalize_user_query"]).normalize_user_query(current)
    assert normalized == "А как его установить?"
    standalone, used = rewrite_followup_question(normalized, last)
    assert used is True
    assert standalone == "Как установить AngioPlus Core?"
    # Must never produce the ungrammatical word-permutation form.
    assert "AngioPlus Core установить" not in standalone


# --- standalone questions: no rewrite ---

def test_what_is_dicom_no_rewrite():
    last = "Какие системные требования у AngioPlus Core?"
    current = "Что такое DICOM?"
    assert is_follow_up_candidate(current, last) is False


def test_how_install_angio_no_rewrite():
    last = "Какие системные требования у AngioPlus Core?"
    current = "Как установить AngioPlus Core?"
    standalone, used = rewrite_followup_question(current, last)
    assert used is False
    assert standalone == current


def test_short_question_without_history_no_rewrite():
    standalone, used = rewrite_followup_question("А как его настроить?", None)
    assert used is False
    assert standalone == "А как его настроить?"


# --- continuation-only ---

def test_and_next_is_follow_up_candidate():
    last = "Как установить AngioPlus Core?"
    current = "И дальше?"
    assert is_follow_up_candidate(current, last) is True
    standalone, used = rewrite_followup_question(current, last)
    # "И дальше?" is not a supported template -> keep normalized original,
    # do not guess.
    assert used is False
    assert standalone == current


# --- ambiguity disables rewrite ---

def test_two_entities_ambiguity_no_rewrite():
    last = "Расскажи про AngioPlus Core и PStation"
    current = "А как его настроить?"
    standalone, used = rewrite_followup_question(current, last)
    assert used is False
    assert standalone == current  # normalized original, ambiguity=true


# --- unsupported construction -> keep original ---

def test_unsupported_construction_not_rewritten():
    last = "Какие системные требования у AngioPlus Core?"
    current = "А его установить как?"
    # Not a supported template -> keep normalized original.
    standalone, used = rewrite_followup_question(current, last)
    assert used is False
    assert standalone == current


def test_rewrite_where_template():
    last = "Расскажи про PStation"
    current = "А где её настроить?"
    standalone, used = rewrite_followup_question(current, last)
    assert used is True
    assert standalone == "Где настроить PStation?"


def test_rewrite_what_is_this():
    last = "Расскажи про DICOM"
    current = "А что это?"
    standalone, used = rewrite_followup_question(current, last)
    assert used is True
    assert standalone == "Что такое DICOM?"
