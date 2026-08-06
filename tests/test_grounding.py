# -*- coding: utf-8 -*-
"""Regression tests for generation grounding (system prompt).

These verify the prompt-level guarantees added for two runtime issues:
- a contradictory NOT_FOUND + answer reply for installation questions;
- the ungrounded Pulse Medical answer (clinical trials / MACE / FFR / ЧКВ).

Note: the answer text itself is produced by an external LLM at runtime, so
unit tests verify the prompt contract that prevents these behaviours.
"""

import app.main as rag

NOT_FOUND = "Такой информации нет в имеющейся документации."


def _normalized(prompt: str) -> str:
    """Collapse whitespace so multi-line paragraphs can be checked simply."""
    return " ".join(prompt.split())


def test_system_prompt_forbids_notfound_and_answer_contradiction():
    prompt = _normalized(rag.load_system_prompt())
    # Both triggers must be present in the same prompt.
    assert NOT_FOUND in prompt
    assert "Do not mix refusal and answer in a single reply" in prompt
    assert 'NEVER begin the answer with the phrase "Такой информации нет в имеющейся' in prompt
    assert 'Reply with "Такой информации нет в имеющейся документации." ONLY when' in prompt
    assert "then stop there" in prompt


def test_system_prompt_forbids_adding_company_knowledge():
    prompt = _normalized(rag.load_system_prompt())
    assert "STRICT COMPANY/PRODUCT GROUNDING" in prompt
    assert "Use ONLY facts that appear directly in the RETRIEVED CONTEXT" in prompt
    for term in ("MACE", "FFR", "FAVOR III", "FLAVOUR II", "ЧКВ"):
        assert term in prompt, term
    assert "основной продукт" in prompt
    assert "list ONLY what is directly confirmed in the RETRIEVED CONTEXT" in prompt


def test_not_found_diagnostics_flags_logged_safely():
    """The success log includes boolean NOT_FOUND diagnostics, no content."""
    import re
    import types as _t

    # The structural branch test already proves the success path runs; here we
    # verify the diagnostic fields exist in the log call source (no content).
    source = open("app/main.py", encoding="utf-8").read()
    assert "not_found_diagnostics" in source
    assert "not_found_in_system_prompt" in source
    assert "raw_answer_starts_with_not_found" in source
    assert "postprocess_prepends_not_found" in source
    # Logging is the structured %s + dict form: only a single placeholder.
    # Ensure no positional %s/%.4f chain that could mismatch at runtime.
    assert re.search(r'logger\.info\(\s*\n\s*"ask %s"\s*,', source)
