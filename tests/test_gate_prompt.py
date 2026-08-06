# -*- coding: utf-8 -*-
"""Structural verification of the identification gate rule across providers.

The three LLM providers share a direct-answer gate prompt. Real LLM calls
cannot be made in unit tests, so this checks that the identification rule and
its distinguishing content fragments are present in every provider source.

Assertions use single contiguous substrings (they also appear inside Russian
examples), avoiding the split across adjacent Python string literals.
"""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]

PROVIDER_FILES = [
    _ROOT / "app" / "llm" / "deepseek_provider.py",
    _ROOT / "app" / "llm" / "gemini_provider.py",
    _ROOT / "app" / "llm" / "gigachat_provider.py",
]

# Unambiguous contiguous fragments of the identification rule/limits.
_RULE_FRAGMENTS = [
    "X is a company/manufacturer/developer",
    "the product or trademark belongs to X",
    "official full company name of X is stated",
    "role of X with respect to AngioPlus Core",
    "do NOT require a full corporate",
    "history, or detailed business description",
    "merge separate sentences into claims the context does",
]

# Distinguishing content of the three required gate cases.
_CASE_FRAGMENTS = [
    "Pulse Medical",
    "техническую поддержку",
    "представителю",
    "цифровой передачи медицинской информации",
]

def test_identification_rule_fragments_in_all_providers():
    for path in PROVIDER_FILES:
        source = path.read_text(encoding="utf-8")
        for frag in _RULE_FRAGMENTS:
            assert frag in source, (path.name, frag)


def test_required_gate_case_content_in_all_providers():
    for path in PROVIDER_FILES:
        source = path.read_text(encoding="utf-8")
        for frag in _CASE_FRAGMENTS:
            assert frag in source, (path.name, frag)
        # Both answer outcomes must be configured in the examples.
        assert "direct_answer" in source
        assert "true" in source and "false" in source

