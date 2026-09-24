# -*- coding: utf-8 -*-
"""Slice 10 -- ProcedureCatalog cold-start selection contract (PURE).

This file pins the approved, TEMPORARY cold-start design:

  * the model is fixed verbatim as ``COLD_START_SELECTION_MODEL ==
    "mvp_lexical_bootstrap"`` and the module docstring states the six
    approved boundaries (lexical bootstrap, ONLY mapper, LLM never
    selects procedure_id, ...);
  * ``ProcedureCatalog.select`` takes ONLY ``text`` -- never a
    ``SemanticObservation`` / analyzer / provider object -- so procedure
    identity can never be model-supplied;
  * ``select`` deterministically maps an in-domain cold-start signal to
    exactly one procedure identity, and returns ``None`` for
    empty/unrelated text;
  * ``issue_match`` is the temporary lexical mechanism behind ``select``;
  * the module is pure (no LLM / network / DB / framework) and is the ONLY
    cold-start mapper in ``support/`` (anti-regression guard);
  * it is NOT wired into any runtime path or the package ``__init__``.

No DB, no cloud, no LLM: this is a plain unit test.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from support import procedure_catalog as pc
from support.observation import SemanticObservation
from support.procedure import Procedure, load_procedure, load_procedure_from_dict
from support.procedure_catalog import (
    COLD_START_SELECTION_MODEL,
    ProcedureCatalog,
    ProcedureSelection,
)


# A real, in-domain cold-start signal (a user *message*, not a step id).
_COLD_START_TEXT = (
    "После настройки не передаются данные DICOM на PACS, проверьте соединение"
)

_ROOT = Path(pc.__file__).resolve().parents[1]
_PROCEDURES_DIR = Path(pc.__file__).parent / "procedures"
_DICOM_YAML = _PROCEDURES_DIR / "dicom_pacs_connectivity_v1.yaml"


def _catalog() -> ProcedureCatalog:
    return ProcedureCatalog.from_directory()


# ================================================================
# 1. THE MODEL CONSTANT + APPROVED FRAMING
# ================================================================

def test_cold_start_selection_model_is_fixed_verbatim():
    assert COLD_START_SELECTION_MODEL == "mvp_lexical_bootstrap"
    assert pc.COLD_START_SELECTION_MODEL == "mvp_lexical_bootstrap"


@pytest.mark.parametrize(
    "phrase",
    [
        'COLD-START SELECTION MODEL = "mvp_lexical_bootstrap"',
        "# TEMPORARY.",
        "catalog.issue_match` is NOT the long-term semantic routing architecture",
        "It exists ONLY to bootstrap the first production Procedure without "
        "changing",
        "FROZEN SemanticObservation contract (Slice 4)",
        "ProcedureCatalog remains the ONLY component allowed to map a cold-start",
        "The LLM must NEVER output or select procedure_id.",
        "A FUTURE slice may introduce a closed semantic issue taxonomy produced by the",
        "Do NOT treat substring matching as the final selection mechanism.",
    ],
)
def test_module_docstring_states_the_approved_boundaries(phrase):
    assert phrase in (pc.__doc__ or ""), phrase


# ================================================================
# 2. ROUTING-POLICY GUARD -- select() can never take a model object
# ================================================================

def test_select_takes_only_text_never_an_analyzer_or_observation():
    params = list(inspect.signature(ProcedureCatalog.select).parameters)
    assert params == ["self", "text"], params
    assert not any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in inspect.signature(ProcedureCatalog.select).parameters.values()
    )


def test_select_parameter_is_not_a_model_supplied_object():
    annotation = inspect.signature(
        ProcedureCatalog.select
    ).parameters["text"].annotation
    # Annotated as ``str`` (may be the string 'str' or the class str).
    assert annotation in (str, "str")


# ================================================================
# 3. DETERMINISTIC COLD-START -> IDENTITY MAPPING
# ================================================================

def test_select_maps_cold_start_signal_to_procedure_identity():
    selection = _catalog().select(_COLD_START_TEXT)
    assert isinstance(selection, ProcedureSelection)
    assert selection.procedure_id == "DICOM_PACS_CONNECTIVITY"
    assert selection.version == 1
    assert selection.identity() == ("DICOM_PACS_CONNECTIVITY", 1)
    assert isinstance(selection.procedure, Procedure)


def test_select_delegates_to_the_temporary_issue_match():
    catalog = _catalog()
    assert catalog.select(_COLD_START_TEXT) == catalog.issue_match(
        _COLD_START_TEXT
    )


def test_selection_is_deterministic():
    assert _catalog().select(_COLD_START_TEXT) == _catalog().select(
        _COLD_START_TEXT
    )


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t", None, "Здравствуйте, как оформить заказ?", "порт 1108"],
)
def test_empty_or_unrelated_text_selects_nothing(text):
    assert _catalog().select(text) is None


def test_a_single_issue_token_is_not_enough():
    # "dicom" alone is one issue token: below the two-token floor.
    assert _catalog().select("dicom") is None


def test_matched_tokens_are_meaningful_issue_tokens():
    selection = _catalog().select(_COLD_START_TEXT)
    assert {"dicom", "pacs"} <= set(selection.matched_tokens)


# ================================================================
# 3b. AMBIGUITY -- >1 qualifying procedure selects NOTHING (no guessing)
# ================================================================

def _synthetic(procedure_id: str, issue: str) -> Procedure:
    """A minimal VALID single-step procedure; only ``issue`` varies."""
    return load_procedure_from_dict(
        {
            "procedure_id": procedure_id,
            "version": 1,
            "issue": issue,
            "entry_step": "STEP_ONE",
            "steps": [
                {
                    "id": "STEP_ONE",
                    "action": "Выполните первый шаг.",
                    "evidence_query": "первый шаг",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                }
            ],
            "terminators": ["ESCALATE", "RESOLVED"],
        }
    )


def test_one_qualifying_procedure_is_selected():
    matching = _synthetic("PROC_MATCH", "dicom pacs connection setup")
    unrelated = _synthetic("PROC_OTHER", "printing invoice export")
    catalog = ProcedureCatalog([matching, unrelated])
    selection = catalog.select("dicom pacs connection")
    assert selection is not None
    assert selection.procedure_id == "PROC_MATCH"
    assert selection.procedure == matching


def test_zero_qualifying_procedures_select_nothing():
    catalog = ProcedureCatalog(
        [
            _synthetic("PROC_A", "dicom pacs connection"),
            _synthetic("PROC_B", "printing invoice export"),
        ]
    )
    assert catalog.select("совершенно другая тема") is None


def test_two_qualifying_procedures_with_unequal_hits_select_nothing():
    """One procedure has strictly MORE lexical hits -- there is still NO
    best-match winner: the result is an explicit no-match."""
    rich = _synthetic("PROC_RICH", "dicom pacs connection setup")
    poor = _synthetic("PROC_POOR", "dicom pacs")
    text = "dicom pacs connection setup failed"
    # Sanity: each procedure qualifies on its OWN (checked first).
    assert ProcedureCatalog([rich]).select(text) is not None
    assert ProcedureCatalog([poor]).select(text) is not None
    # ...but the pair is ambiguous -> nothing is selected.
    assert ProcedureCatalog([rich, poor]).select(text) is None
    assert ProcedureCatalog([poor, rich]).select(text) is None


def test_two_qualifying_procedures_with_equal_hits_select_nothing():
    alpha = _synthetic("PROC_ALPHA", "dicom pacs alpha")
    beta = _synthetic("PROC_BETA", "dicom pacs beta")
    assert ProcedureCatalog([alpha]).select("dicom pacs") is not None
    assert ProcedureCatalog([beta]).select("dicom pacs") is not None
    assert ProcedureCatalog([alpha, beta]).select("dicom pacs") is None


def test_selection_is_independent_of_procedure_id_ordering():
    """Reversing the (procedure_id) order must not change the result."""
    high = _synthetic("ZZZ_PROC", "dicom pacs connection")
    low = _synthetic("AAA_PROC", "dicom pacs node")
    forward = ProcedureCatalog([high, low]).select("dicom pacs")
    reverse = ProcedureCatalog([low, high]).select("dicom pacs")
    assert forward is None
    assert reverse is None
    assert forward == reverse


# ================================================================
# 4. LOADING + VALIDATION
# ================================================================

def test_from_directory_loads_and_validates_the_production_procedure():
    catalog = _catalog()
    assert "DICOM_PACS_CONNECTIVITY" in catalog.procedure_ids
    selection = catalog.select(_COLD_START_TEXT)
    # The loaded graph is byte-for-byte the validated declarative file.
    assert selection.procedure == load_procedure(_DICOM_YAML)


def test_catalog_rejects_non_procedure_entries():
    with pytest.raises(TypeError):
        ProcedureCatalog([object()])  # type: ignore[list-item]


def test_catalog_rejects_duplicate_procedure_ids():
    proc = load_procedure(_DICOM_YAML)
    with pytest.raises(ValueError):
        ProcedureCatalog([proc, proc])


def test_from_directory_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProcedureCatalog.from_directory(tmp_path / "does_not_exist")


# ================================================================
# 5. ProcedureSelection VALUE OBJECT
# ================================================================

def test_selection_is_frozen():
    from dataclasses import FrozenInstanceError

    selection = _catalog().select(_COLD_START_TEXT)
    with pytest.raises(FrozenInstanceError):
        selection.procedure_id = "OTHER"  # type: ignore[misc]


# ================================================================
# 6. ANTI-REGRESSION GUARDS
# ================================================================

def _module_source() -> str:
    return Path(pc.__file__).read_text(encoding="utf-8")


def _module_imports() -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(_module_source())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_forbidden_imports():
    forbidden_roots = {
        "yandex",
        "openai",
        "requests",
        "httpx",
        "aiohttp",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
    }
    forbidden_modules = {
        "support.procedure_state",
        "support.observation",
        "support.router",
        "support.response_generator",
        "support.alice_analyzer",
        "support.knowledge_yandex",
        "support.db",
    }
    for name in _module_imports():
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden import: {name}"
        for mod in forbidden_modules:
            assert name != mod and not name.startswith(mod + "."), (
                f"forbidden import: {name}"
            )


def test_only_allowed_imports():
    allowed = {"__future__", "re", "dataclasses", "pathlib", "typing", "support"}
    roots = {name.split(".")[0] for name in _module_imports()}
    assert roots <= allowed, f"unexpected imports: {roots - allowed}"


def test_no_best_match_or_ranking_mechanism_in_source():
    """Anti-regression: qualifying procedures are never ranked/scored and
    procedure_id ordering is never used as a tie-break."""
    source = _module_source()
    assert "best_hits" not in source
    assert "key=lambda" not in source


def test_only_procedure_is_imported_from_support():
    support_mods = {
        n for n in _module_imports() if n == "support" or n.startswith("support.")
    }
    assert support_mods <= {"support.procedure"}, support_mods


def _identifiers() -> set[str]:
    """Identifier names actually USED in the module (strings/docstrings
    excluded, so approved prose can mention forbidden concepts)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(_module_source())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_the_catalog_module_never_talks_to_an_analyzer_or_observation():
    identifiers = _identifiers()
    for token in (
        "SemanticObservation",
        "ObservationScope",
        "ProblemStatus",
        "ReportedAction",
        "ProcedureState",
        "StepAttempt",
        "alice_analyzer",
        "analyze",
        "update_procedure_state",
        "route",
        "render_response",
        "authoritative_outcome",
    ):
        assert token not in identifiers, f"forbidden use: {token}"


def test_catalog_is_the_only_cold_start_mapper_in_support():
    """Anti-regression: no OTHER support module may map a cold-start signal
    to a procedure identity."""
    for path in Path(pc.__file__).parent.glob("*.py"):
        if path.name == "procedure_catalog.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "COLD_START_SELECTION_MODEL" not in text, path.name
        assert "issue_match" not in text, path.name


def test_semantic_observation_cannot_carry_procedure_identity():
    """The FROZEN Slice 4 contract has no place for a model-chosen
    procedure_id / procedure_version."""
    names = {f.name for f in fields(SemanticObservation)}
    assert "procedure_id" not in names
    assert "procedure_version" not in names


_RUNTIME_MODULES = (
    Path("yandex/chat/server.py"),
    Path("yandex/chat/alice_support_agent.py"),
    Path("yandex/realtime/web/server.py"),
)


def test_not_wired_into_any_runtime_path():
    for rel in _RUNTIME_MODULES:
        path = _ROOT / rel
        if not path.exists():
            continue
        assert "procedure_catalog" not in path.read_text(encoding="utf-8"), rel


def test_not_exported_from_package_init():
    init_path = Path(pc.__file__).parent / "__init__.py"
    if not init_path.exists():
        pytest.skip("support/__init__.py not present")
    assert "procedure_catalog" not in init_path.read_text(encoding="utf-8")


def test_public_api_surface():
    assert set(pc.__all__) == {
        "COLD_START_SELECTION_MODEL",
        "ProcedureSelection",
        "ProcedureCatalog",
    }
