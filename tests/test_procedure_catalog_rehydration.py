# -*- coding: utf-8 -*-
"""Slice 11A.1 -- exact Procedure rehydration (ProcedureCatalog.get).

Pins the identity-based lookup ADDED to the frozen Slice 10 catalog:

  * ``get(procedure_id, version)`` requires BOTH arguments and matches the
    pair EXACTLY;
  * it is the SAME validated :class:`~support.procedure.Procedure` the
    catalog holds (no re-parse);
  * unknown id -> ``None``; wrong/unknown version -> ``None``;
  * there is NO fallback, NO "latest version", NO silent upgrade and NO
    lexical matching -- it can never be used as a second selector;
  * cold-start selection (``select`` / ``issue_match`` /
    ``COLD_START_SELECTION_MODEL``) is unchanged.

No DB, no cloud, no LLM: this is a plain unit test.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from support import procedure_catalog as pc
from support.procedure import Procedure, load_procedure, load_procedure_from_dict
from support.procedure_catalog import ProcedureCatalog


_PROCEDURES_DIR = Path(pc.__file__).parent / "procedures"
_DICOM_YAML = _PROCEDURES_DIR / "dicom_pacs_connectivity_v1.yaml"

_COLD_START_TEXT = (
    "После настройки не передаются данные DICOM на PACS, проверьте соединение"
)


def _synthetic(
    procedure_id: str, version: int, action: str = "Выполните шаг."
) -> Procedure:
    """A minimal VALID single-step procedure; only version/action vary."""
    return load_procedure_from_dict(
        {
            "procedure_id": procedure_id,
            "version": version,
            "issue": "dicom pacs connection",
            "entry_step": "STEP_ONE",
            "steps": [
                {
                    "id": "STEP_ONE",
                    "action": action,
                    "evidence_query": "шаг",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                }
            ],
            "terminators": ["ESCALATE", "RESOLVED"],
        }
    )


# ================================================================
# 1. EXACT IDENTITY HIT -> THE SAME VALIDATED PROCEDURE
# ================================================================

def test_get_returns_the_exact_validated_procedure():
    catalog = ProcedureCatalog.from_directory()
    proc = catalog.get("DICOM_PACS_CONNECTIVITY", 1)
    assert proc is not None
    assert isinstance(proc, Procedure)
    assert proc.procedure_id == "DICOM_PACS_CONNECTIVITY"
    assert proc.version == 1
    # The graph is byte-for-byte the validated declarative file (no re-parse).
    assert proc == load_procedure(_DICOM_YAML)


def test_get_returns_the_same_object_the_catalog_holds():
    catalog = ProcedureCatalog.from_directory()
    # `select` exposes the stored graph too -- `get` must return that very
    # object, so identity lookup can never fabricate a divergent graph.
    selection = catalog.select(_COLD_START_TEXT)
    assert selection is not None
    assert catalog.get("DICOM_PACS_CONNECTIVITY", 1) is selection.procedure


# ================================================================
# 2. UNKNOWN ID / WRONG VERSION -> None (NO FALLBACK)
# ================================================================

def test_get_unknown_id_returns_none():
    catalog = ProcedureCatalog.from_directory()
    assert catalog.get("NO_SUCH_PROCEDURE", 1) is None


@pytest.mark.parametrize("version", [0, 2, 99, -1])
def test_get_wrong_version_returns_none(version):
    catalog = ProcedureCatalog.from_directory()
    assert catalog.get("DICOM_PACS_CONNECTIVITY", version) is None


def test_get_is_exact_and_never_normalizes_identity():
    catalog = ProcedureCatalog.from_directory()
    assert catalog.get("dicom_pacs_connectivity", 1) is None  # case-sensitive
    assert catalog.get("DICOM_PACS_CONNECTIVITY ", 1) is None  # no trimming
    assert catalog.get(" DICOM_PACS_CONNECTIVITY", 1) is None


# ================================================================
# 3. MULTIPLE VERSIONS -- EXACT REHYDRATION, NO "LATEST"
# ================================================================

def test_two_versions_of_the_same_procedure_are_rehydrated_exactly():
    v1 = _synthetic("MULTI_VERSION", 1, "Шаг версии один.")
    v2 = _synthetic("MULTI_VERSION", 2, "Шаг версии два.")
    catalog = ProcedureCatalog([v1, v2])

    got1 = catalog.get("MULTI_VERSION", 1)
    got2 = catalog.get("MULTI_VERSION", 2)
    assert got1 is v1
    assert got2 is v2
    # No cross-version leakage.
    assert got1.steps["STEP_ONE"].action == "Шаг версии один."
    assert got2.steps["STEP_ONE"].action == "Шаг версии два."


def test_no_implicit_latest_version_when_a_higher_version_exists():
    v1 = _synthetic("MULTI_VERSION", 1)
    v2 = _synthetic("MULTI_VERSION", 2)
    catalog = ProcedureCatalog([v1, v2])
    # v1 present -> returned; v2 present -> returned; v3 absent -> None
    # (never silently resolved to the highest stored version).
    assert catalog.get("MULTI_VERSION", 1) is v1
    assert catalog.get("MULTI_VERSION", 2) is v2
    assert catalog.get("MULTI_VERSION", 3) is None


def test_no_silent_upgrade_when_only_a_newer_version_is_present():
    v2 = _synthetic("MULTI_VERSION", 2)
    catalog = ProcedureCatalog([v2])
    # An older request must NOT be upgraded to the only stored version.
    assert catalog.get("MULTI_VERSION", 1) is None
    assert catalog.get("MULTI_VERSION", 3) is None


def test_requesting_a_version_never_substitutes_another_version():
    v1 = _synthetic("MULTI_VERSION", 1, "Шаг версии один.")
    v2 = _synthetic("MULTI_VERSION", 2, "Шаг версии два.")
    catalog = ProcedureCatalog([v1, v2])
    assert catalog.get("MULTI_VERSION", 1).steps["STEP_ONE"].action != (
        catalog.get("MULTI_VERSION", 2).steps["STEP_ONE"].action
    )


# ================================================================
# 4. SIGNATURE / SHAPE -- BOTH ARGS REQUIRED, NO CLEVERNESS
# ================================================================

def test_get_requires_both_procedure_id_and_version():
    params = inspect.signature(ProcedureCatalog.get).parameters
    assert list(params) == ["self", "procedure_id", "version"]
    assert params["procedure_id"].default is inspect.Parameter.empty
    assert params["version"].default is inspect.Parameter.empty
    assert not any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params.values()
    )


def _get_method_identifiers() -> set[str]:
    """Identifier names ACTUALLY used in ``get`` (docstrings excluded)."""
    source = textwrap.dedent(inspect.getsource(ProcedureCatalog.get))
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_get_uses_no_lexical_or_analyzer_machinery():
    identifiers = _get_method_identifiers()
    for token in (
        "issue_match",
        "_issue_tokens",
        "_qualifying",
        "analyze",
        "lower",
        "casefold",
        "sorted",
        "max",
        "min",
    ):
        assert token not in identifiers, f"forbidden use in get(): {token}"
    # ...and it really does compare the exact identity pair.
    assert {"procedure_id", "version"} <= identifiers


# ================================================================
# 5. DUPLICATE HANDLING -- EXACT IDENTITY ONLY
# ================================================================

def test_exact_duplicate_identity_is_rejected():
    proc = load_procedure(_DICOM_YAML)
    with pytest.raises(ValueError):
        ProcedureCatalog([proc, proc])


def test_distinct_versions_of_the_same_id_coexist():
    v1 = _synthetic("MULTI_VERSION", 1)
    v2 = _synthetic("MULTI_VERSION", 2)
    catalog = ProcedureCatalog([v1, v2])
    assert len(catalog) == 2
    assert catalog.procedure_ids == frozenset({"MULTI_VERSION"})


# ================================================================
# 6. COLD-START SEMANTICS UNCHANGED
# ================================================================

def test_cold_start_selection_model_is_unchanged():
    assert pc.COLD_START_SELECTION_MODEL == "mvp_lexical_bootstrap"


def test_select_still_maps_cold_start_and_get_is_not_a_selector():
    catalog = ProcedureCatalog.from_directory()
    selection = catalog.select(_COLD_START_TEXT)
    assert selection is not None
    assert selection.identity() == ("DICOM_PACS_CONNECTIVITY", 1)
    # `select` still delegates to the temporary lexical mechanism.
    assert catalog.select(_COLD_START_TEXT) == catalog.issue_match(
        _COLD_START_TEXT
    )
    # `get` needs an identity and cannot stand in for `select`.
    assert catalog.select("Здравствуйте, как оформить заказ?") is None


def test_public_api_surface_is_unchanged_by_get():
    assert set(pc.__all__) == {
        "COLD_START_SELECTION_MODEL",
        "ProcedureSelection",
        "ProcedureCatalog",
    }
