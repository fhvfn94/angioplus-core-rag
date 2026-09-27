# -*- coding: utf-8 -*-
"""Slice 11B.1b.4a -- offline proofs for the executor's precomputed path.

NO DB, NO network: an in-memory store + a fake session factory exercise the
SHARED READ -> PROCESS -> CAS WRITE lifecycle of
:class:`~support.turn_executor.SupportTurnExecutor` and prove that the new
``execute_persisted_turn_from_observation`` entry point:

  * consumes a precomputed observation -- the analyzer is unreachable;
  * writes through the SAME single CAS writer (one ``save_cas`` per turn);
  * is durable-equivalent to the legacy path for the same observation.

The real PostgreSQL CAS / conflict / atomicity behaviour stays covered by the
DB-marked ``tests/test_turn_executor.py``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Optional

import pytest

import support.orchestrator as orchestrator
from support.observation import (
    ObservationScope,
    RequestKind,
    SemanticObservation,
)
from support.procedure import load_procedure
from support.procedure_state import ProcedureState
from support.procedure_state_store import (
    CasResult,
    PersistedProcedureState,
    ProcedureStateConflict,
)
from support.turn_executor import SupportTurnExecutor


pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "procedures"


# ============================================================
# IN-MEMORY DOUBLE (one DB object shared by both sessions)
# ============================================================

class _FakeSession:
    def __init__(self, db: "_InMemoryDb") -> None:
        self._db = db

    async def __aenter__(self):
        self._db.sessions_opened += 1
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


class _FakeStore:
    """Minimal CAS store over an in-memory row map."""

    def __init__(self, db: "_InMemoryDb") -> None:
        self._db = db

    async def read(self, conversation_id: str):
        return self._db.rows.get(conversation_id)

    async def save_cas(
        self, conversation_id: str, state: ProcedureState, *, expected_revision
    ) -> CasResult:
        self._db.save_cas_calls += 1
        current = self._db.rows.get(conversation_id)
        current_rev = current.revision if current is not None else 0
        if expected_revision is not None and expected_revision != current_rev:
            raise ProcedureStateConflict(
                f"stale revision for {conversation_id!r}"
            )
        new_rev = current_rev + 1
        self._db.rows[conversation_id] = PersistedProcedureState(
            state=state, revision=new_rev
        )
        return CasResult(revision=new_rev, applied=True, state=state)


class _InMemoryDb:
    def __init__(self) -> None:
        self.rows: dict[str, PersistedProcedureState] = {}
        self.save_cas_calls = 0
        self.sessions_opened = 0


def _executor(db: _InMemoryDb) -> SupportTurnExecutor:
    return SupportTurnExecutor(
        session_factory=lambda: _FakeSession(db),
        store_factory=lambda session: _FakeStore(db),
    )


def _black_screen():
    return load_procedure(FIXTURES / "black_screen.yaml")


def _obs_json() -> str:
    return json.dumps(
        {
            "scope": "in_scope",
            "reported_actions": [],
            "problem_status": None,
            "human_requested": False,
            "clarification_facts": [],
            "new_issue": None,
            "request_kind": "problem",
        }
    )


def _observation() -> SemanticObservation:
    return SemanticObservation(
        scope=ObservationScope.IN_SCOPE,
        reported_actions=(),
        problem_status=None,
        human_requested=False,
        clarification_facts=(),
        new_issue=None,
        request_kind=RequestKind.PROBLEM,
    )


# ============================================================
# CONTRACT
# ============================================================

async def test_observation_path_persists_via_the_single_writer(monkeypatch):
    procedure = _black_screen()
    db = _InMemoryDb()

    def _boom(*args, **kwargs):
        raise AssertionError("analyzer must NOT run on the precomputed path")

    monkeypatch.setattr(orchestrator, "analyze", _boom)

    result = await _executor(db).execute_persisted_turn_from_observation(
        conversation_id="c1",
        observation=_observation(),
        procedure=procedure,
    )

    assert result.revision == 1
    assert result.turn.decision.next_step == "CHECK_POWER"
    assert db.save_cas_calls == 1
    assert db.rows["c1"].revision == 1
    assert db.rows["c1"].state == result.state


async def test_both_entry_points_share_one_lifecycle_and_writer(monkeypatch):
    procedure = _black_screen()

    # Legacy path: analyzer runs once, exactly ONE CAS write, two sessions.
    db_legacy = _InMemoryDb()
    calls = {"n": 0}

    def analyzer(prompt, history):
        calls["n"] += 1
        return _obs_json()

    await _executor(db_legacy).execute_persisted_turn(
        conversation_id="c1",
        message="m",
        procedure=procedure,
        analyzer_provider=analyzer,
    )
    assert calls["n"] == 1
    assert db_legacy.save_cas_calls == 1
    assert db_legacy.sessions_opened == 2

    # Precomputed path: NO analyzer, SAME lifecycle, SAME one CAS write.
    db_pre = _InMemoryDb()

    def _boom(*args, **kwargs):
        raise AssertionError("analyzer must NOT run")

    monkeypatch.setattr(orchestrator, "analyze", _boom)

    await _executor(db_pre).execute_persisted_turn_from_observation(
        conversation_id="c1",
        observation=_observation(),
        procedure=procedure,
    )
    assert db_pre.save_cas_calls == 1
    assert db_pre.sessions_opened == 2


async def test_observation_path_is_durable_equivalent_to_legacy():
    procedure = _black_screen()

    legacy = await _executor(_InMemoryDb()).execute_persisted_turn(
        conversation_id="c1",
        message="m",
        procedure=procedure,
        analyzer_provider=lambda prompt, history: _obs_json(),
    )
    precomputed = await _executor(
        _InMemoryDb()
    ).execute_persisted_turn_from_observation(
        conversation_id="c1",
        observation=_observation(),
        procedure=procedure,
    )

    assert precomputed.state == legacy.state
    assert precomputed.turn.decision == legacy.turn.decision
    assert precomputed.turn.assistant_text == legacy.turn.assistant_text


async def test_observation_entry_has_no_analyzer_argument():
    params = set(
        inspect.signature(
            SupportTurnExecutor.execute_persisted_turn_from_observation
        ).parameters
    )
    assert "analyzer_provider" not in params
    assert "message" not in params
