# -*- coding: utf-8 -*-
"""Slice 11B.1b.4c -- offline proofs for the family-dispatch boundary.

PURE: NO network, NO credentials, NO database, NO event loop blocking. Fakes
and recorders exercise the behavioral invariants of
:class:`~support.turn_dispatch.SupportTurnDispatcher`:

  * the analyzer runs EXACTLY ONCE per dispatch;
  * PROCEDURE_SUPPORT delegates with the SAME observation (precomputed path);
  * KNOWLEDGE_QA never touches the procedure executor;
  * CLARIFICATION touches neither the executor nor the knowledge engine;
  * exactly ONE family executes;
  * a ``ProcedureStateConflict`` propagates unchanged (no retry);
  * knowledge statuses are preserved intact;
  * a PROBLEM turn with no procedure is a typed ``NoProcedureDispatch``;
  * the module is not wired into any runtime path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from support.alice_analyzer import build_allowed_steps, parse_semantic_output
from support.clarification import clarify
from support.knowledge import KnowledgeItem
from support.knowledge_qa import KnowledgeQAStatus, KnowledgeQAResult
from support.procedure import load_procedure_from_dict
from support.procedure_state import ProcedureState
from support.procedure_state_store import ProcedureStateConflict
from support.turn_dispatch import (
    ExecutedDispatch,
    NoProcedureDispatch,
    SupportTurnDispatcher,
)
from support.turn_executor import PersistedTurnResult
from support.turn_family import TurnFamily


# ============================================================
# DOUBLES (no network, no DB)
# ============================================================

def _obs_json(request_kind: str) -> str:
    return json.dumps(
        {
            "scope": "in_scope",
            "reported_actions": [],
            "problem_status": None,
            "human_requested": False,
            "clarification_facts": [],
            "new_issue": None,
            "request_kind": request_kind,
        }
    )


class _CountingAnalyzer:
    """Frozen Slice 4 ``AnalyzerProvider`` double that counts its own calls."""

    def __init__(self, request_kind: str = "problem", error=None) -> None:
        self._request_kind = request_kind
        self._error = error
        self.calls = 0

    def __call__(self, prompt, history=()):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return _obs_json(self._request_kind)


class _RecordingExecutor:
    """``SupportTurnExecutor`` double: records the ONE delegated precomputed
    call and the observation it received."""

    def __init__(self, result=None, error=None) -> None:
        self._result = result
        self._error = error
        self.calls: list[dict] = []
        self.observations: list = []

    async def execute_persisted_turn_from_observation(
        self,
        *,
        conversation_id,
        observation,
        procedure,
        knowledge_backend=None,
        response_provider=None,
    ):
        self.calls.append(
            {"conversation_id": conversation_id, "procedure": procedure}
        )
        self.observations.append(observation)
        if self._error is not None:
            raise self._error
        return self._result if self._result is not None else _persisted()

    async def execute_persisted_turn(self, **kwargs):
        raise AssertionError(
            "the dispatcher must use the precomputed-observation path only"
        )


class _FakeBackend:
    """``KnowledgeBackend`` double."""

    def __init__(self, items=(), error=None) -> None:
        self._items = items
        self._error = error
        self.calls = 0

    def search(self, query, *, max_num_results):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._items


class _FakeQA:
    """``KnowledgeQAProvider`` double."""

    def __init__(self, raw: str = '{"answer": "Grounded answer."}') -> None:
        self._raw = raw
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self._raw


# ============================================================
# HELPERS
# ============================================================

def _persisted(text: str = "Step text.", revision: int = 1):
    return PersistedTurnResult(
        conversation_id="c1",
        turn=SimpleNamespace(assistant_text=text),
        state=ProcedureState(),
        revision=revision,
    )


def _procedure():
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 1,
            "issue": "black screen",
            "entry_step": "STEP_A",
            "terminators": ["ESCALATE", "RESOLVED"],
            "steps": [
                {
                    "id": "STEP_A",
                    "action": "Confirm the power LED is lit.",
                    "evidence_query": "angioplus black screen power led",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                }
            ],
        }
    )


def _dispatcher(analyzer, *, executor=None, backend=None, qa=None):
    return SupportTurnDispatcher(
        analyzer_provider=analyzer,
        procedure_executor=executor if executor is not None else _RecordingExecutor(),
        knowledge_backend=backend if backend is not None else _FakeBackend(),
        knowledge_qa_provider=qa if qa is not None else _FakeQA(),
    )


# ============================================================
# SINGLE ANALYSIS
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["problem", "information", "unclear"])
async def test_analyzer_runs_exactly_once_per_dispatch(kind):
    analyzer = _CountingAnalyzer(kind)

    await _dispatcher(analyzer).dispatch(
        conversation_id="c1", message="m", procedure=_procedure()
    )

    assert analyzer.calls == 1


@pytest.mark.asyncio
async def test_analyzer_fail_safe_routes_to_clarification():
    analyzer = _CountingAnalyzer(error=RuntimeError("provider down"))

    result = await _dispatcher(analyzer).dispatch(
        conversation_id="c1", message="m", procedure=_procedure()
    )

    assert isinstance(result, ExecutedDispatch)
    assert result.family is TurnFamily.CLARIFICATION
    assert analyzer.calls == 1  # attempted exactly once, fail-safe


# ============================================================
# PROCEDURE_SUPPORT
# ============================================================

@pytest.mark.asyncio
async def test_procedure_family_delegates_the_same_observation():
    procedure = _procedure()
    analyzer = _CountingAnalyzer("problem")
    executor = _RecordingExecutor(result=_persisted(text="step text"))

    result = await _dispatcher(analyzer, executor=executor).dispatch(
        conversation_id="c1", message="black screen", procedure=procedure
    )

    assert isinstance(result, ExecutedDispatch)
    assert result.family is TurnFamily.PROCEDURE_SUPPORT
    assert result.family == "procedure_support"  # stable string value
    assert result.text == "step text"
    assert result.procedure_result is executor._result

    assert analyzer.calls == 1
    expected = parse_semantic_output(
        _obs_json("problem"), allowed_steps=build_allowed_steps(procedure)
    )
    assert executor.observations == [expected]  # the SAME observation
    assert executor.calls[0]["procedure"] is procedure
    assert executor.calls[0]["conversation_id"] == "c1"


@pytest.mark.asyncio
async def test_procedure_state_conflict_propagates_unchanged():
    executor = _RecordingExecutor(error=ProcedureStateConflict("cas miss"))

    with pytest.raises(ProcedureStateConflict):
        await _dispatcher(_CountingAnalyzer("problem"), executor=executor).dispatch(
            conversation_id="c1", message="m", procedure=_procedure()
        )

    assert len(executor.calls) == 1  # no retry, no overwrite


# ============================================================
# KNOWLEDGE_QA
# ============================================================

@pytest.mark.asyncio
async def test_knowledge_family_never_calls_the_procedure_executor():
    executor = _RecordingExecutor()
    backend = _FakeBackend(items=(KnowledgeItem(text="Windows 10 is supported."),))
    qa = _FakeQA('{"answer": "AngioPlus supports Windows 10."}')

    result = await _dispatcher(
        _CountingAnalyzer("information"), executor=executor, backend=backend, qa=qa
    ).dispatch(conversation_id="c1", message="Which Windows versions?")

    assert isinstance(result, ExecutedDispatch)
    assert result.family is TurnFamily.KNOWLEDGE_QA
    assert result.text == "AngioPlus supports Windows 10."
    assert result.knowledge_result.status is KnowledgeQAStatus.ANSWERED
    assert executor.calls == []  # procedure executor untouched
    assert backend.calls == 1
    assert qa.calls == 1


@pytest.mark.asyncio
async def test_knowledge_statuses_are_preserved():
    answered = await _dispatcher(
        _CountingAnalyzer("information"),
        backend=_FakeBackend(items=(KnowledgeItem(text="Evidence text."),)),
        qa=_FakeQA('{"answer": "Answer."}'),
    ).dispatch(conversation_id="c1", message="q")
    assert answered.knowledge_result.status is KnowledgeQAStatus.ANSWERED
    assert answered.text == "Answer."

    qa = _FakeQA()
    not_answered = await _dispatcher(
        _CountingAnalyzer("information"), backend=_FakeBackend(items=()), qa=qa
    ).dispatch(conversation_id="c1", message="q")
    assert not_answered.knowledge_result.status is KnowledgeQAStatus.NOT_ANSWERED
    assert not_answered.text.strip()  # neutral, non-blank
    assert qa.calls == 0  # no evidence -> provider not called

    internal = await _dispatcher(
        _CountingAnalyzer("information"),
        backend=_FakeBackend(error=RuntimeError("boom")),
    ).dispatch(conversation_id="c1", message="q")
    assert internal.knowledge_result.status is KnowledgeQAStatus.INTERNAL_ERROR
    assert internal.text.strip()


# ============================================================
# CLARIFICATION
# ============================================================

@pytest.mark.asyncio
async def test_clarification_family_touches_neither_executor_nor_knowledge():
    executor = _RecordingExecutor()
    backend = _FakeBackend()
    qa = _FakeQA()

    result = await _dispatcher(
        _CountingAnalyzer("unclear"), executor=executor, backend=backend, qa=qa
    ).dispatch(conversation_id="c1", message="???")

    assert isinstance(result, ExecutedDispatch)
    assert result.family is TurnFamily.CLARIFICATION
    assert result.text == clarify().text
    assert result.clarification_result == clarify()
    assert executor.calls == []
    assert backend.calls == 0
    assert qa.calls == 0


# ============================================================
# EXACTLY ONE FAMILY
# ============================================================

@pytest.mark.asyncio
async def test_exactly_one_family_executes():
    # PROCEDURE_SUPPORT -> executor only
    ex, bk, qa = _RecordingExecutor(), _FakeBackend(), _FakeQA()
    await _dispatcher(_CountingAnalyzer("problem"), executor=ex, backend=bk, qa=qa).dispatch(
        conversation_id="c1", message="m", procedure=_procedure()
    )
    assert len(ex.calls) == 1 and bk.calls == 0 and qa.calls == 0

    # KNOWLEDGE_QA -> backend + provider only
    ex, qa = _RecordingExecutor(), _FakeQA()
    bk = _FakeBackend(items=(KnowledgeItem(text="Evidence text."),))
    await _dispatcher(_CountingAnalyzer("information"), executor=ex, backend=bk, qa=qa).dispatch(
        conversation_id="c1", message="m"
    )
    assert ex.calls == [] and bk.calls == 1 and qa.calls == 1

    # CLARIFICATION -> neither
    ex, bk, qa = _RecordingExecutor(), _FakeBackend(), _FakeQA()
    await _dispatcher(_CountingAnalyzer("unclear"), executor=ex, backend=bk, qa=qa).dispatch(
        conversation_id="c1", message="m"
    )
    assert ex.calls == [] and bk.calls == 0 and qa.calls == 0


# ============================================================
# NO-PROCEDURE (typed, not a None policy signal)
# ============================================================

@pytest.mark.asyncio
async def test_problem_without_a_procedure_returns_no_procedure():
    analyzer = _CountingAnalyzer("problem")
    executor, backend, qa = _RecordingExecutor(), _FakeBackend(), _FakeQA()

    result = await _dispatcher(
        analyzer, executor=executor, backend=backend, qa=qa
    ).dispatch(conversation_id="c1", message="black screen", procedure=None)

    assert isinstance(result, NoProcedureDispatch)
    assert not isinstance(result, ExecutedDispatch)
    assert executor.calls == []  # never run without a Procedure
    assert backend.calls == 0 and qa.calls == 0  # not re-classified
    assert analyzer.calls == 1  # analysis still happened exactly once


# ============================================================
# RESULT CONTRACT
# ============================================================

def test_executed_dispatch_enforces_its_shape():
    good = ExecutedDispatch(
        family=TurnFamily.CLARIFICATION,
        text=clarify().text,
        clarification_result=clarify(),
    )
    assert good.text.strip()

    with pytest.raises(ValueError):  # blank text is impossible for "executed"
        ExecutedDispatch(
            family=TurnFamily.CLARIFICATION,
            text="   ",
            clarification_result=clarify(),
        )

    with pytest.raises(ValueError):  # missing its one payload
        ExecutedDispatch(family=TurnFamily.KNOWLEDGE_QA, text="x")

    with pytest.raises(ValueError):  # mismatched payload
        ExecutedDispatch(
            family=TurnFamily.CLARIFICATION,
            text="x",
            knowledge_result=KnowledgeQAResult(
                status=KnowledgeQAStatus.NOT_ANSWERED, text="neutral"
            ),
        )


# ============================================================
# NOT WIRED ANYWHERE
# ============================================================

def test_module_is_not_wired_into_runtime():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "support/support_core.py",
        "support/orchestrator.py",
        "support/turn_executor.py",
        "support/alice_analyzer.py",
        "support/knowledge_qa.py",
        "support/clarification.py",
        "support/turn_family.py",
        "support/router.py",
        "yandex/chat/server.py",
        "yandex/chat/alice_support_agent.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "turn_dispatch" not in text, relative
        assert "SupportTurnDispatcher" not in text, relative
