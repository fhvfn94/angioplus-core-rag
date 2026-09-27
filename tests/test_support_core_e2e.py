# -*- coding: utf-8 -*-
"""
Slice 11B.3 -- OFFLINE END-TO-END support-core scenarios.

These tests exercise the ASSEMBLED SupportCore architecture end to end with
NO real network / Yandex / File Search / Tracker / Telegram / n8n / DB / HTTP /
voice. They deliberately drive the REAL composition chain:

    SupportCore.handle_turn
      -> the REAL SupportTurnDispatcher
           -> the REAL analyzer boundary (INJECTED provider double)
           -> global policy + deterministic family selection
           -> the REAL SupportTurnExecutor (INJECTED session/store doubles)
                -> the REAL orchestrator -> reducer -> router -> generator

Only the four leaf PORTS (analyzer provider, knowledge backend, knowledge-QA
provider, response provider) and the persistence boundary (session factory +
ProcedureStateStore) are doubled, so these tests prove the COMPOSITION
contract rather than re-testing each helper in isolation.

Cross-cutting invariants asserted by the scenarios below:

  * SINGLE ANALYSIS -- the analyzer runs EXACTLY ONCE per turn, on every path;
  * STATE OWNERSHIP -- ProcedureState is persisted ONLY by the executor's
    CAS path (never by SupportCore), so knowledge / clarification / terminal /
    no-procedure turns write NOTHING;
  * BOUND IDENTITY -- a bound conversation reuses the EXACT persisted
    (procedure_id, procedure_version) pair, never silently switching;
  * IDEMPOTENT REPLAY -- repeating the SAME opaque execution id never
    advances the durable state twice, never writes twice and never re-words
    a replayed turn (HL-2c.1).

DEFERRED (documented, NOT exercised here): the RESOLVED + procedure=None
lifecycle stays deferred -- this slice invents no behaviour for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import pytest

from support import support_core as sc
from support.clarification import clarify
from support.global_policy import GlobalAction
from support.knowledge import KnowledgeItem
from support.knowledge_qa import KnowledgeQAStatus
from support.observation import ProblemStatus, RequestKind
from support.procedure import load_procedure_from_dict
from support.procedure_catalog import ProcedureCatalog
from support.procedure_state import ProcedureState
from support.procedure_state_store import (
    COLD_START_REVISION,
    CasResult,
    PersistedProcedureState,
    ProcedureStateConflict,
)
from support.support_core import SupportCore
from support.turn_projection import project_global_terminal


# ============================================================
# IN-MEMORY PERSISTENCE BOUNDARY (the "database")
# ============================================================

class _InMemoryDB:
    """The shared state of one pretend database.

    ``states`` maps a conversation id to ``(ProcedureState, revision)``.
    ``execution_ids`` maps a conversation id to the OPAQUE idempotency id of
    its LAST write (HL-2b), so a repeated id can be recognised. Counters make
    the persistence invariants observable.
    """

    def __init__(self) -> None:
        self.states: dict[str, tuple[ProcedureState, int]] = {}
        self.execution_ids: dict[str, Optional[str]] = {}
        self.read_calls = 0
        self.write_attempts = 0
        self.commits = 0
        self.rollbacks = 0
        self.raise_on_write = False
        self.journal: list[tuple] = []


class _FakeSession:
    """Minimal AsyncSession double (async context manager + commit/rollback)."""

    def __init__(self, db: _InMemoryDB) -> None:
        self._db = db

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def commit(self) -> None:
        self._db.commits += 1

    async def rollback(self) -> None:
        self._db.rollbacks += 1


class _InMemoryStore:
    """ProcedureStateStore double with real compare-and-swap semantics."""

    def __init__(self, db: _InMemoryDB) -> None:
        self._db = db

    async def read(
        self, conversation_id: str
    ) -> Optional[PersistedProcedureState]:
        self._db.read_calls += 1
        self._db.journal.append(("read", conversation_id))
        entry = self._db.states.get(conversation_id)
        if entry is None:
            return None
        state, revision = entry
        return PersistedProcedureState(
            state=state,
            revision=revision,
            last_execution_id=self._db.execution_ids.get(conversation_id),
        )

    async def save_cas(
        self,
        conversation_id: str,
        state: ProcedureState,
        *,
        expected_revision: Optional[int],
        execution_id: Optional[str] = None,
    ) -> CasResult:
        self._db.write_attempts += 1
        self._db.journal.append(
            ("save_cas", conversation_id, expected_revision, execution_id)
        )
        if self._db.raise_on_write:
            # A concurrent writer won the race; the store refuses to write.
            raise ProcedureStateConflict(
                f"procedure state for {conversation_id!r} raced"
            )

        # HL-2b: one execution id advances the state AT MOST ONCE. A CAS that
        # loses to the SAME id is an IDEMPOTENT NO-OP that returns the
        # persisted winner (never a second advance, never an overwrite).
        if (
            execution_id is not None
            and self._db.execution_ids.get(conversation_id) == execution_id
        ):
            winner_state, winner_revision = self._db.states[conversation_id]
            return CasResult(
                revision=winner_revision, applied=False, state=winner_state
            )

        entry = self._db.states.get(conversation_id)
        if expected_revision is None:
            if entry is not None:
                raise ProcedureStateConflict("row appeared concurrently")
            new_revision = COLD_START_REVISION
        else:
            if entry is None or entry[1] != expected_revision:
                raise ProcedureStateConflict("stale revision")
            new_revision = expected_revision + 1
        self._db.states[conversation_id] = (state, new_revision)
        self._db.execution_ids[conversation_id] = execution_id
        return CasResult(revision=new_revision, applied=True, state=state)


def _session_factory(db: _InMemoryDB):
    return lambda: _FakeSession(db)


def _store_factory(db: _InMemoryDB):
    return lambda session: _InMemoryStore(db)


# ============================================================
# LEAF PORT DOUBLES (no network)
# ============================================================

def _obs_json(
    request_kind: str,
    *,
    scope: str = "in_scope",
    problem_status: Optional[str] = None,
    human_requested: bool = False,
    reported_actions: tuple = (),
) -> str:
    """A full, strict analyzer JSON payload (all seven required fields)."""
    return json.dumps(
        {
            "scope": scope,
            "reported_actions": [
                {"step_id": step_id, "outcome": outcome}
                for step_id, outcome in reported_actions
            ],
            "problem_status": problem_status,
            "human_requested": human_requested,
            "clarification_facts": [],
            "new_issue": None,
            "request_kind": request_kind,
        }
    )


class _RecordingAnalyzer:
    """Frozen analyzer-port double: returns canned JSON, counts its calls.

    ``return_raw`` re-programs the analyzer (e.g. for the second turn of a
    continuation), ``fail_with`` simulates a provider failure.
    """

    def __init__(self, request_kind: str = "problem", **kwargs) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.histories: list[list] = []
        self._raw = _obs_json(request_kind, **kwargs)
        self._error: Optional[BaseException] = None

    def return_raw(self, raw: str) -> "_RecordingAnalyzer":
        self._raw = raw
        return self

    def fail_with(self, error: BaseException) -> "_RecordingAnalyzer":
        self._error = error
        return self

    def __call__(self, prompt, history=()):
        self.calls += 1
        self.prompts.append(prompt)
        self.histories.append(list(history))
        if self._error is not None:
            raise self._error
        return self._raw


class _FakeBackend:
    """KnowledgeBackend double returning a fixed (possibly empty) list."""

    def __init__(self, items=()) -> None:
        self._items = list(items)
        self.calls = 0

    def search(self, query, *, max_num_results):
        self.calls += 1
        return list(self._items)


class _FailingBackend:
    """KnowledgeBackend double that raises (retrieval infrastructure down)."""

    def __init__(self) -> None:
        self.calls = 0

    def search(self, query, *, max_num_results):
        self.calls += 1
        raise RuntimeError("retrieval transport down")


class _FakeQA:
    """KnowledgeQAProvider double."""

    def __init__(self, raw: str = '{"answer": "Grounded answer."}') -> None:
        self._raw = raw
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self._raw


class _FakeResponseProvider:
    """ResponseProvider double (never reaches the network).

    An empty explanation is rejected by the generator, so the deterministic
    procedure-owned step action is rendered -- no provider text is surfaced.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return '{"explanation": ""}'


class _RecordingCatalog:
    """Delegating ProcedureCatalog double recording which lookup ran."""

    def __init__(self, catalog: ProcedureCatalog) -> None:
        self._catalog = catalog
        self.select_calls: list[str] = []
        self.get_calls: list[tuple] = []

    def select(self, text):
        self.select_calls.append(text)
        return self._catalog.select(text)

    def get(self, procedure_id, version):
        self.get_calls.append((procedure_id, version))
        return self._catalog.get(procedure_id, version)


# ============================================================
# A REAL TWO-STEP PROCEDURE (declarative, validated)
# ============================================================

def _two_step_procedure():
    """BLACK_SCREEN v1: STEP_A -> (failed) -> STEP_B -> (failed) -> ESCALATE."""
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
                    "action": "Check the power LED is lit.",
                    "evidence_query": "angioplus black screen power led",
                    "transitions": {
                        "attempted_failed": "STEP_B",
                        "resolved": "RESOLVED",
                    },
                },
                {
                    "id": "STEP_B",
                    "action": "Reseat the display cable.",
                    "evidence_query": "angioplus black screen display cable",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                },
            ],
        }
    )


# ============================================================
# HARNESS
# ============================================================

@dataclass
class _Harness:
    core: SupportCore
    db: _InMemoryDB
    catalog: object
    analyzer: _RecordingAnalyzer
    backend: object
    qa: _FakeQA
    response: _FakeResponseProvider


def _build(
    catalog=None,
    *,
    analyzer: Optional[_RecordingAnalyzer] = None,
    backend=None,
    qa: Optional[_FakeQA] = None,
    response: Optional[_FakeResponseProvider] = None,
    db: Optional[_InMemoryDB] = None,
) -> _Harness:
    """Compose the REAL SupportCore with only the leaf ports / DB doubled."""
    db = db if db is not None else _InMemoryDB()
    catalog = (
        catalog
        if catalog is not None
        else ProcedureCatalog([_two_step_procedure()])
    )
    analyzer = analyzer if analyzer is not None else _RecordingAnalyzer("problem")
    backend = backend if backend is not None else _FakeBackend()
    qa = qa if qa is not None else _FakeQA()
    response = response if response is not None else _FakeResponseProvider()

    core = SupportCore(
        procedure_catalog=catalog,
        analyzer_provider=analyzer,
        knowledge_backend=backend,
        knowledge_qa_provider=qa,
        response_provider=response,
        session_factory=_session_factory(db),
        store_factory=_store_factory(db),
    )
    return _Harness(core, db, catalog, analyzer, backend, qa, response)


def _recording_catalog() -> _RecordingCatalog:
    return _RecordingCatalog(ProcedureCatalog([_two_step_procedure()]))


def _seed_bound(
    db: _InMemoryDB,
    conversation_id: str,
    procedure_id: str,
    version: Optional[int],
    *,
    revision: int = 4,
    last_recommended_step: Optional[str] = None,
) -> ProcedureState:
    """Seed an already-bound conversation's persisted ProcedureState."""
    state = ProcedureState(
        procedure_id=procedure_id,
        procedure_version=version,
        last_recommended_step=last_recommended_step,
    )
    db.states[conversation_id] = (state, revision)
    return state


# ============================================================
# 1. PROBLEM -- procedure starts
# ============================================================

@pytest.mark.asyncio
async def test_problem_starts_a_procedure_end_to_end():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen on boot"
    )

    assert outcome.status == sc.TURN_EXECUTED
    assert outcome.persisted is not None

    # The catalog is the ONLY cold-start mapper and it saw the raw message.
    assert catalog.select_calls == ["black screen on boot"]
    assert catalog.get_calls == []

    # ONE analysis, ONE CAS write through the executor.
    assert h.analyzer.calls == 1
    assert h.db.write_attempts == 1
    assert h.db.commits == 1

    # Correct procedure identity, bound by the executor's next state.
    assert outcome.persisted.state.procedure_id == "BLACK_SCREEN"
    assert outcome.persisted.state.procedure_version == 1
    assert outcome.persisted.state.last_recommended_step == "STEP_A"
    assert outcome.persisted.revision == COLD_START_REVISION

    # The persisted revision/state in the store IS the executor's result.
    stored_state, stored_revision = h.db.states["chat-1"]
    assert stored_state is outcome.persisted.state
    assert stored_revision == outcome.persisted.revision

    # A non-blank, user-facing response -- the procedure-owned step action.
    assert outcome.persisted.turn.assistant_text.strip()
    assert outcome.persisted.turn.assistant_text == "Check the power LED is lit."
    assert outcome.text is None  # the response is carried by the persisted turn

    # The executor consumed the SAME single analysis (no second analyzer run).
    assert outcome.persisted.turn.observation.request_kind is RequestKind.PROBLEM
    assert h.response.calls == 0  # no evidence -> deterministic step action


# ============================================================
# 2. PROBLEM -- procedure continues across turns
# ============================================================

@pytest.mark.asyncio
async def test_problem_continues_across_turns_without_switching_procedure():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))

    first = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen on boot"
    )
    assert first.status == sc.TURN_EXECUTED
    assert first.persisted.revision == 1
    assert first.persisted.state.last_recommended_step == "STEP_A"

    # Turn 2: the user reports the first step did NOT help.
    h.analyzer.return_raw(
        _obs_json(
            "problem",
            problem_status="ongoing",
            reported_actions=(("STEP_A", "failed"),),
        )
    )
    second = await h.core.handle_turn(
        conversation_id="chat-1", message="I tried that, still black screen"
    )

    assert second.status == sc.TURN_EXECUTED
    assert second.persisted.revision == 2  # revision progressed
    assert second.persisted.state.last_recommended_step == "STEP_B"
    assert second.persisted.state.problem_status is ProblemStatus.ONGOING

    # The EXACT persisted identity was reused; cold-start select never re-ran.
    assert catalog.select_calls == ["black screen on boot"]
    assert catalog.get_calls == [("BLACK_SCREEN", 1)]

    # No silent procedure switch: same id/version on both turns.
    assert second.persisted.state.procedure_id == "BLACK_SCREEN"
    assert second.persisted.state.procedure_version == 1

    # ONE analysis per turn.
    assert h.analyzer.calls == 2
    assert h.db.states["chat-1"][1] == 2


# ============================================================
# 3. INFORMATION -- cold conversation
# ============================================================

@pytest.mark.asyncio
async def test_cold_information_is_answered_by_knowledge_qa():
    h = _build(
        analyzer=_RecordingAnalyzer("information"),
        backend=_FakeBackend(
            items=(KnowledgeItem(text="AngioPlus Core supports Windows 10/11."),)
        ),
        qa=_FakeQA('{"answer": "AngioPlus Core supports Windows 10 and 11."}'),
    )

    outcome = await h.core.handle_turn(
        conversation_id="chat-1",
        message="Which Windows versions does AngioPlus support?",
    )

    assert outcome.status == sc.KNOWLEDGE_QA
    assert outcome.knowledge.status is KnowledgeQAStatus.ANSWERED
    assert outcome.knowledge.text == "AngioPlus Core supports Windows 10 and 11."

    assert h.analyzer.calls == 1  # ONE analysis
    assert h.backend.calls == 1
    assert h.qa.calls == 1

    # No procedure ran and ZERO ProcedureState writes happened.
    assert h.db.write_attempts == 0
    assert h.response.calls == 0


# ============================================================
# 4. INFORMATION while a procedure is bound  (critical)
# ============================================================

@pytest.mark.asyncio
async def test_knowledge_question_while_bound_does_not_touch_the_procedure():
    catalog = _recording_catalog()
    h = _build(
        catalog,
        analyzer=_RecordingAnalyzer("information"),
        backend=_FakeBackend(items=(KnowledgeItem(text="Windows support."),)),
        qa=_FakeQA('{"answer": "Yes, Windows is supported."}'),
    )
    seeded = _seed_bound(
        h.db, "chat-1", "BLACK_SCREEN", 1, revision=4, last_recommended_step="STEP_A"
    )

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="Which Windows versions are supported?"
    )

    assert outcome.status == sc.KNOWLEDGE_QA
    assert outcome.knowledge.status is KnowledgeQAStatus.ANSWERED

    # The exact bound identity was rehydrated, but the procedure did NOT run.
    assert catalog.get_calls == [("BLACK_SCREEN", 1)]
    assert catalog.select_calls == []
    assert h.analyzer.calls == 1

    # The active procedure state is byte-for-byte unchanged, and NOTHING was
    # written merely because a procedure happens to be bound.
    assert h.db.write_attempts == 0
    assert h.db.states["chat-1"] == (seeded, 4)


# ============================================================
# 5. UNCLEAR / analyzer fail-safe
# ============================================================

@pytest.mark.asyncio
async def test_unclear_message_yields_deterministic_clarification():
    h = _build(analyzer=_RecordingAnalyzer("unclear"))

    outcome = await h.core.handle_turn(conversation_id="chat-1", message="???")

    assert outcome.status == sc.CLARIFICATION
    assert outcome.clarification.text == clarify().text
    assert h.analyzer.calls == 1
    assert h.db.write_attempts == 0


@pytest.mark.asyncio
async def test_analyzer_provider_failure_fails_safe_to_clarification():
    h = _build(
        analyzer=_RecordingAnalyzer().fail_with(RuntimeError("provider down"))
    )

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.CLARIFICATION
    assert h.analyzer.calls == 1  # attempted exactly once, fail-safe
    assert h.db.write_attempts == 0


@pytest.mark.asyncio
async def test_malformed_analyzer_output_fails_safe_to_clarification():
    h = _build(analyzer=_RecordingAnalyzer().return_raw("{not valid json"))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.CLARIFICATION
    assert h.analyzer.calls == 1
    assert h.db.write_attempts == 0


# ============================================================
# 6. HUMAN REQUEST
# ============================================================

@pytest.mark.asyncio
async def test_human_request_escalates_without_running_a_family():
    h = _build(analyzer=_RecordingAnalyzer("problem", human_requested=True))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="please connect me to a human"
    )

    assert outcome.status == sc.ESCALATE
    assert outcome.text == project_global_terminal(GlobalAction.ESCALATE)
    assert h.analyzer.calls == 1  # ONE analysis preceded the terminal
    # No family capability, no procedure executor, ZERO writes (no Tracker).
    assert h.db.write_attempts == 0
    assert h.response.calls == 0
    assert h.backend.calls == 0


# ============================================================
# 7. OUT OF SCOPE
# ============================================================

@pytest.mark.asyncio
async def test_out_of_scope_declines_without_running_a_family():
    h = _build(analyzer=_RecordingAnalyzer("information", scope="out_of_scope"))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="what is the weather today?"
    )

    assert outcome.status == sc.OUT_OF_SCOPE
    assert outcome.text == project_global_terminal(GlobalAction.OUT_OF_SCOPE)
    assert h.analyzer.calls == 1
    assert h.db.write_attempts == 0
    assert h.backend.calls == 0


@pytest.mark.asyncio
async def test_out_of_scope_wins_over_a_human_request():
    h = _build(
        analyzer=_RecordingAnalyzer(
            "information", scope="out_of_scope", human_requested=True
        )
    )

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="weather? and get me a human too"
    )

    assert outcome.status == sc.OUT_OF_SCOPE
    assert h.db.write_attempts == 0


# ============================================================
# 8. PROBLEM + no procedure candidate
# ============================================================

@pytest.mark.asyncio
async def test_problem_without_a_candidate_is_no_procedure_not_clarification():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="my coffee machine is broken"
    )

    assert outcome.status == sc.NO_PROCEDURE
    assert outcome.persisted is None
    assert outcome.text is None  # nothing invented
    assert outcome.knowledge is None and outcome.clarification is None

    assert catalog.select_calls == ["my coffee machine is broken"]
    assert h.analyzer.calls == 1  # analysis still happened exactly once
    assert h.db.write_attempts == 0  # no executor call, no writes


# ============================================================
# 9. Persisted exact procedure identity is missing
# ============================================================

@pytest.mark.asyncio
async def test_missing_bound_procedure_id_fails_closed():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))
    seeded = _seed_bound(h.db, "chat-1", "GHOST", 1, revision=3)

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen on boot"
    )

    assert outcome.status == sc.PROCEDURE_NOT_FOUND
    assert outcome.persisted is None

    assert h.analyzer.calls == 0  # fail closed BEFORE any analysis
    assert catalog.select_calls == []  # never re-selected from raw text
    assert catalog.get_calls == [("GHOST", 1)]  # EXACT pair only
    assert h.db.write_attempts == 0
    assert h.db.states["chat-1"] == (seeded, 3)  # state not overwritten


@pytest.mark.asyncio
async def test_wrong_bound_version_never_falls_back_to_latest():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))
    seeded = _seed_bound(h.db, "chat-1", "BLACK_SCREEN", 99, revision=5)

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen on boot"
    )

    assert outcome.status == sc.PROCEDURE_NOT_FOUND
    assert h.analyzer.calls == 0
    assert catalog.select_calls == []
    assert catalog.get_calls == [("BLACK_SCREEN", 99)]  # no upgrade to v1
    assert h.db.write_attempts == 0
    assert h.db.states["chat-1"] == (seeded, 5)


# ============================================================
# 10. ProcedureStateConflict (CAS loses the race)
# ============================================================

@pytest.mark.asyncio
async def test_cas_conflict_propagates_unchanged_without_retry_or_overwrite():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))
    seeded = _seed_bound(
        h.db, "chat-1", "BLACK_SCREEN", 1, revision=4, last_recommended_step="STEP_A"
    )
    h.db.raise_on_write = True  # a concurrent turn persisted first

    with pytest.raises(ProcedureStateConflict):
        await h.core.handle_turn(
            conversation_id="chat-1", message="still black screen"
        )

    assert h.analyzer.calls == 1          # no second analysis
    assert h.db.write_attempts == 1        # exactly ONE attempt: no retry
    assert h.db.rollbacks == 1             # the executor rolled back
    assert h.db.states["chat-1"] == (seeded, 4)  # nothing overwritten


# ============================================================
# OPTIONAL -- Knowledge QA non-answers
# ============================================================

@pytest.mark.asyncio
async def test_knowledge_qa_without_evidence_is_a_neutral_non_answer():
    h = _build(analyzer=_RecordingAnalyzer("information"), backend=_FakeBackend())

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="What is the warranty period?"
    )

    assert outcome.status == sc.KNOWLEDGE_QA
    assert outcome.knowledge.status is KnowledgeQAStatus.NOT_ANSWERED
    assert h.qa.calls == 0  # no evidence -> provider never called
    assert h.db.write_attempts == 0


@pytest.mark.asyncio
async def test_knowledge_qa_infrastructure_failure_is_internal_error():
    h = _build(analyzer=_RecordingAnalyzer("information"), backend=_FailingBackend())

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="What is PACS?"
    )

    assert outcome.status == sc.KNOWLEDGE_QA
    assert outcome.knowledge.status is KnowledgeQAStatus.INTERNAL_ERROR
    assert h.db.write_attempts == 0


# ============================================================
# 11. IDEMPOTENT REPLAY -- the SAME execution id (HL-2c.1)
# ============================================================

@pytest.mark.asyncio
async def test_same_execution_id_replay_does_not_advance_or_reword():
    catalog = _recording_catalog()
    h = _build(catalog, analyzer=_RecordingAnalyzer("problem"))
    execution_id = "evt-opaque-123"

    first = await h.core.handle_turn(
        conversation_id="chat-1",
        message="black screen on boot",
        execution_id=execution_id,
    )

    # FIRST CALL: a NORMAL, newly executed turn.
    assert first.status == sc.TURN_EXECUTED
    assert first.persisted.replayed is False
    assert first.persisted.turn is not None
    assert first.persisted.turn.assistant_text.strip()
    assert first.persisted.revision == COLD_START_REVISION

    # Exactly ONE CAS write, and the OPAQUE id was recorded WITH the state.
    assert h.db.write_attempts == 1
    assert h.db.commits == 1
    assert h.db.execution_ids["chat-1"] == execution_id
    winner_state, winner_revision = h.db.states["chat-1"]

    # SECOND CALL: the SAME execution id -> a legitimate HL-2b replay.
    second = await h.core.handle_turn(
        conversation_id="chat-1",
        message="black screen on boot",
        execution_id=execution_id,
    )

    # SupportCore did NOT crash and reused the EXISTING outcome vocabulary.
    assert second.status == sc.TURN_EXECUTED
    assert second.conversation_id == "chat-1"

    # The replay carries the persisted WINNER, not a recomputed candidate.
    assert second.persisted.replayed is True
    assert second.persisted.turn is None
    assert second.persisted.state is winner_state
    assert second.persisted.revision == winner_revision

    # NO second advancement, NO second write, NO second transaction.
    assert h.db.write_attempts == 1
    assert h.db.commits == 1
    assert h.db.states["chat-1"] == (winner_state, winner_revision)

    # NOTHING was fabricated to represent the replay.
    assert second.text is None
    assert second.knowledge is None
    assert second.clarification is None

    # Two SupportCore calls -> exactly two analyses: the replay is detected
    # INSIDE the executor, i.e. AFTER the ONE analysis that names the family
    # (runtime-level finalized replay will later short-circuit ABOVE SupportCore).
    assert h.analyzer.calls == 2


@pytest.mark.asyncio
async def test_a_different_execution_id_is_a_new_turn_not_a_replay():
    h = _build(_recording_catalog(), analyzer=_RecordingAnalyzer("problem"))

    first = await h.core.handle_turn(
        conversation_id="chat-1",
        message="black screen on boot",
        execution_id="evt-1",
    )
    second = await h.core.handle_turn(
        conversation_id="chat-1",
        message="black screen on boot",
        execution_id="evt-2",
    )

    # A DIFFERENT id is NOT idempotent: the durable state advanced again.
    assert first.persisted.replayed is False
    assert second.persisted.replayed is False
    assert second.persisted.turn is not None
    assert second.persisted.revision == first.persisted.revision + 1
    assert h.db.write_attempts == 2
    assert h.db.execution_ids["chat-1"] == "evt-2"


@pytest.mark.asyncio
async def test_omitting_the_execution_id_keeps_the_legacy_flow():
    h = _build(_recording_catalog(), analyzer=_RecordingAnalyzer("problem"))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen on boot"
    )

    assert outcome.status == sc.TURN_EXECUTED
    assert outcome.persisted.replayed is False
    assert outcome.persisted.turn is not None
    # No id was EVER recorded: a non-idempotent caller stays non-idempotent.
    assert h.db.execution_ids["chat-1"] is None
    assert h.db.write_attempts == 1


@pytest.mark.asyncio
async def test_execution_id_is_inert_on_every_non_procedure_family():
    # KNOWLEDGE_QA: the id must NOT create a write / executor call / change.
    h = _build(
        analyzer=_RecordingAnalyzer("information"),
        backend=_FakeBackend(items=(KnowledgeItem(text="Windows support."),)),
        qa=_FakeQA('{"answer": "Yes."}'),
    )
    knowledge = await h.core.handle_turn(
        conversation_id="chat-1",
        message="Which Windows versions are supported?",
        execution_id="evt-opaque-123",
    )
    assert knowledge.status == sc.KNOWLEDGE_QA
    assert knowledge.knowledge.status is KnowledgeQAStatus.ANSWERED
    assert h.db.write_attempts == 0
    assert h.db.execution_ids == {}
    assert h.analyzer.calls == 1

    # GLOBAL ESCALATE: inert.
    h2 = _build(analyzer=_RecordingAnalyzer("problem", human_requested=True))
    escalated = await h2.core.handle_turn(
        conversation_id="chat-1",
        message="connect me to a human",
        execution_id="evt-opaque-123",
    )
    assert escalated.status == sc.ESCALATE
    assert h2.db.write_attempts == 0
    assert h2.db.execution_ids == {}

    # CLARIFICATION: inert.
    h3 = _build(analyzer=_RecordingAnalyzer("unclear"))
    clarified = await h3.core.handle_turn(
        conversation_id="chat-1",
        message="???",
        execution_id="evt-opaque-123",
    )
    assert clarified.status == sc.CLARIFICATION
    assert h3.db.write_attempts == 0
    assert h3.db.execution_ids == {}

    # OUT_OF_SCOPE: inert.
    h4 = _build(
        analyzer=_RecordingAnalyzer("information", scope="out_of_scope")
    )
    declined = await h4.core.handle_turn(
        conversation_id="chat-1",
        message="what is the weather today?",
        execution_id="evt-opaque-123",
    )
    assert declined.status == sc.OUT_OF_SCOPE
    assert h4.db.write_attempts == 0
    assert h4.db.execution_ids == {}

    # PROBLEM + no candidate (NO_PROCEDURE): inert.
    h5 = _build(_recording_catalog(), analyzer=_RecordingAnalyzer("problem"))
    no_proc = await h5.core.handle_turn(
        conversation_id="chat-1",
        message="my coffee machine is broken",
        execution_id="evt-opaque-123",
    )
    assert no_proc.status == sc.NO_PROCEDURE
    assert h5.db.write_attempts == 0
    assert h5.db.execution_ids == {}

    # PROCEDURE_NOT_FOUND: inert (fail closed BEFORE any analysis).
    h6 = _build(_recording_catalog(), analyzer=_RecordingAnalyzer("problem"))
    _seed_bound(h6.db, "chat-1", "GHOST", 1, revision=3)
    not_found = await h6.core.handle_turn(
        conversation_id="chat-1",
        message="black screen",
        execution_id="evt-opaque-123",
    )
    assert not_found.status == sc.PROCEDURE_NOT_FOUND
    assert h6.analyzer.calls == 0
    assert h6.db.write_attempts == 0
    assert h6.db.execution_ids == {}
