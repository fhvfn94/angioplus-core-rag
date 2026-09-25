# -*- coding: utf-8 -*-
"""
Focused tests for the Slice 11A.3a Support Core COMPOSITION boundary and the
Slice 11A.3b turn EXECUTION flow.

These tests are PURE: NO network, NO credentials, NO real database. They
prove that ``support/support_core.py`` composes the already-existing Support
Core components, re-implements NONE of their responsibilities, and drives
ONE turn by DELEGATING to them:

  1. the production composition uses ``ProcedureCatalog``;
  2. the analyzer dependency is ``YandexAnalyzerProvider``;
  3. the knowledge dependency is ``YandexKnowledgeBackend``;
  4. the response dependency is ``YandexExplanationProvider``;
  5. the existing ``SupportTurnExecutor`` / ``ProcedureStateStore`` are
     COMPOSED, not re-implemented;
  6. ``support_core.py`` contains no routing / reducing / retrieval /
     generation implementation (no module-level helper, no async behaviour
     outside the composition boundary);
  7. no runtime module imports ``support_core`` yet;
  8. ``yandex/chat/server.py`` stays untouched by this sub-slice;
  9. ``handle_turn`` resolves the Procedure cold-start via
     ``ProcedureCatalog.select`` (unique match -> that Procedure; no match or
     ambiguous -> safe ``NO_PROCEDURE`` with the executor NOT called);
 10. a BOUND conversation resolves via the EXACT persisted
     ``(procedure_id, procedure_version)`` pair (``catalog.get``), never
     re-running ``select``, never upgrading/falling back to another version
     and never using ``latest``;
 11. an un-rehydratable identity -> safe ``PROCEDURE_NOT_FOUND`` with the
     executor NOT called;
 12. the executor receives the SupportCore-composed analyzer / knowledge /
     response providers and the caller's opaque ``history``;
 13. a ``ProcedureStateConflict`` propagates unchanged (no retry, no
     overwrite) and SupportCore never persists ``ProcedureState`` itself;
 14. the durable read -> process -> CAS WRITE transaction lifecycle stays
     owned by ``SupportTurnExecutor`` (SupportCore's own read is closed
     BEFORE the executor is called);
 15. SupportCore neither imports nor calls the orchestrator / router.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from support import support_core as sc
from support.alice_analyzer_yandex import YandexAnalyzerProvider
from support.knowledge_yandex import YandexKnowledgeBackend
from support.procedure import load_procedure_from_dict
from support.procedure_catalog import ProcedureCatalog
from support.procedure_state import ProcedureState
from support.procedure_state_store import (
    PersistedProcedureState,
    ProcedureStateConflict,
    ProcedureStateStore,
)
from support.response_generator_yandex import YandexExplanationProvider
from support.turn_executor import PersistedTurnResult, SupportTurnExecutor

# ============================================================
# DOUBLES (no network, no credentials, no DB)
# ============================================================

class _FakeSessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc_info):
        return False


class _FakeSessionFactory:
    """Session-factory double: opening it is observable but harmless."""

    def __init__(self) -> None:
        self.opened = 0

    def __call__(self):
        self.opened += 1
        return _FakeSessionContext()


def _sentinel_store_factory(session):
    return ("state-store-for", session)


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


_RUNTIME_MODULES = (
    Path("yandex/chat/server.py"),
    Path("yandex/chat/alice_support_agent.py"),
    Path("yandex/realtime/web/server.py"),
)


def _repo_root() -> Path:
    return Path(sc.__file__).resolve().parents[1]


# ============================================================
# 1. THE PRODUCTION COMPOSITION USES ProcedureCatalog
# ============================================================

def test_production_composition_uses_the_procedure_catalog():
    core = sc.SupportCore()

    assert isinstance(core.procedure_catalog, ProcedureCatalog)
    # The DEFAULT is the production catalog loaded from support/procedures/.
    production = ProcedureCatalog.from_directory()
    assert core.procedure_catalog.procedure_ids == production.procedure_ids
    assert core.procedure_catalog.procedure_ids  # non-empty

    injected = ProcedureCatalog([_procedure()])
    assert sc.SupportCore(procedure_catalog=injected).procedure_catalog is injected


# ============================================================
# 2-4. THE PORT IMPLEMENTATIONS ARE THE EXISTING YANDEX ADAPTERS
# ============================================================

def test_analyzer_dependency_is_the_existing_yandex_adapter():
    assert isinstance(
        sc.SupportCore().analyzer_provider, YandexAnalyzerProvider
    )

    injected = object()
    assert (
        sc.SupportCore(analyzer_provider=injected).analyzer_provider
        is injected
    )


def test_knowledge_dependency_is_the_existing_yandex_adapter():
    assert isinstance(
        sc.SupportCore().knowledge_backend, YandexKnowledgeBackend
    )

    injected = object()
    assert (
        sc.SupportCore(knowledge_backend=injected).knowledge_backend
        is injected
    )


def test_response_dependency_is_the_existing_yandex_adapter():
    assert isinstance(
        sc.SupportCore().response_provider, YandexExplanationProvider
    )

    injected = object()
    assert (
        sc.SupportCore(response_provider=injected).response_provider
        is injected
    )


# ============================================================
# 5. EXISTING EXECUTOR / STATE STORE ARE COMPOSED, NOT RE-IMPLEMENTED
# ============================================================

def test_existing_turn_executor_and_state_store_are_composed():
    sessions = _FakeSessionFactory()

    core = sc.SupportCore(
        session_factory=sessions,
        store_factory=_sentinel_store_factory,
    )

    assert core.store_factory is _sentinel_store_factory
    assert sessions.opened == 0  # composing opens nothing

    executor = core.turn_executor
    assert isinstance(executor, SupportTurnExecutor)
    assert executor is core.turn_executor  # composed once, then reused
    assert sessions.opened == 0  # still no session opened


def test_composed_dependencies_are_imported_not_reimplemented():
    tree = _tree()

    top_level_classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    # The container + the ONE outcome value object it returns; nothing else.
    assert top_level_classes == {"SupportCore", "SupportTurnOutcome"}

    imported = _module_level_imports(tree)
    assert {
        "support.procedure_catalog",
        "support.alice_analyzer_yandex",
        "support.knowledge_yandex",
        "support.response_generator_yandex",
        "support.procedure_state_store",
        "support.turn_executor",
    } <= imported

    # The executor stays the ONLY path to the orchestrator: neither the
    # orchestrator nor the router is imported or bypassed here.
    assert "support.orchestrator" not in imported
    assert "support.router" not in imported

    # No transport / runtime module is composed here.
    assert {name.split(".")[0] for name in imported}.isdisjoint({"yandex"})


def test_state_store_dependency_defaults_to_the_existing_store():
    assert sc.SupportCore().store_factory is ProcedureStateStore


# ============================================================
# 6. NO ROUTING / REDUCING / RETRIEVAL / GENERATION IMPLEMENTATION
# ============================================================

def _source() -> str:
    return Path(sc.__file__).read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _module_level_imports(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _code_only_text(tree: ast.Module) -> str:
    """Identifiers / attribute names / keywords / def+class names and string
    literals -- everything EXCEPT comments and docstrings."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            parts.append(node.arg)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            parts.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                parts.append(node.value)

    return " ".join(parts).lower()


def test_support_core_contains_no_routing_reducing_retrieval_or_generation():
    tree = _tree()

    # Composition only: no module-level helper at all.
    assert [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ] == []

    # The ONLY async behaviour is the composition boundary itself, as CLASS
    # methods: no nested/module-level async helper can hide policy here.
    class_level_async = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
    ]
    assert {node.name for node in class_level_async} == {
        "handle_turn",
        "_read_persisted_state",
    }
    allowed = {id(node) for node in class_level_async}
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            assert id(node) in allowed, node.name

    code = _code_only_text(tree)
    for token in (
        # routing / policy
        "routeaction",
        "routingdecision",
        "apply_support_policy",
        "authoritative_outcome",
        # orchestration / reduction / durable-turn duplication
        "process_support_turn",
        "reduce",
        "update_procedure_state",
        # retrieval / generation
        "retrieve",
        "search",
        "render_response",
        "build_renderer_prompt",
        # cold-start selection (ProcedureCatalog stays the only mapper)
        "issue_match",
        "cold_start_selection_model",
        # FSM / Tracker / transport ownership (later slices own those)
        "telegram",
        "tracker",
        "handoff",
        "notify",
        "conversation_mode",
    ):
        assert token not in code, token

    # No result/decision value objects are redefined here either (the ONE
    # outcome value object is the composition boundary's own return type).
    assert [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name not in {"SupportCore", "SupportTurnOutcome"}
    ] == []


def test_public_api_surface():
    assert set(sc.__all__) == {
        "NO_PROCEDURE",
        "PROCEDURE_NOT_FOUND",
        "TURN_EXECUTED",
        "SupportCore",
        "SupportTurnOutcome",
    }


# ============================================================
# LAZINESS / TESTABILITY OF THE COMPOSITION
# ============================================================

def test_construction_touches_no_database_configuration(monkeypatch):
    resolved: list[int] = []
    sessions = _FakeSessionFactory()

    def _factory():
        resolved.append(1)
        return sessions

    monkeypatch.setattr(sc, "get_session_factory", _factory)

    core = sc.SupportCore()  # real providers + real catalog, no doubles
    assert resolved == []  # nothing is resolved at construction time
    assert sessions.opened == 0

    executor = core.turn_executor
    assert isinstance(executor, SupportTurnExecutor)
    assert resolved == [1]  # resolved exactly once, lazily
    assert sessions.opened == 0


def test_construction_needs_no_credentials(monkeypatch):
    for variable in (
        "YANDEX_API_KEY",
        "YANDEX_FOLDER_ID",
    ):
        monkeypatch.delenv(variable, raising=False)

    core = sc.SupportCore()

    assert isinstance(core.analyzer_provider, YandexAnalyzerProvider)
    assert isinstance(core.knowledge_backend, YandexKnowledgeBackend)
    assert isinstance(core.response_provider, YandexExplanationProvider)


def test_every_dependency_is_injectable():
    catalog = ProcedureCatalog([_procedure()])
    sessions = _FakeSessionFactory()
    executor = object()

    core = sc.SupportCore(
        procedure_catalog=catalog,
        analyzer_provider="analyzer",
        knowledge_backend="knowledge",
        response_provider="response",
        session_factory=sessions,
        store_factory=_sentinel_store_factory,
        turn_executor=executor,
    )

    assert core.procedure_catalog is catalog
    assert core.analyzer_provider == "analyzer"
    assert core.knowledge_backend == "knowledge"
    assert core.response_provider == "response"
    assert core.session_factory is sessions
    assert core.store_factory is _sentinel_store_factory
    assert core.turn_executor is executor  # injected, not rebuilt
    assert sessions.opened == 0


# ============================================================
# 7-8. NOT WIRED ANYWHERE YET / server.py UNTOUCHED
# ============================================================

def _imports_support_core(path: Path) -> bool:
    """True when ``path`` has a REAL production dependency on ``support_core``.

    AST/import-aware: only actual ``import`` / ``from ... import`` statements
    count as wiring. A docstring, comment or string literal that merely
    *mentions* ``support_core`` / ``SupportCore`` is NOT a dependency (it wires
    nothing) and must never be flagged.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "support.support_core" or name.startswith(
                    "support.support_core."
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # ``from support.support_core import ...``
            if module == "support.support_core" or module.startswith(
                "support.support_core."
            ):
                return True
            # ``from .support_core import ...`` (relative, inside support/)
            if node.level >= 1 and (
                module == "support_core" or module.startswith("support_core.")
            ):
                return True
            # ``from support import support_core`` / ``from . import support_core``
            if module in ("support", ""):
                for alias in node.names:
                    if alias.name == "support_core":
                        return True
    return False


def test_no_runtime_module_imports_support_core_yet():
    core_path = Path(sc.__file__).resolve()
    root = _repo_root()

    offenders: list[str] = []
    for pattern in ("support/**/*.py", "yandex/**/*.py", "*.py"):
        for path in root.glob(pattern):
            resolved = path.resolve()
            if resolved == core_path or "__pycache__" in path.parts:
                continue
            if "tests" in path.parts:
                continue
            if _imports_support_core(path):
                offenders.append(str(path))

    assert offenders == []

    for relative in _RUNTIME_MODULES:
        path = root / relative
        if not path.exists():
            continue
        assert not _imports_support_core(path), relative


def test_server_py_is_untouched_by_this_slice():
    server = _repo_root() / "yandex/chat/server.py"
    assert server.exists()

    text = server.read_text(encoding="utf-8")
    assert "support_core" not in text
    assert "SupportCore" not in text
    assert "support.alice_analyzer_yandex" not in text
    assert "support.response_generator_yandex" not in text
    assert "YandexKnowledgeBackend" not in text
    assert "ProcedureCatalog" not in text


# ============================================================
# 9-15. TURN EXECUTION: DELEGATED COMPOSITION (Slice 11A.3b)
# ============================================================

class _RecordingSessionContext:
    """Session double that records the transaction boundary ordering."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self):
        self._events.append("session.open")
        return object()

    async def __aexit__(self, *exc_info):
        self._events.append("session.closed")
        return False


class _RecordingSessionFactory:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.opened = 0

    def __call__(self):
        self.opened += 1
        return _RecordingSessionContext(self._events)


class _ReadOnlyStore:
    """ProcedureStateStore double that can READ and must NEVER write.

    ``save_cas`` deliberately explodes: if SupportCore ever tried to persist
    ``ProcedureState`` itself, the test fails loudly instead of silently
    duplicating the executor's durable boundary.
    """

    def __init__(self, stored=None, events=None) -> None:
        self.stored = stored
        self.events = events if events is not None else []
        self.reads: list[str] = []

    async def read(self, conversation_id):
        self.reads.append(conversation_id)
        self.events.append("store.read")
        return self.stored

    async def save_cas(self, *args, **kwargs):
        raise AssertionError("SupportCore must never persist ProcedureState")


class _RecordingExecutor:
    """SupportTurnExecutor double: records the ONE delegated call."""

    def __init__(self, result=None, error=None, events=None) -> None:
        self.result = result
        self.error = error
        self.events = events if events is not None else []
        self.calls: list[dict] = []

    async def execute_persisted_turn(self, **kwargs):
        self.calls.append(kwargs)
        self.events.append("executor")
        if self.error is not None:
            raise self.error
        return self.result


class _RecordingCatalog:
    """Delegating catalog double recording which lookup was used."""

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


def _procedure_v2():
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 2,
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


def _other_procedure():
    """A DIFFERENT procedure whose issue label overlaps "black screen"."""
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN_FLICKER",
            "version": 1,
            "issue": "black screen flicker",
            "entry_step": "STEP_A",
            "terminators": ["ESCALATE", "RESOLVED"],
            "steps": [
                {
                    "id": "STEP_A",
                    "action": "Check the display cable.",
                    "evidence_query": "angioplus black screen flicker cable",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                }
            ],
        }
    )


def _persisted_turn_result(conversation_id="chat-1", revision=1):
    return PersistedTurnResult(
        conversation_id=conversation_id,
        turn=object(),
        state=ProcedureState(),
        revision=revision,
    )


def _stored_identity(procedure_id, procedure_version, *, revision=4):
    return PersistedProcedureState(
        state=ProcedureState(
            procedure_id=procedure_id,
            procedure_version=procedure_version,
        ),
        revision=revision,
    )


@dataclass
class _Harness:
    core: sc.SupportCore
    store: _ReadOnlyStore
    sessions: _RecordingSessionFactory
    executor: _RecordingExecutor
    events: list


def _harness(catalog, *, stored=None, executor=None) -> _Harness:
    events: list[str] = []
    store = _ReadOnlyStore(stored=stored, events=events)
    sessions = _RecordingSessionFactory(events)
    if executor is None:
        executor = _RecordingExecutor(
            result=_persisted_turn_result(), events=events
        )
    core = sc.SupportCore(
        procedure_catalog=catalog,
        analyzer_provider="analyzer-provider",
        knowledge_backend="knowledge-backend",
        response_provider="response-provider",
        session_factory=sessions,
        store_factory=lambda session: store,
        turn_executor=executor,
    )
    return _Harness(core, store, sessions, executor, events)


# ------------------------------------------------------------
# COLD START (unbound conversation)
# ------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_turn_cold_start_uses_the_catalog_and_runs_one_turn():
    procedure = _procedure()
    catalog = _RecordingCatalog(ProcedureCatalog([procedure]))
    h = _harness(catalog)

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen on boot"
    )

    assert outcome.status == sc.TURN_EXECUTED
    assert outcome.conversation_id == "chat-1"
    assert isinstance(outcome.persisted, PersistedTurnResult)

    # The catalog is the ONLY cold-start mapper, and it saw the raw message.
    assert catalog.select_calls == ["black screen on boot"]
    assert catalog.get_calls == []

    # ONE deferred turn, with the EXACT selected Procedure object.
    assert len(h.executor.calls) == 1
    assert h.executor.calls[0]["procedure"] is procedure
    assert len(h.store.reads) == 1  # identity read only; never a write


@pytest.mark.asyncio
async def test_handle_turn_cold_start_without_a_match_returns_no_procedure():
    catalog = _RecordingCatalog(ProcedureCatalog([_procedure()]))
    h = _harness(catalog)

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="unrelated coffee machine report"
    )

    assert outcome.status == sc.NO_PROCEDURE
    assert outcome.conversation_id == "chat-1"
    assert outcome.persisted is None
    assert h.executor.calls == []  # no procedure invented, no turn run
    assert catalog.get_calls == []


@pytest.mark.asyncio
async def test_handle_turn_cold_start_ambiguity_returns_no_procedure():
    catalog = ProcedureCatalog([_procedure(), _other_procedure()])
    # Precondition: the frozen catalog itself refuses to guess the ambiguity.
    assert catalog.select("black screen") is None

    recording = _RecordingCatalog(catalog)
    h = _harness(recording)
    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.NO_PROCEDURE
    assert outcome.persisted is None
    assert h.executor.calls == []  # never a best-match / tie-break guess


# ------------------------------------------------------------
# CONTINUATION (bound conversation): EXACT identity only
# ------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_turn_bound_state_uses_the_exact_persisted_identity():
    v1, v2 = _procedure(), _procedure_v2()
    catalog = ProcedureCatalog([v1, v2])
    h = _harness(
        catalog, stored=_stored_identity("BLACK_SCREEN", 2)
    )

    # Text that would NOT cold-start-select: identity comes from the DB.
    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="nothing recognisable here"
    )

    assert outcome.status == sc.TURN_EXECUTED
    assert h.executor.calls[0]["procedure"] is v2
    assert h.executor.calls[0]["procedure"] is catalog.get("BLACK_SCREEN", 2)


@pytest.mark.asyncio
async def test_handle_turn_bound_state_never_runs_cold_start_selection():
    catalog = _RecordingCatalog(
        ProcedureCatalog([_procedure(), _procedure_v2()])
    )
    h = _harness(catalog, stored=_stored_identity("BLACK_SCREEN", 2))

    # Even a message that WOULD have selected a procedure cold-start must not
    # reach ``select`` once the conversation is bound.
    await h.core.handle_turn(
        conversation_id="chat-1", message="black screen flicker"
    )

    assert catalog.select_calls == []
    assert catalog.get_calls == [("BLACK_SCREEN", 2)]


@pytest.mark.asyncio
async def test_handle_turn_uses_the_persisted_version_not_the_latest():
    v1, v2 = _procedure(), _procedure_v2()
    catalog = ProcedureCatalog([v1, v2])
    h = _harness(catalog, stored=_stored_identity("BLACK_SCREEN", 1))

    await h.core.handle_turn(conversation_id="chat-1", message="black screen")

    used = h.executor.calls[0]["procedure"]
    assert used is v1
    assert used.version == 1  # never silently upgraded to v2


@pytest.mark.asyncio
async def test_handle_turn_unknown_bound_procedure_is_a_safe_failure():
    catalog = _RecordingCatalog(ProcedureCatalog([_procedure()]))
    h = _harness(catalog, stored=_stored_identity("GHOST", 1))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.PROCEDURE_NOT_FOUND
    assert outcome.persisted is None
    assert h.executor.calls == []  # no procedure, no turn
    # Identity was NEVER guessed from the text, even though the text matches.
    assert catalog.select_calls == []
    assert catalog.get_calls == [("GHOST", 1)]


@pytest.mark.asyncio
async def test_handle_turn_wrong_bound_version_never_falls_back_to_latest():
    catalog = _RecordingCatalog(
        ProcedureCatalog([_procedure(), _procedure_v2()])
    )
    h = _harness(catalog, stored=_stored_identity("BLACK_SCREEN", 99))

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.PROCEDURE_NOT_FOUND
    assert h.executor.calls == []
    assert catalog.select_calls == []
    assert catalog.get_calls == [("BLACK_SCREEN", 99)]


@pytest.mark.asyncio
async def test_handle_turn_bound_without_a_version_is_not_guessed():
    catalog = _RecordingCatalog(ProcedureCatalog([_procedure()]))
    h = _harness(
        catalog, stored=_stored_identity("BLACK_SCREEN", None)
    )

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.PROCEDURE_NOT_FOUND
    assert h.executor.calls == []
    assert catalog.select_calls == []
    assert catalog.get_calls == []  # no version -> not even looked up


# ------------------------------------------------------------
# DELEGATION PAYLOAD: providers, history, verbatim result
# ------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_turn_forwards_the_composed_providers_and_history():
    catalog = ProcedureCatalog([_procedure()])
    h = _harness(catalog)
    history = [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "how can I help?"},
    ]

    await h.core.handle_turn(
        conversation_id="chat-1",
        message="black screen",
        history=history,
    )

    call = h.executor.calls[0]
    assert set(call) == {
        "conversation_id",
        "message",
        "procedure",
        "analyzer_provider",
        "history",
        "knowledge_backend",
        "response_provider",
    }
    assert call["conversation_id"] == "chat-1"
    assert call["message"] == "black screen"
    assert call["history"] is history  # opaque pass-through, unchanged
    assert call["analyzer_provider"] is h.core.analyzer_provider
    assert call["knowledge_backend"] is h.core.knowledge_backend
    assert call["response_provider"] is h.core.response_provider


@pytest.mark.asyncio
async def test_handle_turn_returns_the_executors_result_verbatim():
    persisted = _persisted_turn_result(revision=7)
    executor = _RecordingExecutor(result=persisted)
    h = _harness(ProcedureCatalog([_procedure()]), executor=executor)

    outcome = await h.core.handle_turn(
        conversation_id="chat-1", message="black screen"
    )

    assert outcome.status == sc.TURN_EXECUTED
    assert outcome.persisted is persisted  # not rebuilt / re-wrapped


# ------------------------------------------------------------
# CONFLICT: no retry, no overwrite, no new vocabulary
# ------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_turn_propagates_conflict_without_retry_or_overwrite():
    executor = _RecordingExecutor(error=ProcedureStateConflict("cas miss"))
    h = _harness(ProcedureCatalog([_procedure()]), executor=executor)

    with pytest.raises(ProcedureStateConflict):
        await h.core.handle_turn(
            conversation_id="chat-1", message="black screen"
        )

    # Exactly ONE attempt: no automatic retry, no provider re-run.
    assert len(executor.calls) == 1
    assert len(h.store.reads) == 1
    assert h.sessions.opened == 1


# ------------------------------------------------------------
# TRANSACTION BOUNDARY: still owned by the executor
# ------------------------------------------------------------

@pytest.mark.asyncio
async def test_transaction_boundary_stays_owned_by_the_executor():
    h = _harness(ProcedureCatalog([_procedure()]))

    await h.core.handle_turn(conversation_id="chat-1", message="black screen")

    # SupportCore's single identity read is OPENED then CLOSED before the
    # executor is called; SupportCore opens no OTHER session (the executor
    # owns the durable read -> process -> CAS write lifecycle itself).
    assert h.events == [
        "session.open",
        "store.read",
        "session.closed",
        "executor",
    ]
    assert h.sessions.opened == 1


def test_support_core_opens_exactly_one_session_and_never_commits():
    tree = _tree()

    # ONE ``async with session:`` in the whole module: the identity read.
    assert len([n for n in ast.walk(tree) if isinstance(n, ast.AsyncWith)]) == 1

    code = _code_only_text(tree)
    for token in ("commit", "rollback", "flush", "save_cas", "add_all"):
        assert token not in code, token


def test_support_core_does_not_call_the_orchestrator_or_the_router():
    tree = _tree()
    imported = _module_level_imports(tree)

    assert "support.orchestrator" not in imported
    assert "support.router" not in imported
    assert "support.turn_executor" in imported  # positive control

    code = _code_only_text(tree)
    for token in ("process_support_turn", "route", "update_procedure_state"):
        assert token not in code, token


# ------------------------------------------------------------
# THE OUTCOME VALUE OBJECT
# ------------------------------------------------------------

def test_turn_outcome_statuses_are_stable_strings():
    assert sc.TURN_EXECUTED == "TURN_EXECUTED"
    assert sc.NO_PROCEDURE == "NO_PROCEDURE"
    assert sc.PROCEDURE_NOT_FOUND == "PROCEDURE_NOT_FOUND"
    assert len({sc.TURN_EXECUTED, sc.NO_PROCEDURE, sc.PROCEDURE_NOT_FOUND}) == 3


def test_turn_outcome_enforces_its_shape():
    persisted = _persisted_turn_result()

    done = sc.SupportTurnOutcome(
        status=sc.TURN_EXECUTED, conversation_id="chat-1", persisted=persisted
    )
    assert done.persisted is persisted

    for failing_status in (sc.NO_PROCEDURE, sc.PROCEDURE_NOT_FOUND):
        outcome = sc.SupportTurnOutcome(
            status=failing_status, conversation_id="chat-1"
        )
        assert outcome.persisted is None
        with pytest.raises(ValueError):
            sc.SupportTurnOutcome(
                status=failing_status,
                conversation_id="chat-1",
                persisted=persisted,
            )

    with pytest.raises(ValueError):
        sc.SupportTurnOutcome(status="TURN_EXECUTED", conversation_id="chat-1")

    with pytest.raises(ValueError):
        sc.SupportTurnOutcome(status="handed_off", conversation_id="chat-1")
