"""
Alice Semantic Analyzer -- turn ONE user message into a SemanticObservation.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

WHAT THIS MODULE IS (and is NOT)
--------------------------------
This module is the single, provider-agnostic boundary that converts a raw
user message (plus recent conversation context and a small, ALLOWLISTED
view of the current procedure) into a routing-free
:class:`~support.observation.SemanticObservation`.

It is deliberately split into three PURE stages so it can be unit-tested
with zero network access:

  * :func:`build_allowed_steps`  -- project a :class:`Procedure` down to a
    minimal ``[{step_id, action}]`` allowlist. Transitions, terminators,
    evidence queries and any File Search view are NEVER exposed to the LLM.
  * :func:`build_analyzer_prompt` -- render the strict JSON-only prompt.
  * :func:`parse_semantic_output` -- STRICTLY validate the raw JSON text
    the provider returned and build the :class:`SemanticObservation`.

:func:`analyze` glues the three together around an INJECTED provider
callable. The provider is the ONLY thing that performs I/O; it is supplied
by the caller (a cloud test today, future production wiring later). This
module therefore imports no HTTP client, no ``yandex`` package and no DB,
and it makes no File Search call of any kind.

PARSER CONTRACT (two hard rules)
--------------------------------
1. NO SEMANTIC ATTRIBUTION. The parser performs validation, allowlist
   enforcement and projection ONLY. It never performs natural-language
   understanding. In particular it does NOT bind an ambiguous reference
   ("не помогло", "I did that") to ``last_recommended_step``: resolving
   such a referent is Alice's job (she receives the last recommendation as
   READ-ONLY context). If Alice emits ``{"step_id": null, "outcome":
   "failed"}`` the parser preserves it EXACTLY as
   ``ReportedAction(step_id=None, outcome=StepOutcome.FAILED)``.

2. STRICT SCHEMA; NO SILENT DRIFT. Any structural violation -- malformed
   JSON, a missing required field, an EXTRA/unknown field (top-level or
   per-action), a wrong type, or an invalid enum value -- is a hard
   rejection. The rejection is uniform: it returns the single fail-safe
   observation (:func:`_fail_safe_observation`). An invented ``step_id``
   (one not in the allowlist) is also rejected as a step: it is preserved
   with ``step_id=None`` so the downstream reducer flags clarification.

The fail-safe observation is deterministic and has been verified against
the FROZEN Slice 2 reducer + Slice 3 router:

  * reducer: an action with ``step_id=None`` sets
    ``pending_clarification=True`` (unmappable-action branch);
  * router: gate 7 on ``pending_clarification`` yields
    ``CLARIFY(PENDING_CLARIFICATION)``.

So a parser failure, a provider exception, a missing procedure or no
allowlist all converge on the SAME safe outcome: ask for clarification --
never a silent ANSWER and never a spurious ESCALATE.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Sequence

from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    RequestKind,
    SemanticObservation,
    StepOutcome,
)
from support.procedure import Procedure


# ============================================================
# LEGACY-LEAKAGE GUARD (anti-regression vocabulary)
# ============================================================
# These fields belong to the LEGACY string-priority cascade
# (``apply_support_policy``). The analyzer must never emit, request or
# accept them. Kept here as an explicit, testable deny-list.
LEGACY_FORBIDDEN_FIELDS = frozenset(
    {
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "documentation_requires_support",
        "action_match",
        "previous_recommended_action",
        "needs_clarification",
        "reported_user_action",
    }
)


# ============================================================
# CLOSED VOCABULARIES / SCHEMA SHAPE
# ============================================================

# Top-level keys the analyzer output MUST contain -- exactly these, no more.
_TOP_LEVEL_FIELDS = (
    "scope",
    "reported_actions",
    "problem_status",
    "human_requested",
    "clarification_facts",
    "new_issue",
    "request_kind",
)

# Per-action keys -- exactly these, no more.
_ACTION_FIELDS = ("step_id", "outcome")

_SCOPE_VALUES = {s.value for s in ObservationScope}
_PROBLEM_VALUES = {p.value for p in ProblemStatus}
_OUTCOME_VALUES = {o.value for o in StepOutcome}
_REQUEST_KIND_VALUES = {k.value for k in RequestKind}


# ============================================================
# FAIL-SAFE OBSERVATION
# ============================================================

def _fail_safe_observation() -> SemanticObservation:
    """The single, uniform safe observation for ANY analyzer failure.

    A message we could not read is reported as in-scope-but-unclear with one
    UNMAPPABLE action (``step_id=None``). Verified against frozen Slice 2/3:
    the reducer turns the unmappable action into ``pending_clarification``,
    and router gate 7 turns that into ``CLARIFY(PENDING_CLARIFICATION)``.
    """
    return SemanticObservation(
        scope=ObservationScope.UNCLEAR,
        reported_actions=(ReportedAction(step_id=None, outcome=None),),
        problem_status=None,
        human_requested=False,
        clarification_facts=(),
        new_issue=None,
        request_kind=RequestKind.UNCLEAR,
    )


# ============================================================
# STAGE 1 -- ALLOWLISTED STEP VIEW
# ============================================================

def build_allowed_steps(
    procedure: Optional[Procedure],
) -> tuple[dict[str, str], ...]:
    """Project a procedure to the MINIMAL view the LLM may see.

    Returns a tuple of ``{"step_id": ..., "action": ...}`` mappings ONLY.

    Deliberately excluded (must never reach the model):
      * ``transitions`` -- the model must not walk the graph; the router
        does that deterministically after the fact.
      * ``terminators``.
      * ``evidence_query`` -- no File Search is performed in this slice.
    """
    if procedure is None:
        return ()
    return tuple(
        {"step_id": step.id, "action": step.action}
        for step in procedure.steps.values()
    )


# ============================================================
# STAGE 2 -- PROMPT
# ============================================================

_SYSTEM_RULES = """\
Ты — этап СЕМАНТИЧЕСКОГО АНАЛИЗА системы технической поддержки AngioPlus.

Твоя задача — ТОЛЬКО извлечь факты из текущего сообщения пользователя
(с учётом истории диалога). Ты НЕ принимаешь решение поддержки.

ЗАПРЕЩЕНО: выбирать ANSWER / CLARIFY / ESCALATE, решать о передаче
специалисту, управлять маршрутизацией или конечным автоматом. Возвращай
только семантические факты в строгом JSON.

request_kind — это ФАКТ о сообщении, а НЕ решение о маршрутизации.
НИКОГДА не используй request_kind, чтобы закодировать: выбор процедуры,
передачу специалисту, возможность системы ответить, наличие знаний или
необходимость поиска.

ПРО ССЫЛКИ В КОНТЕКСТЕ:
Если пользователь говорит «не помогло», «я это уже делал», «сделал это»
и подобное — САМ сопоставь такую ссылку с последней рекомендованной
инструкцией из истории диалога. Если инструкция определяется достаточно
ясно, укажи её step_id и outcome. Если определить нельзя — верни
step_id = null.

ПРО ОТСУТСТВИЕ АКТИВНОЙ ПРОЦЕДУРЫ:
Если список разрешённых шагов пуст, ты не можешь сопоставить действие с
конкретным шагом. Тогда ВСЕГДА возвращай step_id = null (но можешь
заполнить outcome, если пользователь явно сообщил результат).

step_id ОБЯЗАН быть одним из разрешённых значений или null. Никогда не
придумывай новый step_id.

Шаг, которого нет в списке разрешённых, — недействителен: верни
step_id = null для него.

ПОЛЯ (ровно эти, лишних быть не должно):

  scope: "in_scope" | "out_of_scope" | "unclear"
    - "out_of_scope": сообщение НЕ про AngioPlus Core.
    - "in_scope": сообщение про AngioPlus Core, его использование, ошибки,
      установку или поддержку.
    - "unclear": тему определить нельзя даже с учётом истории.
    Такие вопросы, как цвет банана, погода, стихи, арифметика — out_of_scope.

  reported_actions: массив объектов {step_id, outcome}
    - step_id: строка из списка разрешённых или null.
    - outcome: "failed" | "succeeded" | "skipped" | "done_unknown" | null.
      * "failed"      — пользователь выполнил шаг, проблема осталась.
      * "succeeded"   — пользователь выполнил шаг, проблема исчезла.
      * "skipped"     — пользователь говорит, что шаг уже был выполнен
                        ранее, или не может/не относится к делу. Это
                        НЕИНФОРМАТИВНЫЙ результат: он НЕ означает успех
                        и НЕ означает неудачу.
      * "done_unknown"— пользователь выполнил шаг, но результат пока
                        неясен.
    - Если пользователь сообщает, что проблема осталась, НО не называет
      и не подразумевает конкретный шаг, верни
      {"step_id": null, "outcome": "failed"}.

  problem_status: "unknown" | "ongoing" | "resolved" | null

  human_requested: true | false — пользователь прямо просит человека.

  clarification_facts: массив строк — свободные факты для уточнения.

  new_issue: строковый НОВЫЙ проблемы или null. Заполняй ТОЛЬКО когда
    пользователь поднимает ДРУГУЮ проблему. Продолжение той же проблемы
    (например, повторный повтор или «не помогло») — это НЕ new_issue.

  request_kind: "problem" | "information" | "unclear"
    - "problem": пользователь сообщает о симптоме, неисправности, сбое,
      проблеме подключения/настройки, неожиданном поведении ИЛИ просит
      помочь устранить/настроить/подключить что-либо.
    - "information": пользователь просит фактическую/справочную
      информацию о продукте или документации и НЕ сообщает о текущей
      проблеме, которую нужно устранять.
    - "unclear": безопасно определить нельзя.
    Примеры: «Какие версии Windows поддерживает AngioPlus?» -> "information";
      «AngioPlus не запускается на Windows 11» -> "problem";
      «PACS не подключается» -> "problem"; «Что такое PACS?» -> "information";
      «Не работает» -> "problem";
      «Как подключить AngioPlus к PACS?» -> "problem" (просьба настроить).
    НИКОГДА не выбирай "information" только потому, что problem_status
    не указан.

Отвечай РОВНО одним JSON-объектом указанной формы, без markdown.
"""


def _render_allowed_steps(allowed_steps: Sequence[Mapping[str, str]]) -> str:
    if not allowed_steps:
        return "(нет доступных шагов)"
    lines = [
        f'- step_id: "{s["step_id"]}" — {s["action"]}'
        for s in allowed_steps
    ]
    return "\n".join(lines)


def _render_history(history: Optional[Sequence[Mapping[str, str]]]) -> str:
    parts = [
        f"{item.get('role')}: {item.get('text')}"
        for item in (history or [])
        if item.get("role") in {"user", "assistant", "operator"}
        and item.get("text")
    ]
    return "\n".join(parts) if parts else "(нет предыдущей переписки)"


def build_analyzer_prompt(
    *,
    message: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
    allowed_steps: Sequence[Mapping[str, str]] = (),
    last_recommendation: Optional[str] = None,
) -> str:
    """Render the STRICT JSON-only analyzer prompt.

    ``last_recommendation`` is provided as READ-ONLY CONTEXT so Alice can
    resolve an ambiguous reference ("не помогло"). It is NOT a procedure
    edge and it does not authorise an action: the router still walks the
    declarative transition table itself.
    """
    last_rec_text = last_recommendation or "(нет)"
    return (
        f"{_SYSTEM_RULES}\n"
        "=== КОНТЕКСТ ===\n"
        f"ПОСЛЕДНЯЯ РЕКОМЕНДАЦИЯ АССИСТЕНТА (только для понимания ссылок, "
        f"не является действием): {last_rec_text}\n\n"
        "ИСТОРИЯ ДИАЛОГА:\n"
        f"{_render_history(history)}\n\n"
        "РАЗРЕШЁННЫЕ ШАГИ (используй step_id ТОЛЬКО из этого списка "
        "или null):\n"
        f"{_render_allowed_steps(allowed_steps)}\n\n"
        "ТЕКУЩЕЕ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n"
        f"{message}\n"
    )


# ============================================================
# STAGE 3 -- STRICT PARSER
# ============================================================

class _Reject(Exception):
    """Internal marker: the raw output violated the strict schema."""


def _require_exact_keys(
    obj: Mapping[str, Any], required: Sequence[str], *, where: str
) -> None:
    actual = set(obj)
    expected = set(required)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise _Reject(f"{where}: missing field(s) {sorted(missing)}")
    if extra:
        # EXTRA fields are a hard rejection (no silent drift), and any
        # legacy-cascade field is called out explicitly.
        leaked = sorted(extra & LEGACY_FORBIDDEN_FIELDS)
        if leaked:
            raise _Reject(f"{where}: forbidden legacy field(s) {leaked}")
        raise _Reject(f"{where}: unexpected field(s) {sorted(extra)}")


def _parse_outcome(raw: Any, *, where: str) -> Optional[StepOutcome]:
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in _OUTCOME_VALUES:
        raise _Reject(f"{where}: invalid outcome {raw!r}")
    return StepOutcome(raw)


def _parse_one_action(
    raw: Any, *, allowed_ids: frozenset[str], index: int
) -> ReportedAction:
    where = f"reported_actions[{index}]"
    if not isinstance(raw, dict):
        raise _Reject(f"{where}: must be an object")

    _require_exact_keys(raw, _ACTION_FIELDS, where=where)

    step_id_raw = raw["step_id"]
    if step_id_raw is not None and not isinstance(step_id_raw, str):
        raise _Reject(f"{where}.step_id: must be a string or null")

    outcome = _parse_outcome(raw["outcome"], where=f"{where}.outcome")

    # The parser does NO semantic attribution: a null step_id is preserved
    # verbatim. An invented step_id is NOT trusted as a step -- it is
    # downgraded to null (the reducer will then flag clarification).
    if step_id_raw is None:
        step_id: Optional[str] = None
    elif step_id_raw in allowed_ids:
        step_id = step_id_raw
    else:
        step_id = None

    return ReportedAction(step_id=step_id, outcome=outcome)


def _parse_reported_actions(
    raw: Any, *, allowed_ids: frozenset[str]
) -> tuple[ReportedAction, ...]:
    if not isinstance(raw, list):
        raise _Reject("reported_actions: must be a list")
    return tuple(
        _parse_one_action(item, allowed_ids=allowed_ids, index=i)
        for i, item in enumerate(raw)
    )


def _parse_scope(raw: Any) -> ObservationScope:
    if not isinstance(raw, str) or raw not in _SCOPE_VALUES:
        raise _Reject(f"scope: invalid value {raw!r}")
    return ObservationScope(raw)


def _parse_problem_status(raw: Any) -> Optional[ProblemStatus]:
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in _PROBLEM_VALUES:
        raise _Reject(f"problem_status: invalid value {raw!r}")
    return ProblemStatus(raw)


def _parse_request_kind(raw: Any) -> RequestKind:
    if not isinstance(raw, str) or raw not in _REQUEST_KIND_VALUES:
        raise _Reject(f"request_kind: invalid value {raw!r}")
    return RequestKind(raw)


def _parse_bool(raw: Any, *, where: str) -> bool:
    # bool is a subclass of int; require the real bool.
    if not isinstance(raw, bool):
        raise _Reject(f"{where}: must be a boolean")
    return raw


def _parse_facts(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise _Reject("clarification_facts: must be a list")
    facts: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise _Reject(f"clarification_facts[{i}]: must be a string")
        facts.append(item)
    return tuple(facts)


def _parse_new_issue(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise _Reject("new_issue: must be a string or null")
    stripped = raw.strip()
    return stripped or None


def parse_semantic_output(
    raw_text: str,
    *,
    allowed_steps: Sequence[Mapping[str, str]] = (),
) -> SemanticObservation:
    """STRICTLY validate raw analyzer JSON and build the observation.

    Any schema violation (bad JSON, missing/extra field, wrong type, bad
    enum) returns the uniform fail-safe observation. An invented step_id is
    downgraded to ``None`` (never silently dropped, never trusted).
    """
    allowed_ids = frozenset(s["step_id"] for s in allowed_steps)

    if not isinstance(raw_text, str):
        return _fail_safe_observation()

    try:
        data = json.loads(raw_text)
    except (ValueError, TypeError):
        return _fail_safe_observation()

    if not isinstance(data, dict):
        return _fail_safe_observation()

    try:
        _require_exact_keys(data, _TOP_LEVEL_FIELDS, where="root")
        scope = _parse_scope(data["scope"])
        reported_actions = _parse_reported_actions(
            data["reported_actions"], allowed_ids=allowed_ids
        )
        problem_status = _parse_problem_status(data["problem_status"])
        human_requested = _parse_bool(
            data["human_requested"], where="human_requested"
        )
        facts = _parse_facts(data["clarification_facts"])
        new_issue = _parse_new_issue(data["new_issue"])
        request_kind = _parse_request_kind(data["request_kind"])
    except _Reject:
        return _fail_safe_observation()

    return SemanticObservation(
        scope=scope,
        reported_actions=reported_actions,
        problem_status=problem_status,
        human_requested=human_requested,
        clarification_facts=facts,
        new_issue=new_issue,
        request_kind=request_kind,
    )


# ============================================================
# PUBLIC ENTRY POINT (provider-injected; the ONLY I/O boundary)
# ============================================================

# A provider receives (prompt, history) and returns the raw JSON TEXT the
# model produced. It is supplied by the caller; this module never creates
# one and never performs network I/O itself.
AnalyzerProvider = Callable[[str, Sequence[Mapping[str, str]]], str]


def analyze(
    *,
    message: str,
    history: Optional[Sequence[Mapping[str, str]]] = None,
    procedure: Optional[Procedure] = None,
    last_recommendation: Optional[str] = None,
    provider: AnalyzerProvider,
) -> SemanticObservation:
    """Build the prompt, call ``provider``, and strictly parse the result.

    Any provider exception OR any schema violation converges on the SAME
    deterministic fail-safe observation (-> CLARIFY at the router).
    """
    allowed_steps = build_allowed_steps(procedure)
    prompt = build_analyzer_prompt(
        message=message,
        history=history,
        allowed_steps=allowed_steps,
        last_recommendation=last_recommendation,
    )

    try:
        raw_text = provider(prompt, list(history or []))
    except Exception:
        # A broken/absent provider must never crash the pipeline.
        return _fail_safe_observation()

    return parse_semantic_output(raw_text, allowed_steps=allowed_steps)


__all__ = [
    "LEGACY_FORBIDDEN_FIELDS",
    "AnalyzerProvider",
    "build_allowed_steps",
    "build_analyzer_prompt",
    "parse_semantic_output",
    "analyze",
]
