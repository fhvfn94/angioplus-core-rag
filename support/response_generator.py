"""
Deterministic Response Generator (Slice 6A) -- renderer ONLY.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

SCOPE (Slice 6A)
----------------
This module turns an ALREADY-MADE, IMMUTABLE
:class:`~support.router.RoutingDecision` into the user-facing text for
every router action, DETERMINISTICALLY and WITHOUT any LLM. It is PURE:
no network, no provider, no Knowledge retrieval, no DB, no framework, no
conversation history.

RESPONSIBILITY SPLIT
--------------------
  Router (support.router)        Response Generator (this module)
  ----------------------------   ----------------------------------
  decides WHAT to do next        renders WHAT was decided
  owns the procedure cursor      never inspects the procedure graph
  is the only decision maker     never returns or changes a RouteAction
  reads declarative edges        reads only the selected step's action

The generator is a CONSUMER of a decision, never a decision maker:

  * it never returns or mutates a :class:`RouteAction`;
  * it never inspects a step's outgoing edges;
  * it never retrieves Knowledge, calls a provider, or reads history;
  * for ``RECOMMEND_STEP`` it renders EXACTLY the selected step's
    procedure-owned ``action`` and nothing else (no extra troubleshooting).

ONLY ``RouteAction.ESCALATE`` may render handoff / escalation wording.
``INTERNAL_ERROR`` (an unusable decision) is deliberately NEUTRAL: it must
NOT claim any handoff, escalation, ticket, or routing outcome, because the
generator is NOT the router.

Knowledge- and provider-aware rendering (including evidence association
and an LLM) belongs to LATER checkpoints and is deliberately absent here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from support.procedure import Procedure
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)


# ============================================================
# CLOSED VOCABULARY
# ============================================================

class _StrEnum(str, Enum):
    """Python 3.10 compatible str-enum (values are plain strings)."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.value)


class ResponseStatus(_StrEnum):
    """The provenance of a rendered response.

    * ``RENDERED``       -- the INTENDED path produced the text (a
                            deterministic template, or the selected
                            procedure-owned step action).
    * ``FALLBACK``       -- a safe deterministic text was substituted
                            because the preferred text could not be proven
                            safe to show.
    * ``INTERNAL_ERROR`` -- the decision could not be rendered at all
                            (an invariant violation, e.g. a missing or
                            unresolvable step). The text stays NEUTRAL and
                            never claims a routing/handoff outcome.
    """

    RENDERED = "rendered"
    FALLBACK = "fallback"
    INTERNAL_ERROR = "internal_error"


# ============================================================
# DETERMINISTIC TEMPLATES (no LLM, no provider)
# ============================================================

# CLARIFY -- ask for missing information. NEVER a handoff.
_CLARIFY_TEMPLATES = {
    ClarifyReason.PENDING_CLARIFICATION: (
        "Уточните, пожалуйста, ответ на предыдущий вопрос — "
        "это поможет продолжить."
    ),
    ClarifyReason.NO_PROCEDURE: (
        "Опишите, пожалуйста, подробнее, что именно происходит "
        "и на каком шаге."
    ),
    ClarifyReason.NEW_ISSUE_RESEELECT: (
        "Похоже, речь о другой проблеме. Опишите её, пожалуйста, подробнее."
    ),
    ClarifyReason.STEP_OUTCOME_UNKNOWN: (
        "Подскажите, что произошло после предыдущего шага — "
        "помогло, не помогло или не удалось выполнить?"
    ),
}
_CLARIFY_DEFAULT_TEXT = (
    "Уточните, пожалуйста, детали — это поможет продолжить."
)

# ESCALATE -- the ONLY place handoff / escalation wording is allowed.
_ESCALATE_TEMPLATES = {
    EscalateReason.USER_REQUESTED_HUMAN: (
        "Хорошо, передаю ваш вопрос специалисту технической поддержки."
    ),
    EscalateReason.PROCEDURE_TERMINAL: (
        "К сожалению, предложенные шаги не помогли. "
        "Передаю обращение специалисту технической поддержки."
    ),
    EscalateReason.STEPS_EXHAUSTED: (
        "Проверенные шаги исчерпаны. "
        "Передаю обращение специалисту технической поддержки."
    ),
    EscalateReason.INTERNAL_INCONSISTENCY: (
        "Не удалось корректно продолжить диагностику. "
        "Передаю ваш вопрос специалисту технической поддержки."
    ),
}
_ESCALATE_DEFAULT_TEXT = (
    "Передаю ваш вопрос специалисту технической поддержки."
)

# RESOLVE -- closing, no handoff, no troubleshooting.
_RESOLVE_TEXT = (
    "Отлично, рада, что помогло! Если понадобится помощь снова — обращайтесь."
)

# OUT_OF_SCOPE -- declines, no handoff claim (the generator is not the
# router, so it must not announce a transfer).
_OUT_OF_SCOPE_TEXT = "К сожалению, этот вопрос вне того, с чем я могу помочь."

# INTERNAL_ERROR -- NEUTRAL. MUST NOT claim any handoff / escalation /
# ticket / routing outcome; it must not imply a transfer to a human.
_INTERNAL_ERROR_TEXT = (
    "Не удалось корректно продолжить диагностику. "
    "Пожалуйста, повторите запрос или попробуйте ещё раз."
)

# FALLBACK for a procedure-owned step action that cannot be shown safely
# (e.g. it contains an external-contact directive). Neutral: no action,
# no handoff.
_UNSAFE_STEP_TEXT = (
    "Не удалось подобрать безопасную формулировку шага. "
    "Пожалуйста, повторите запрос или попробуйте ещё раз."
)


# ============================================================
# EXTERNAL-CONTACT DIRECTIVE GUARD
# ============================================================
# This guard protects the PROCEDURE-OWNED fallback text (a step's action)
# from being shown to the user when it directs the user to an EXTERNAL
# entity (manufacturer / distributor / supplier). It blocks a DIRECTIVE,
# never a mere factual mention.
#
# Dangerous pattern = contact intent AND external entity WITHIN THE SAME
# SENTENCE. A factual mention ("According to the manufacturer ...") with no
# contact intent is explicitly allowed.

_CONTACT_INTENT = (
    "свяжитесь",
    "связаться",
    "обратитесь",
    "обратиться",
    "обращайтесь",
    "позвоните",
    "позвонить",
    "напишите",
    "написать",
    "запросите",
    "contact",
    "reach out",
    "get in touch",
    "email",
)

_EXTERNAL_ENTITY = (
    "производител",
    "изготовител",
    "дистрибьютор",
    "поставщик",
    "shanghai pulse",
    "medical technology",
    "manufacturer",
    "distributor",
    "supplier",
    "vendor",
)

_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")


def _iter_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _is_external_contact_directive(text: str) -> bool:
    """True iff some single sentence is BOTH a contact intent and an
    external-entity reference. Deterministic; conservative (a factual
    mention alone is never blocked)."""
    for sentence in _iter_sentences(text):
        low = sentence.lower()
        if any(token in low for token in _CONTACT_INTENT) and any(
            token in low for token in _EXTERNAL_ENTITY
        ):
            return True
    return False


# ============================================================
# RESULT VALUE OBJECT
# ============================================================

@dataclass(frozen=True)
class ResponseResult:
    """The generator's immutable rendering of ONE router decision.

    It carries NO :class:`RouteAction`: the generator can neither return
    nor change a routing decision. ``text`` is ALWAYS a non-blank,
    user-facing string. ``step_id`` identifies the rendered step (or
    ``None``). ``reason`` is INTERNAL diagnostic metadata (tests /
    observability) and is NEVER rendered to the user.
    """

    status: ResponseStatus
    text: str
    step_id: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResponseStatus):
            raise ValueError("status must be a ResponseStatus.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-blank string.")
        if self.step_id is not None and (
            not isinstance(self.step_id, str) or not self.step_id.strip()
        ):
            raise ValueError("step_id must be None or a non-blank string.")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("reason must be None or a string.")


# ============================================================
# STEP INSTRUCTION (procedure-owned, deterministic)
# ============================================================

def _step_instruction(step) -> str:
    """The selected step's procedure-owned action, whitespace-normalized.

    Rendered verbatim: the generator invents NO additional troubleshooting
    action and appends NOTHING.
    """
    return " ".join(step.action.split())


# ============================================================
# PER-ACTION RENDERERS
# ============================================================

def _render_recommend_step(
    decision: RoutingDecision,
    procedure: Optional[Procedure],
) -> ResponseResult:
    """Render EXACTLY the selected step's procedure-owned action.

    Reads only ``decision.next_step`` and ``procedure.steps``. Never
    inspects the procedure graph, never retrieves Knowledge, never calls a
    provider, never reads history.
    """
    step_id = decision.next_step
    if not isinstance(step_id, str) or not step_id.strip():
        return ResponseResult(
            status=ResponseStatus.INTERNAL_ERROR,
            text=_INTERNAL_ERROR_TEXT,
            reason="missing_next_step",
        )
    step_id = step_id.strip()

    if procedure is None:
        return ResponseResult(
            status=ResponseStatus.INTERNAL_ERROR,
            text=_INTERNAL_ERROR_TEXT,
            reason="missing_procedure",
        )

    step = procedure.steps.get(step_id)
    if step is None:
        return ResponseResult(
            status=ResponseStatus.INTERNAL_ERROR,
            text=_INTERNAL_ERROR_TEXT,
            reason="unknown_next_step",
        )

    text = _step_instruction(step)
    if _is_external_contact_directive(text):
        return ResponseResult(
            status=ResponseStatus.FALLBACK,
            text=_UNSAFE_STEP_TEXT,
            step_id=step_id,
            reason="contact_directive_in_step_action",
        )

    return ResponseResult(
        status=ResponseStatus.RENDERED,
        text=text,
        step_id=step_id,
    )


def _render_clarify(decision: RoutingDecision) -> ResponseResult:
    text = _CLARIFY_TEMPLATES.get(decision.clarify_reason)
    if text is None:
        return ResponseResult(
            status=ResponseStatus.RENDERED,
            text=_CLARIFY_DEFAULT_TEXT,
            reason="unknown_clarify_reason",
        )
    return ResponseResult(status=ResponseStatus.RENDERED, text=text)


def _render_escalate(decision: RoutingDecision) -> ResponseResult:
    text = _ESCALATE_TEMPLATES.get(decision.escalate_reason)
    if text is None:
        return ResponseResult(
            status=ResponseStatus.RENDERED,
            text=_ESCALATE_DEFAULT_TEXT,
            reason="unknown_escalate_reason",
        )
    return ResponseResult(status=ResponseStatus.RENDERED, text=text)


# ============================================================
# PUBLIC API
# ============================================================

def render_response(
    decision: RoutingDecision,
    *,
    procedure: Optional[Procedure] = None,
) -> ResponseResult:
    """Deterministically render ONE router decision. Total, pure, no I/O.

    This is the ONLY entry point. It never raises for a valid decision and
    never changes or returns a :class:`RouteAction`.

    * ``RECOMMEND_STEP`` -- renders exactly ``procedure.steps[next_step]
      .action``; a missing / unresolvable step is ``INTERNAL_ERROR``.
    * ``CLARIFY``        -- deterministic per-``clarify_reason`` template.
    * ``ESCALATE``       -- deterministic per-``escalate_reason`` template
      (the ONLY handoff wording).
    * ``RESOLVE``        -- deterministic closing template.
    * ``OUT_OF_SCOPE``   -- deterministic decline template.
    """
    if decision is None:
        return ResponseResult(
            status=ResponseStatus.INTERNAL_ERROR,
            text=_INTERNAL_ERROR_TEXT,
            reason="missing_decision",
        )

    action = decision.action

    if action is RouteAction.RECOMMEND_STEP:
        return _render_recommend_step(decision, procedure)
    if action is RouteAction.CLARIFY:
        return _render_clarify(decision)
    if action is RouteAction.ESCALATE:
        return _render_escalate(decision)
    if action is RouteAction.RESOLVE:
        return ResponseResult(status=ResponseStatus.RENDERED, text=_RESOLVE_TEXT)
    if action is RouteAction.OUT_OF_SCOPE:
        return ResponseResult(
            status=ResponseStatus.RENDERED, text=_OUT_OF_SCOPE_TEXT
        )

    # Defensive: an action outside the closed vocabulary. Never raise.
    return ResponseResult(
        status=ResponseStatus.INTERNAL_ERROR,
        text=_INTERNAL_ERROR_TEXT,
        reason="unknown_action",
    )


__all__ = [
    "ResponseStatus",
    "ResponseResult",
    "render_response",
]
