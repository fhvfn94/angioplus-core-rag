# -*- coding: utf-8 -*-
"""Slice 11B.2 -- deterministic projection of GLOBAL terminals (PURE, offline).

The projection maps the closed :class:`~support.global_policy.GlobalAction`
terminal verdict into the EXISTING deterministic user-facing wording, so the
dispatcher can terminate a turn WITHOUT importing the legacy Router itself.
It owns NO wording and NO decision: it only re-uses the frozen generator text
for an already-frozen decision.

NO network, NO credentials, NO DB, NO LLM, NO Procedure, NO Knowledge.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from support import turn_projection as tp
from support.global_policy import GlobalAction
from support.response_generator import render_response
from support.router import EscalateReason, RouteAction, RoutingDecision


def _imports_module(path: Path, module: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == module for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "") == module:
            return True
    return False


# ============================================================
# 1. TERMINALS PROJECT TO NON-BLANK, DETERMINISTIC WORDING
# ============================================================

@pytest.mark.parametrize(
    "action", [GlobalAction.ESCALATE, GlobalAction.OUT_OF_SCOPE]
)
def test_terminal_projects_to_non_blank_text(action):
    text = tp.project_global_terminal(action)
    assert isinstance(text, str)
    assert text.strip()


def test_projection_is_deterministic_and_terminal_specific():
    escalate = tp.project_global_terminal(GlobalAction.ESCALATE)
    out_of_scope = tp.project_global_terminal(GlobalAction.OUT_OF_SCOPE)

    assert escalate == tp.project_global_terminal(GlobalAction.ESCALATE)
    assert out_of_scope == tp.project_global_terminal(GlobalAction.OUT_OF_SCOPE)
    assert escalate != out_of_scope


def test_escalate_reuses_the_existing_user_requested_wording():
    # The generator stays the SINGLE wording owner: the projection invents
    # no prose of its own.
    expected = render_response(
        RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.USER_REQUESTED_HUMAN,
        )
    ).text
    assert tp.project_global_terminal(GlobalAction.ESCALATE) == expected


def test_out_of_scope_reuses_the_existing_wording():
    expected = render_response(
        RoutingDecision(action=RouteAction.OUT_OF_SCOPE)
    ).text
    assert tp.project_global_terminal(GlobalAction.OUT_OF_SCOPE) == expected


def test_continue_and_non_actions_are_refused():
    with pytest.raises(ValueError):
        tp.project_global_terminal(GlobalAction.CONTINUE)
    with pytest.raises(ValueError):
        tp.project_global_terminal("escalate")
    with pytest.raises(ValueError):
        tp.project_global_terminal(None)


# ============================================================
# 2. NO ROUTER DECISION COUPLING IN THE CALLER / NOT WIRED ANYWHERE
# ============================================================

def test_public_api_surface():
    assert set(tp.__all__) == {"project_global_terminal"}


def test_module_is_wired_only_into_the_dispatch_boundary():
    root = Path(tp.__file__).resolve().parents[1]

    assert _imports_module(
        root / "support/turn_dispatch.py", "support.turn_projection"
    )

    for rel in (
        "support/support_core.py",
        "support/router.py",
        "support/global_policy.py",
        "yandex/chat/server.py",
        "yandex/chat/alice_support_agent.py",
        "yandex/realtime/web/server.py",
    ):
        assert not _imports_module(
            root / rel, "support.turn_projection"
        ), rel
