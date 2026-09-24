"""
Import-isolation guard for the Slice 9 application boundary.

The new modules are INFRASTRUCTURE ONLY: they must NOT pull any runtime
transport (``yandex.chat.server``), conversation FSM, Tracker or idempotency
code into the process. This is asserted in a *fresh* interpreter per module
so that any transitive import introduced later fails the guard.

Deliberately NOT a ``@pytest.mark.db`` test: it touches no database and must
stay fast. ``support.domain`` is excluded from the forbidden set because
``support/__init__.py`` re-exports it for every ``support.*`` import.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


MODULES_UNDER_TEST = (
    "support.turn_executor",
    "support.procedure_state_store",
)

# Runtime transport / FSM / Tracker / idempotency modules. Importing any of
# these from the boundary would couple it to a transport or lifecycle it must
# not own.
FORBIDDEN_MODULES = (
    "yandex.chat.server",
    "support.tracker_service",
    "support.idempotency",
    "support.session_service",
    "support.message_service",
    "support.state_service",
    "support.procedure_state_service",
    "support.conversation_context",
    "support.schemas",
)


@pytest.mark.parametrize("module", MODULES_UNDER_TEST)
def test_import_pulls_no_runtime_or_tracker_modules(module: str) -> None:
    code = (
        "import sys\n"
        f"import {module}\n"
        f"forbidden = {list(FORBIDDEN_MODULES)!r}\n"
        "loaded = [m for m in forbidden if m in sys.modules]\n"
        "print('FORBIDDEN:' + ','.join(loaded) if loaded else 'OK')\n"
        "raise SystemExit(1 if loaded else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
