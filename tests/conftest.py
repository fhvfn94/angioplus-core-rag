# -*- coding: utf-8 -*-
"""Shared test configuration.

Adds the repository root and the ``bot`` package directory to ``sys.path`` so
that both ``import app.main`` / ``import scripts.ask`` and ``bot/main.py``
(which imports ``services.stt``) can be imported from tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT, REPO_ROOT / "bot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
