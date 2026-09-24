"""
Cold-start Procedure selection -- MVP lexical bootstrap (Slice 10).

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

PURPOSE
-------
Bootstrap the FIRST production Procedure: given the raw CURRENT user
message (a cold-start signal), deterministically map it to ONE declarative,
already-validated :class:`~support.procedure.Procedure` identity
(``procedure_id`` / ``procedure_version``). This module loads and validates
the procedure files under ``support/procedures/`` and is the ONLY component
in the repository allowed to perform that mapping.

COLD-START SELECTION MODEL = "mvp_lexical_bootstrap"      # TEMPORARY.

  * `catalog.issue_match` is NOT the long-term semantic routing architecture.
  * It exists ONLY to bootstrap the first production Procedure without changing
    the FROZEN SemanticObservation contract (Slice 4).
  * ProcedureCatalog remains the ONLY component allowed to map a cold-start
    signal to procedure_id / procedure_version.
  * The LLM must NEVER output or select procedure_id.
  * Ambiguity is never guessed: if MORE THAN ONE procedure qualifies, the
    result is no selection (``None``) -- there is deliberately no
    best-match ranking and no ``procedure_id`` ordering tie-break.
  * A FUTURE slice may introduce a closed semantic issue taxonomy produced by the
    analyzer and deterministically mapped by ProcedureCatalog -- NOT implemented here.
  * Do NOT treat substring matching as the final selection mechanism.

DESIGN BOUNDARY
---------------
  * ``select(text)`` is the FROZEN public entry point. Its signature takes
    ONLY ``text`` -- deliberately NOT a ``SemanticObservation`` and NOT an
    analyzer/provider object -- so procedure identity can never be supplied
    by a model. Nothing here consults an analyzer.
  * ``issue_match(text)`` is the TEMPORARY lexical mechanism behind
    ``select``. A future slice may replace ``issue_match`` (e.g. with a
    closed semantic issue taxonomy produced by the analyzer) WITHOUT
    changing the frozen ``select`` boundary.
  * Deterministic, total and pure: no LLM, no network, no DB, no framework,
    no conversation history.

NON-GOALS (deliberately absent)
-------------------------------
No wiring into any runtime path, no routing, no rendering, no observation
parsing, no LLM/embedding similarity, no score/threshold tuning beyond the
explicit lexical bootstrap, and no persistence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from support.procedure import Procedure, load_procedure


# ============================================================
# COLD-START MODEL (TEMPORARY BOOTSTRAP)
# ============================================================
# Fixed verbatim so the bootstrap cannot quietly become "the architecture".
COLD_START_SELECTION_MODEL = "mvp_lexical_bootstrap"

# Default location of the declarative production procedures.
_DEFAULT_PROCEDURES_DIR = Path(__file__).parent / "procedures"


# ============================================================
# LEXICAL TOKENS (bootstrap only)
# ============================================================

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")
_MIN_TOKEN_LEN = 3

# Structural words carry no discriminative power for an issue match. They
# only ever filter the PROCEDURE issue label (the user text is matched as a
# plain lowercased string).
_LEXICAL_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are",
        "для", "это", "как", "что", "или", "при", "над", "под", "без",
        "про", "чем", "все", "всё",
    }
)

# An issue match requires at least TWO distinct issue tokens to be present,
# so a single generic token can never select a procedure.
_MIN_ISSUE_HITS = 2


def _issue_tokens(issue: str) -> frozenset[str]:
    """Meaningful lexical tokens of a procedure's issue label.

    Lowercased, length >= ``_MIN_TOKEN_LEN``, minus structural stopwords.
    Purely lexical -- no stemming, no semantics.
    """
    low = (issue or "").lower()
    return frozenset(
        tok
        for tok in _TOKEN_RE.findall(low)
        if len(tok) >= _MIN_TOKEN_LEN and tok not in _LEXICAL_STOPWORDS
    )


# ============================================================
# SELECTION VALUE OBJECT
# ============================================================

@dataclass(frozen=True)
class ProcedureSelection:
    """The cold-start selection result: a procedure IDENTITY + the graph.

    This is the ONLY shape in which a cold-start signal becomes a
    ``procedure_id`` / ``procedure_version``. ``matched_tokens`` is
    INTERNAL diagnostic metadata (why the match happened) and is never
    rendered to a user.
    """

    procedure_id: str
    version: int
    procedure: Procedure
    matched_tokens: frozenset[str]

    def identity(self) -> tuple[str, int]:
        return (self.procedure_id, self.version)

    def __post_init__(self) -> None:
        if not isinstance(self.procedure_id, str) or not self.procedure_id.strip():
            raise ValueError("procedure_id must be a non-blank string.")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise ValueError("version must be an integer.")
        if not isinstance(self.procedure, Procedure):
            raise ValueError("procedure must be a Procedure.")


# ============================================================
# CATALOG
# ============================================================

class ProcedureCatalog:
    """The ONLY cold-start -> procedure-identity mapper (TEMPORARY)."""

    def __init__(self, procedures: Sequence[Procedure]) -> None:
        if procedures is None:
            raise ValueError("procedures must be a sequence of Procedure.")
        indexed: list[tuple[Procedure, frozenset[str]]] = []
        # Identity is the PAIR (procedure_id, version): one procedure_id may
        # therefore carry several versions. Only an EXACT duplicate identity
        # is rejected -- there is no per-id collapse and no implicit
        # "keep the latest version" behaviour.
        seen: set[tuple[str, int]] = set()
        for procedure in procedures:
            if not isinstance(procedure, Procedure):
                raise TypeError(
                    "catalog entries must be Procedure instances, got "
                    f"{type(procedure).__name__}."
                )
            identity = (procedure.procedure_id, procedure.version)
            if identity in seen:
                raise ValueError(
                    f"duplicate procedure identity {identity!r}."
                )
            seen.add(identity)
            indexed.append((procedure, _issue_tokens(procedure.issue)))
        # No ordering is imposed: the selection decision never ranks or
        # tie-breaks on procedure_id (or on anything else).
        self._entries: tuple[tuple[Procedure, frozenset[str]], ...] = tuple(
            indexed
        )

    # ---------------------------------------------------------
    # LOADING
    # ---------------------------------------------------------

    @classmethod
    def from_directory(
        cls, directory: Optional[str | Path] = None
    ) -> "ProcedureCatalog":
        """Load + validate every ``*.yaml`` procedure in ``directory``.

        Each file is parsed and validated by the frozen
        :func:`support.procedure.load_procedure`; a malformed procedure
        raises :class:`~support.procedure.ProcedureError`.
        """
        root = (
            Path(directory)
            if directory is not None
            else _DEFAULT_PROCEDURES_DIR
        )
        if not root.is_dir():
            raise FileNotFoundError(f"procedure directory not found: {root}")
        procedures = [
            load_procedure(path) for path in sorted(root.glob("*.yaml"))
        ]
        return cls(procedures)

    # ---------------------------------------------------------
    # READ-ONLY VIEWS
    # ---------------------------------------------------------

    @property
    def procedure_ids(self) -> frozenset[str]:
        return frozenset(proc.procedure_id for proc, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ---------------------------------------------------------
    # COLD-START SELECTION (TEMPORARY BOOTSTRAP)
    # ---------------------------------------------------------

    def _qualifying(self, text: str) -> list[ProcedureSelection]:
        """Every procedure that QUALIFIES for ``text`` -- unranked.

        A procedure qualifies, independently of every other procedure,
        when at least ``_MIN_ISSUE_HITS`` distinct issue tokens occur (as
        substrings) in the lowercased ``text``. The hit count only decides
        whether *that* procedure qualifies; qualifying procedures are never
        compared, scored, or ordered against each other.
        """
        low = text.lower()
        qualifying: list[ProcedureSelection] = []
        for procedure, tokens in self._entries:
            hits = frozenset(tok for tok in tokens if tok in low)
            if len(hits) < _MIN_ISSUE_HITS:
                continue
            qualifying.append(
                ProcedureSelection(
                    procedure_id=procedure.procedure_id,
                    version=procedure.version,
                    procedure=procedure,
                    matched_tokens=hits,
                )
            )
        return qualifying

    def issue_match(self, text: str) -> Optional[ProcedureSelection]:
        """TEMPORARY lexical bootstrap: map ``text`` to AT MOST one procedure.

        Deterministic, total, pure. Each procedure is tested INDEPENDENTLY
        for qualification; the number of lexical hits is NEVER used to rank
        procedures against one another. The outcome is:

          * zero qualifying procedures -> ``None``
          * exactly one qualifying procedure -> that procedure
          * more than one qualifying procedure -> ``None`` (ambiguous:
            deliberately no best-match and no ``procedure_id`` tie-break)

        This is NOT the long-term selection mechanism: it exists only to
        bootstrap the first production Procedure without changing the
        frozen ``SemanticObservation`` contract.
        """
        if not isinstance(text, str) or not text.strip():
            return None
        qualifying = self._qualifying(text)
        if len(qualifying) != 1:
            return None
        return qualifying[0]

    def select(self, text: str) -> Optional[ProcedureSelection]:
        """FROZEN cold-start entry point: signal ``text`` -> identity or None.

        Takes ONLY the raw user ``text`` (no analyzer / observation
        argument) on purpose, so a model can never choose the procedure.
        Delegates to the TEMPORARY :meth:`issue_match`; a future slice may
        replace that mechanism without changing this boundary.
        """
        return self.issue_match(text)

    # ---------------------------------------------------------
    # IDENTITY-BASED REHYDRATION (Slice 11A.1)
    # ---------------------------------------------------------

    def get(self, procedure_id: str, version: int) -> Optional[Procedure]:
        """EXACT identity -> the already-validated Procedure, or ``None``.

        Identity-based rehydration ONLY, deliberately NOT a second entry
        point for cold-start selection:

          * BOTH ``procedure_id`` and ``version`` are REQUIRED;
          * the match is EXACT on the pair ``(procedure_id, version)`` --
            no case-folding, no whitespace trimming, no lexical matching;
          * an unknown ``procedure_id`` yields ``None``;
          * a wrong / unknown version yields ``None`` with NO fallback:
            never the "latest" version, never an implicit upgrade, never
            another version of the same procedure;
          * the returned object is the very same validated
            :class:`~support.procedure.Procedure` instance the catalog
            holds -- no re-parse, no LLM / analyzer involvement.

        Cold-start selection remains exclusively :meth:`select`; this
        method never maps text to an identity.
        """
        for procedure, _ in self._entries:
            if (
                procedure.procedure_id == procedure_id
                and procedure.version == version
            ):
                return procedure
        return None


__all__ = [
    "COLD_START_SELECTION_MODEL",
    "ProcedureSelection",
    "ProcedureCatalog",
]
