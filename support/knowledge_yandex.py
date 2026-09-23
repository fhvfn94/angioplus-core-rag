"""
Yandex knowledge backend -- Slice 5C adapter over the EXISTING Yandex
retrieval stack.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search). This
module is the provider-specific implementation of the frozen 5A
:class:`~support.knowledge.KnowledgeBackend` port.

It REUSES the existing retrieval stack -- ``RetrievalController`` ->
``YandexVectorStoreClient`` -- and introduces NO second Yandex pipeline:
no direct HTTP, no auth/env handling, no ranking options, no score parsing.
The controller is built LAZILY inside :meth:`YandexKnowledgeBackend.search`
so that a missing environment / transport failure is raised *inside* the
call and is mapped by the 5A core to ``ERROR`` (never a construction-time
crash), and so the module imports without any Yandex/HTTP dependency.

Evidence policy (deterministic, applied inside this adapter because the 5A
core is deliberately provider-independent):

  * preserve Yandex relevance order -- NO sorting by score, authority or
    source_type;
  * authority / IFU status is METADATA ONLY: it never filters, reorders or
    resolves a conflict here;
  * deterministic retain-first deduplication;
  * cap evidence at the requested max (default 5).

There is NO numeric relevance threshold: no repository evidence establishes
the Yandex score distribution, so low score is never treated as absence.
"""

from __future__ import annotations

from typing import Optional, Sequence

from support.knowledge import DEFAULT_MAX_RESULTS, KnowledgeItem


class YandexKnowledgeBackend:
    """KnowledgeBackend implementation backed by ``RetrievalController``.

    The controller is injectable for pure (network-free) tests. When it is
    not injected, a ``RetrievalController`` is constructed lazily on the
    first ``search`` call.
    """

    def __init__(
        self,
        *,
        controller: Optional[object] = None,
        max_num_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        if max_num_results < 1:
            raise ValueError(
                "max_num_results must be greater than 0"
            )

        self._controller = controller
        self._max_num_results = max_num_results

    def _resolve_controller(self) -> object:
        """Return the retrieval controller, building it lazily if needed.

        The import is local so the module can be imported (and unit-tested)
        without the Yandex HTTP stack or credentials.
        """
        if self._controller is None:
            from yandex.retrieval.controller import (  # local: keep imports pure
                RetrievalController,
            )

            self._controller = RetrievalController(
                max_num_results=self._max_num_results,
            )

        return self._controller

    def search(
        self,
        query: str,
        *,
        max_num_results: int,
    ) -> Sequence[KnowledgeItem]:
        """Run EXACTLY ONE controller search and apply the evidence policy.

        Any provider/transport failure raised by the controller propagates
        out of this method and is translated by the 5A core into
        ``KnowledgeResult(status=ERROR)``; nothing is swallowed here.
        """
        controller = self._resolve_controller()

        response = controller.search(query)

        items = [_to_item(ev) for ev in response.evidence]

        limit = min(max_num_results, self._max_num_results)

        return _apply_evidence_policy(items, max_evidence=limit)


def _to_item(ev: object) -> KnowledgeItem:
    """Map one ``EvidenceChunk`` to a :class:`KnowledgeItem` verbatim.

    Every field representable on the frozen 5A ``KnowledgeItem`` contract is
    carried over unchanged; nothing is reinterpreted, promoted or dropped.
    """
    return KnowledgeItem(
        text=ev.text,
        source=ev.source,
        section=ev.section,
        pages=ev.pages,
        filename=ev.vector_store_filename,
        file_id=ev.file_id,
        source_type=ev.source_type,
        authority=ev.authority,
        score=ev.score,
        rank=ev.rank,
    )


def _apply_evidence_policy(
    items: Sequence[KnowledgeItem],
    *,
    max_evidence: int = DEFAULT_MAX_RESULTS,
) -> list[KnowledgeItem]:
    """Deterministic evidence policy. Pure; never reorders.

    Steps, in order:

      1. drop items with blank text (unusable evidence);
      2. KEEP Yandex order -- no sorting by score / authority / source_type;
      3. deduplicate RETAIN-FIRST:
         * structural identity ``(document_identity, section, pages)`` is
           applied ONLY when ``section`` or ``pages`` is present;
         * when both are absent, only EXACT normalized-text duplicates are
           collapsed (different chunks of the same document are preserved);
         * exact normalized-text duplicates are always collapsed;
      4. cap the result at ``max_evidence``.
    """
    if max_evidence < 1:
        return []

    result: list[KnowledgeItem] = []
    seen_structural: set[tuple] = set()
    seen_text: set[str] = set()

    for item in items:
        text = item.text

        if not text or not text.strip():
            continue

        normalized_text = " ".join(text.split())

        if normalized_text in seen_text:
            continue

        has_span = bool(item.section or item.pages)

        if has_span:
            document_identity = item.file_id or item.filename
            key = (document_identity, item.section, item.pages)
            if key in seen_structural:
                continue
            seen_structural.add(key)

        seen_text.add(normalized_text)
        result.append(item)

        if len(result) >= max_evidence:
            break

    return result


__all__ = ["YandexKnowledgeBackend"]
