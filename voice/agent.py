# -*- coding: utf-8 -*-
"""Minimal LiveKit Agents 1.x entrypoint for the voice component (Phase B).

Phase B intent: prove that
  - livekit.agents imports correctly in this environment,
  - the CURRENT 1.x API (AgentServer / @rtc_session / WorkerOptions /
    JobContext) is used correctly,
  - the agent runtime can start in local development,
  - this `voice` package stays isolated from the RAG pipeline (app.*) and
    from Qdrant.

STT / TTS / SIP / telephony are intentionally NOT implemented yet. This agent
is a *programmatic participant* (see
https://docs.livekit.io/agents/server/job/#entrypoint) — it does NOT create an
AgentSession, so no speech models are required.

The ONLY knowledge/RAG boundary is voice.rag_client.RagClient -> POST /ask.
No Qdrant access, no second RAG pipeline, no conversation-memory duplication:
conversation memory stays in the existing RAG pipeline (app/).
"""
from __future__ import annotations

import asyncio
import logging

from livekit import agents
from livekit.agents.worker import ServerType
from livekit.rtc import DataPacket

from voice.config import load_rag_config
from voice.rag_client import (
    RagClient,
    RagClientError,
    RagInvalidJsonError,
    RagMalformedResponseError,
    RagTimeoutError,
)

logger = logging.getLogger(__name__)

# Default conversation identity used when a minimal client message doesn't
# carry one. The RAG pipeline owns conversation memory; we only forward ids.
_DEFAULT_CONVERSATION_ID = "livekit-voice"
_DEFAULT_USER_ID = "livekit-voice-user"


def _make_rag_client() -> RagClient:
    """Build the single RAG boundary client from the environment."""
    cfg = load_rag_config()
    logger.info("RAG boundary configured at %s (timeout %ss)", cfg.url,
                cfg.timeout_seconds)
    # The client lazily creates its own httpx.AsyncClient per call.
    return RagClient(url=cfg.url, timeout_seconds=cfg.timeout_seconds)


async def entrypoint(ctx: agents.JobContext) -> None:
    """Run once per job: connect to the room and act as a programmatic
    participant that answers textual messages via the existing RAG API."""
    logger.info("entrypoint starting for room=%s job=%s",
                ctx.room.name, ctx.job.id)

    # Connect to the room (AgentSession would auto-connect; as a programmatic
    # participant we control it explicitly).
    await ctx.connect()

    rag = _make_rag_client()

    async def on_data_received(packet: DataPacket) -> None:
        """Handle an inbound text data message and answer via the RAG API.

        Protocol: a remote participant publishes a data message whose payload
        is the user's question text. This proves the RAG boundary works
        end-to-end without any speech models.
        """
        payload = packet.data
        if not payload:
            return
        if isinstance(payload, (bytes, bytearray)):
            user_question = bytes(payload).decode("utf-8", errors="replace")
        else:
            user_question = payload

        if not user_question.strip():
            return

        # Identities only; we avoid logging full question text needlessly.
        sender = (
            packet.participant.identity
            if packet.participant is not None
            else _DEFAULT_USER_ID
        )
        logger.info(
            "data_received: question from %s (%d bytes)",
            sender,
            len(payload),
        )

        # Conversation memory stays in the RAG pipeline; we only forward the
        # caller (participant) identity so the RAG can scope per-conversation
        # memory. We prefix the per-room conversation id with the sender.
        conversation_id = (
            f"{ctx.room.name}:{packet.participant.identity}"
            if packet.participant is not None
            else _DEFAULT_CONVERSATION_ID
        )

        logger.info("RAG request started for %s", sender)
        try:
            answer = await _answer(
                rag,
                user_question,
                conversation_id=conversation_id,
                user_id=sender,
            )
        except RagClientError as exc:
            logger.warning("RAG error for %s: %s", sender, exc)
            answer = handle_rag_error(exc)
        logger.info(
            "RAG request completed for %s: answer %d bytes",
            sender,
            len(answer.encode("utf-8")),
        )

        # Publish the answer back to the room as a data message.
        # NOTE: publish_data is a coroutine in the current livekit.rtc SDK;
        # it must be awaited or the answer is silently never sent.
        await ctx.room.local_participant.publish_data(
            answer.encode("utf-8"),
            destination_identities=(
                [packet.participant.identity]
                if packet.participant is not None
                else []
            ),
            reliable=True,
        )
        logger.info("response published to %s", sender)

    def _log_task_failure(task: asyncio.Task) -> None:
        """Log any exception raised inside a background data-handling task."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "unhandled error in data_received handler: %r", exc,
                exc_info=exc,
            )

    def _on_data_received(packet: DataPacket) -> None:
        """Synchronous RTC callback for Room.on("data_received").

        livekit.rtc Room.on() rejects async callbacks, so we register a
        synchronous callback and schedule the async handler as a background
        task, logging any internal failure so it is never silently lost.
        """
        task = asyncio.create_task(on_data_received(packet))
        task.add_done_callback(_log_task_failure)

    # Data messages arrive on the Room, not on individual participants.
    ctx.room.on("data_received", _on_data_received)

    logger.info("agent ready: connected to %s, awaiting messages", ctx.room.name)
    # Keep the job alive; the server tears it down when participants leave.


async def _answer(
    rag: RagClient,
    question: str,
    *,
    conversation_id: str = _DEFAULT_CONVERSATION_ID,
    user_id: str = _DEFAULT_USER_ID,
) -> str:
    """Ask the existing RAG pipeline and return only the answer text.

    conversation_id/user_id are forwarded for RAG-side conversation memory;
    we do not implement any memory of our own.
    """
    result = await rag.ask(
        question,
        top_k=10,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    return result.answer


def handle_rag_error(exc: RagClientError) -> str:
    """Map RAG client errors to a friendly spoken/plain answer."""
    if isinstance(exc, RagTimeoutError):
        return "Похоже, поиск занял слишком много времени. Попробуйте ещё раз."
    if isinstance(exc, RagInvalidJsonError):
        return "Сервис вернул некорректный ответ. Попробуйте ещё раз."
    if isinstance(exc, RagMalformedResponseError):
        return "Сервис вернул некорректный ответ. Попробуйте ещё раз."
    return "Не удалось получить ответ. Попробуйте ещё раз."


# ---------------------------------------------------------------------------
# Agent server wiring (current 1.x API)
# ---------------------------------------------------------------------------

def build_server() -> agents.AgentServer:
    """Construct and return an AgentServer, registering the RTC session.

    Uses the CURRENT documented pattern:
      server = agents.AgentServer()
      @server.rtc_session(...)
      async def entrypoint(ctx): ...
    """
    server = agents.AgentServer()
    server.rtc_session(
        entrypoint,
        type=ServerType.ROOM,  # start one agent instance per room
    )
    return server


if __name__ == "__main__":
    # Boot the agent server (production/dev/console handled by 'lk agent').
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
