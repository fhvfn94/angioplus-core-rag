# -*- coding: utf-8 -*-
"""Minimal LiveKit Agents 1.x entrypoint for the voice component (Phase B).

Phase B intent: prove that
  - livekit.agents imports correctly in this environment,
  - the CURRENT 1.x API (AgentServer / @rtc_session / WorkerOptions /
    JobContext) is used correctly,
  - the agent runtime can start in local development,
  - this `voice` package stays isolated from the RAG pipeline (app.*) and
    from Qdrant.

Phase D2.1 connects the transcribed STT utterance to the existing RAG response
path: each FINAL_TRANSCRIPT from the real-mic pipeline is sent to the RAG API
(RagClient -> POST /ask) and the answer is published back to the room as a data
message (destination = the speaking participant).

Phase D2.3 adds the FIRST voice-to-voice link: after the RAG answer is sent as
text/data, the same answer is synthesized by a REMOTE Qwen3-TTS HTTP service
(voice.tts_client -> scripts/tts_server.py on a GPU) and the resulting audio is
published into the room as a spoken track (voice.tts_publisher). TTS is
OPTIONAL and OFF by default: it is enabled only when TTS_BASE_URL is set, so the
existing text/data-only path keeps working when no GPU TTS is reachable. SIP /
telephony / streaming / barge-in are intentionally NOT implemented.

This agent is a *programmatic participant* (see
https://docs.livekit.io/agents/server/job/#entrypoint) — it does NOT create an
AgentSession, so no speech models are required (STT/VAD are composed manually
via voice/audio_transcriber.py).

The ONLY knowledge/RAG boundary is voice.rag_client.RagClient -> POST /ask.
No Qdrant access, no second RAG pipeline, no conversation-memory duplication:
conversation memory stays in the existing RAG pipeline (app/).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from livekit import agents, rtc
from livekit.agents.worker import ServerType
from livekit.rtc import DataPacket

from voice.audio_transcriber import AudioTrackListener
from voice.config import load_rag_config, load_tts_config
from voice.rag_client import (
    RagClient,
    RagClientError,
    RagInvalidJsonError,
    RagMalformedResponseError,
    RagTimeoutError,
)
from voice.tts_client import TTSClient, TTSClientError
from voice.tts_publisher import publish_tts_audio

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


def _make_tts_client() -> TTSClient | None:
    """Build the remote Qwen3-TTS HTTP client, or None when TTS is disabled.

    TTS is OFF unless TTS_BASE_URL is set in the environment (the Vast IP is
    intentionally never hardcoded). When disabled, the voice worker falls back
    to the existing text/data-only answer path (Phase D2.1).
    """
    cfg = load_tts_config()
    if not cfg.enabled:
        logger.info(
            "TTS disabled (TTS_BASE_URL not set); replies delivered as "
            "text/data only"
        )
        return None
    logger.info(
        "TTS boundary configured at %s (speaker=%s language=%s timeout=%ss "
        "out_rate=%d)", cfg.base_url, cfg.speaker, cfg.language,
        cfg.timeout_seconds, cfg.sample_rate,
    )
    return TTSClient(
        base_url=cfg.base_url,
        speaker=cfg.speaker,
        language=cfg.language,
        timeout_seconds=cfg.timeout_seconds,
        sample_rate=cfg.sample_rate,
    )


async def entrypoint(ctx: agents.JobContext) -> None:
    """Run once per job: connect to the room and act as a programmatic
    participant that answers textual messages via the existing RAG API."""
    logger.info("entrypoint starting for room=%s job=%s",
                ctx.room.name, ctx.job.id)

    # Connect to the room (AgentSession would auto-connect; as a programmatic
    # participant we control it explicitly).
    await ctx.connect()

    rag = _make_rag_client()
    # Remote GPU TTS client (None when TTS_BASE_URL is unset -> text-only).
    tts = _make_tts_client()

    def _conversation_id(participant: Optional[rtc.RemoteParticipant]) -> str:
        """Scope RAG conversation memory per caller (participant identity)."""
        if participant is not None:
            return f"{ctx.room.name}:{participant.identity}"
        return _DEFAULT_CONVERSATION_ID

    async def _answer_and_publish(
        question: str,
        *,
        participant: Optional[rtc.RemoteParticipant],
    ) -> None:
        """Ask RAG and publish the answer to the room (destination = caller).

        Shared by the text data-message path and the D2.1 voice path so both
        treat the question/transcript identically through the RAG boundary.
        """
        sender = (
            participant.identity
            if participant is not None
            else _DEFAULT_USER_ID
        )
        logger.info("RAG request started for %s", sender)
        try:
            answer = await _answer(
                rag,
                question,
                conversation_id=_conversation_id(participant),
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
                [participant.identity]
                if participant is not None
                else []
            ),
            reliable=True,
        )
        logger.info("response published to %s", sender)

        # D2.3: if a remote GPU TTS service is configured, speak the answer
        # into the room so the caller hears it. Text stays published above for
        # debugging. A TTS failure must NOT break the text answer already sent.
        if tts is not None:
            await _speak_answer(tts, ctx.room.local_participant, answer)

    # ------------------------------------------------------------------
    # (A) Text data-message path (existing text E2E, unchanged behaviour)
    # ------------------------------------------------------------------
    async def _on_data_received_task(packet: DataPacket) -> None:
        """Handle an inbound text data message via the RAG API.

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
        await _answer_and_publish(
            user_question,
            participant=packet.participant,
        )

    def _on_data_received(packet: DataPacket) -> None:
        """Synchronous RTC callback for Room.on("data_received").

        livekit.rtc Room.on() rejects async callbacks, so we register a
        synchronous callback and schedule the async handler as a background
        task, logging any internal failure so it is never silently lost.
        """
        _spawn(_on_data_received_task(packet))

    # Data messages arrive on the Room, not on individual participants.
    ctx.room.on("data_received", _on_data_received)

    # ------------------------------------------------------------------
    # (B) Real-mic voice path (Phase D1.4 STT/VAD + D2.1 RAG bridge)
    # ------------------------------------------------------------------
    def _on_voice_final(transcript: str, participant) -> None:
        """D2.1 bridge: answer a spoken utterance via the existing RAG path.

        The STT FINAL_TRANSCRIPT is attributed to the participant who spoke,
        sent to the RAG API, and the answer is published back to that caller.
        """
        if not transcript or not transcript.strip():
            return
        # _spawn schedules the async "answer + publish" task and returns
        # immediately, keeping this EventEmitter callback synchronous.
        _spawn(_answer_and_publish(transcript, participant=participant))

    def _spawn(coro) -> None:
        """Run a coroutine as a background task, logging any failure."""
        task = asyncio.create_task(coro)
        task.add_done_callback(_log_task_failure)

    def _log_task_failure(task: asyncio.Task) -> None:
        """Log any exception raised inside a background handling task."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "unhandled error in agent handler: %r", exc,
                exc_info=exc,
            )

    def _on_room_disconnected(_reason) -> None:
        """Cancel audio tasks on disconnect (synchronous callback)."""
        _spawn(audio.cancel_all())

    # AudioTrackListener wires each FINAL_TRANSCRIPT to on_final, which now
    # (D2.1) answers the transcript via RAG and publishes the answer back to
    # the speaking participant. The track callbacks are synchronous
    # (EventEmitter rule) and schedule async work internally.
    audio = AudioTrackListener(on_final=_on_voice_final)
    ctx.room.on("track_subscribed", audio.on_track_subscribed)
    ctx.room.on("track_unsubscribed", audio.on_track_unsubscribed)
    ctx.room.on("disconnected", _on_room_disconnected)

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


async def _speak_answer(tts: TTSClient, participant: rtc.LocalParticipant,
                        text: str) -> None:
    """Synthesize ``text`` via the remote TTS service and publish it to the room.

    A remote-TTS failure is caught and logged so it never breaks the LiveKit
    worker or the already-sent text/data reply (Phase D2.3 semantics).
    """
    try:
        audio = await tts.synthesize(text)
    except TTSClientError as exc:
        logger.warning("TTS request failed for answer: %s", exc)
        return
    except ValueError as exc:  # e.g. empty text
        logger.warning("TTS skipped for empty text: %s", exc)
        return

    logger.info(
        "TTS synthesized answer: %.2fs at %d Hz",
        audio.duration_seconds, audio.sample_rate,
    )
    await publish_tts_audio(participant, audio)


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
