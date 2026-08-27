# -*- coding: utf-8 -*-
"""Real-mic -> LiveKit -> VAD -> faster-whisper -> transcript (Phase D1.4).

Phase D2.1 added the session-level `on_final` callback on AudioTrackListener.
It lets the LiveKit voice agent subscribe to every transcribed utterance and
send it onward (e.g. to the RAG response path), while keeping this module free
of RAG / TTS / SIP / aiogram code. The callback receives the transcript text
and the participant that produced it, so the agent can attribute and publish
the answer back to the correct caller.

Pipeline proved in D1.4 / D2.1:

    Windows microphone
      -> LiveKit Room (a remote participant's audio track is subscribed)
      -> Silero VAD            (voice.vad.load_vad)
      -> StreamAdapter          (livekit.agents.stt.StreamAdapter)
      -> FasterWhisperSTT       (voice.stt)
      -> FINAL_TRANSCRIPT       (D2.1: forwarded to `on_final`, sent to RAG)

Design constraints honored here:
    - LiveKit EventEmitter callbacks must stay synchronous. Async work is
      scheduled with asyncio.create_task and task failures are logged.
    - A single STT + VAD pair is reused for the whole job/session, so the
      Whisper model is lazily constructed once, not rebuilt per utterance.
    - No RAG / TTS / SIP / aiogram code in this module.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional, Tuple

from livekit import rtc
from livekit.agents.stt import StreamAdapter
from livekit.agents.stt.stt import SpeechEventType

from voice.stt import FasterWhisperSTT
from voice.vad import load_vad

logger = logging.getLogger(__name__)


def _log_task_failure(task: asyncio.Task, label: str) -> None:
    """Log any exception raised inside a background task so it is never lost."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("unhandled error in %s: %r", label, exc, exc_info=exc)


class AudioTranscriber:
    """Compose StreamAdapter(FasterWhisperSTT, VAD) for a single audio track.

    The STT and VAD instances are shared (created by AudioTrackListener) so the
    Whisper model is only lazily built once per session. Incoming AudioFrames
    are pushed into the StreamAdapter's stream; each FINAL_TRANSCRIPT is logged
    and optionally forwarded to `on_final`.
    """

    def __init__(
        self,
        *,
        track: rtc.Track,
        stt: FasterWhisperSTT,
        vad: object,
        on_final: Optional[Callable[[str], None]] = None,
        audio_stream_factory: Callable = rtc.AudioStream.from_track,
    ) -> None:
        self._track = track
        self._stt = stt
        self._vad = vad
        self._on_final = on_final
        self._audio_stream_factory = audio_stream_factory
        self._adapter = StreamAdapter(stt=stt, vad=vad)

    @property
    def stt(self) -> FasterWhisperSTT:
        return self._stt

    @property
    def vad(self) -> object:
        return self._vad

    async def run(self) -> None:
        """Push AudioFrames into the adapter and consume its SpeechEvents.

        The producer pushes every inbound frame; the consumer emits
        START_OF_SPEECH / END_OF_SPEECH and logs each FINAL_TRANSCRIPT. Both
        run as concurrent tasks in the same event loop so the adapter can take
        audio in one coroutine and emit transcription events out another.
        """
        stt_stream = self._adapter.stream()

        # The audio_server_factory returns an async-iterable of AudioFrame
        # (e.g. rtc.AudioStream.from_track(track=...)).
        audio_stream = self._audio_stream_factory(track=self._track)

        async def _producer() -> None:
            try:
                # The current livekit.rtc AudioStream async-iterates
                # AudioFrameEvent, NOT raw AudioFrame. Unwrap the real audio
                # frame (event.frame) before pushing it into the adapter.
                async for item in audio_stream:
                    # `frame = audio_frame_event.frame`; the getattr fallback
                    # keeps raw-AudioFrame async sources working too.
                    frame = getattr(item, "frame", item)
                    stt_stream.push_frame(frame)
            finally:
                # End the segment so any trailing speech is flushed to the VAD
                # and produces a final END_OF_SPEECH / FINAL_TRANSCRIPT.
                stt_stream.end_input()

        async def _consumer() -> None:
            async for event in stt_stream:
                if event.type is not SpeechEventType.FINAL_TRANSCRIPT:
                    continue
                text = (
                    event.alternatives[0].text if event.alternatives else ""
                )
                logger.info("FINAL_TRANSCRIPT: %s", text)
                if self._on_final is not None:
                    self._on_final(text)

        try:
            await asyncio.gather(_producer(), _consumer())
        finally:
            await audio_stream.aclose()


class AudioTrackListener:
    """Manages AudioTranscriber tasks for subscribed remote audio tracks.

    A single STT + VAD is created lazily (on the first audio track) and reused
    for the whole session. The synchronous Room.on() handlers schedule async
    processing and clean up tasks when tracks are unsubscribed.

    Phase D2.1: a session-level `on_final` callback
    ``Callable[[str, object], None]`` receives every FINAL_TRANSCRIPT with its
    producing participant. It is optional (default None); when set, each
    per-track AudioTranscriber forwards transcripts through it with the
    participant bound in, so the caller can attribute/publish the answer back.
    """

    # Session-level transcript handler: called once per FINAL_TRANSCRIPT with
    # the transcript text and the participant whose track produced it.
    FinalHandler = Callable[[str, object], None]

    def __init__(
        self,
        *,
        stt_factory: Optional[Callable[[], FasterWhisperSTT]] = None,
        vad_factory: Optional[Callable[[], object]] = None,
        audio_stream_factory: Optional[Callable] = None,
        on_final: Optional["AudioTrackListener.FinalHandler"] = None,
    ) -> None:
        self._stt_factory = stt_factory or (lambda: FasterWhisperSTT())
        self._vad_factory = vad_factory or load_vad
        self._audio_stream_factory = (
            audio_stream_factory or rtc.AudioStream.from_track
        )
        self._on_final = on_final
        self._stt: Optional[FasterWhisperSTT] = None
        self._vad: Optional[object] = None
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def stt(self) -> Optional[FasterWhisperSTT]:
        """The session-wide FasterWhisperSTT (None until first audio track)."""
        return self._stt

    @property
    def vad(self) -> Optional[object]:
        """The session-wide VAD (None until first audio track)."""
        return self._vad

    def _ensure_models(self) -> Tuple[FasterWhisperSTT, object]:
        if self._stt is None:
            self._stt = self._stt_factory()
        if self._vad is None:
            self._vad = self._vad_factory()
        return self._stt, self._vad

    def on_track_subscribed(
        self, track, publication=None, participant=None
    ) -> None:
        """Synchronous Room.on("track_subscribed"): handle audio tracks only."""
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        if track.sid in self._tasks:
            return
        # Bind the producing participant so each transcript can be attributed
        # to the caller who spoke it (Phase D2.1).
        task = asyncio.create_task(self._process_track(track, participant))
        task.add_done_callback(
            lambda t: _log_task_failure(t, f"audio track {track.sid}")
        )
        self._tasks[track.sid] = task

    def on_track_unsubscribed(
        self, track, publication=None, participant=None
    ) -> None:
        """Synchronous Room.on("track_unsubscribed"): cancel processing."""
        task = self._tasks.pop(track.sid, None)
        if task is not None:
            task.cancel()

    async def _process_track(self, track, participant=None) -> None:
        def _route_final(text: str) -> None:
            # Keep the AudioTranscriber.on_final contract as a plain
            # (str) -> None callback while attributing the participant here.
            if self._on_final is not None:
                self._on_final(text, participant)

        try:
            stt, vad = self._ensure_models()
            transcriber = AudioTranscriber(
                track=track,
                stt=stt,
                vad=vad,
                on_final=_route_final,
                audio_stream_factory=self._audio_stream_factory,
            )
            await transcriber.run()
        finally:
            self._tasks.pop(track.sid, None)

    async def cancel_all(self) -> None:
        """Cancel all active audio-processing tasks (e.g. on disconnect)."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
