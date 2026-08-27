# -*- coding: utf-8 -*-
"""Unit tests for Phase D1.4 real-audio integration logic (voice/audio_transcriber.py).

No physical microphone is used. We inject:
  - a fake faster-whisper model (via faster_whisper STT _model_factory),
  - a deterministic fake VAD (no neural net),
  - a fake audio-track factory that yields synthetic AudioFrames.

These validate the integration *logic*: audio track detection, non-audio
ignoring, AudioFrame forwarding into the StreamAdapter, FINAL_TRANSCRIPT
delivery, background-task failure logging, and that the existing RAG text path
in voice/agent.py remains importable/intact.
"""
from __future__ import annotations

import asyncio
import io
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream

from voice.audio_transcriber import (
    AudioTranscriber,
    AudioTrackListener,
    _log_task_failure,
)
from voice.stt import FasterWhisperSTT


# ---------------------------------------------------------------------------
# Fake STT model (records transcribe kwargs, returns configurable segments)
# ---------------------------------------------------------------------------

class FakeWhisperModel:
    segments = [SimpleNamespace(text="Калибровка"), SimpleNamespace(text=" катетера")]

    def __init__(self, model, device, compute_type):
        self.model_arg = model
        self.device_arg = device
        self.compute_type_arg = compute_type
        self.calls = []

    def transcribe(self, audio, language=None, initial_prompt=None,
                   beam_size=None, best_of=None, temperature=None, **kw):
        rec = {
            "audio_is_bytesio": isinstance(audio, io.BytesIO),
            "language": language,
            "beam_size": beam_size,
            "temperature": temperature,
        }
        self.calls.append(rec)
        return iter(FakeWhisperModel.segments), {"language": language}


# ---------------------------------------------------------------------------
# Deterministic fake VAD (no neural network)
# ---------------------------------------------------------------------------

class FakeVADStream(VADStream):
    def __init__(self, vad: VAD):
        super().__init__(vad)
        self._frames: list[rtc.AudioFrame] = []
        self._speaking = False

    async def _main_task(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if self._speaking and self._frames:
                    self._event_ch.send_nowait(self._make_event(VADEventType.END_OF_SPEECH))
                    self._speaking = False
                self._frames.clear()
                continue
            self._frames.append(item)
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(self._make_event(VADEventType.START_OF_SPEECH))

    def _make_event(self, ev_type):
        return VADEvent(
            type=ev_type,
            samples_index=0,
            timestamp=0.0,
            speech_duration=0.0,
            silence_duration=0.0,
            frames=[rtc.AudioFrame(
                data=b"".join(f.data for f in self._frames),
                sample_rate=self._frames[0].sample_rate,
                num_channels=1,
                samples_per_channel=sum(f.samples_per_channel for f in self._frames),
            )],
            probability=1.0,
            inference_duration=0.0,
            speaking=(ev_type == VADEventType.START_OF_SPEECH),
            raw_accumulated_silence=0.0,
            raw_accumulated_speech=0.0,
        )


class DeterministicVAD(VAD):
    def __init__(self):
        super().__init__(capabilities=VADCapabilities(update_interval=0.032))

    @property
    def model(self) -> str:
        return "fake-vad"

    @property
    def provider(self) -> str:
        return "fake"

    def stream(self) -> VADStream:
        return FakeVADStream(self)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class FakeAudioTrack:
    def __init__(self, kind=rtc.TrackKind.KIND_AUDIO, sid="atrack-1"):
        self.kind = kind
        self.sid = sid


def _mk_frame(n=320, sample_rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(-4000, 4000, size=n).astype(np.int16)
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=n,
    )


def _frame_source(*, count=3):
    """Async-iterable of synthetic AudioFrames -> StreamAdapter (fake track source)."""

    async def gen():
        for i in range(count):
            yield _mk_frame(seed=i)

    return gen()


def _frame_event_source(*, count=3):
    """Async-iterable yielding rtc.AudioFrameEvent wrappers, like a real
    AudioStream does (current livekit.rtc yields AudioFrameEvent, not raw
    AudioFrame). The real audio frame is `event.frame`."""

    async def gen():
        for i in range(count):
            yield rtc.AudioFrameEvent(frame=_mk_frame(seed=i))

    return gen()


def _make_stt():
    return FasterWhisperSTT(
        language="ru",
        initial_prompt="Ангиография термины",
        _model_factory=FakeWhisperModel,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_segments():
    FakeWhisperModel.segments = [
        SimpleNamespace(text="Калибровка"),
        SimpleNamespace(text=" катетера"),
    ]
    yield


def test_audio_track_detected_and_processed():
    """Audio track detection starts a task; END/transcription delivered."""
    stt = _make_stt()
    vad = DeterministicVAD()
    finals = []
    track = FakeAudioTrack()

    async def run():
        source = _frame_source(count=3)
        transcriber = AudioTranscriber(
            track=track,
            stt=stt,
            vad=vad,
            on_final=lambda text: finals.append(text),
        )
        # Replace the audio source factory with our synthetic one.
        transcriber._audio_stream_factory = lambda *, track: source
        await transcriber.run()

    asyncio.run(run())

    assert finals == ["Калибровка катетера"]
    assert stt._model.calls, "STT model was not invoked"
    assert stt._model.calls[0]["audio_is_bytesio"] is True


def test_non_audio_track_ignored():
    """Non-audio (video) tracks are ignored by the listener."""
    stt = _make_stt()
    vad = DeterministicVAD()
    listener = AudioTrackListener(
        stt_factory=lambda: stt,
        vad_factory=lambda: vad,
    )
    video_track = FakeAudioTrack(kind=rtc.TrackKind.KIND_VIDEO, sid="v-1")

    async def run():
        listener.on_track_subscribed(video_track)
        # No task should be created for a non-audio track.
        return len(listener._tasks)

    n = asyncio.run(run())
    assert n == 0
    assert listener.stt is None  # models not built for non-audio


def test_audio_frames_forwarded_to_stt_stream_and_final_handled():
    """Frames are forwarded through the StreamAdapter -> FINAL_TRANSCRIPT."""
    stt = _make_stt()
    vad = DeterministicVAD()
    finals = []
    track = FakeAudioTrack(sid="a-1")

    async def run():
        source = _frame_source(count=3)
        transcriber = AudioTranscriber(
            track=track,
            stt=stt,
            vad=vad,
            on_final=lambda text: finals.append(text),
        )
        transcriber._audio_stream_factory = lambda *, track: source
        await transcriber.run()

    asyncio.run(run())

    assert finals == ["Калибровка катетера"]
    # The STT transcribe saw merged audio (BytesIO) with baseline decoding opts.
    assert stt._model.calls[0]["beam_size"] == 1
    assert stt._model.calls[0]["temperature"] == 0


def test_audio_frame_event_is_unwrapped_before_push_frame():
    """Regression: AudioStream yields AudioFrameEvent, not AudioFrame.

    The real frame must be unwrapped via event.frame before push_frame. Use a
    fake source that yields rtc.AudioFrameEvent wrappers and verify the inner
    AudioFrame reaches the STT (producing the final transcript).
    """
    stt = _make_stt()
    vad = DeterministicVAD()
    finals = []
    track = FakeAudioTrack(sid="a-event")

    async def run():
        # Yields rtc.AudioFrameEvent(frame=...), like the real AudioStream.
        source = _frame_event_source(count=3)
        transcriber = AudioTranscriber(
            track=track,
            stt=stt,
            vad=vad,
            on_final=lambda text: finals.append(text),
        )
        transcriber._audio_stream_factory = lambda *, track: source
        await transcriber.run()

    asyncio.run(run())

    # The AudioFrameEvent wrappers were unwrapped and the inner AudioFrames
    # flowed through to the STT, yielding the joined final transcript.
    assert finals == ["Калибровка катетера"]
    assert stt._model.calls, "STT model was not invoked"
    assert stt._model.calls[0]["audio_is_bytesio"] is True


def test_listener_unsubscribe_cancels_and_removes_task():
    """on_track_unsubscribed cancels & removes the per-track processing task."""
    stt = _make_stt()
    vad = DeterministicVAD()
    listener = AudioTrackListener(
        stt_factory=lambda: stt,
        vad_factory=lambda: vad,
        audio_stream_factory=lambda *, track: _frame_source(count=1),
    )
    cancelled = []
    started = asyncio.Event()

    async def long_running():
        try:
            started.set()
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def run():
        # Simulate a processing task already registered for this track.
        task = asyncio.create_task(long_running())
        listener._tasks["a-9"] = task
        await started.wait()  # ensure the coroutine has begun awaiting

        assert "a-9" in listener._tasks
        listener.on_track_unsubscribed(FakeAudioTrack(sid="a-9"))
        await asyncio.sleep(0.02)

        # Task removed from the registry and cancelled.
        assert "a-9" not in listener._tasks
        assert task.cancelled()

    asyncio.run(run())
    assert cancelled  # the observer task actually got cancelled


def test_listener_cancel_all_clears_all_tasks():
    """cancel_all cancels every registered task (used on disconnect)."""
    listener = AudioTrackListener(stt_factory=_make_stt, vad_factory=lambda: DeterministicVAD())
    cancelled = []
    n_started = 0
    all_started = asyncio.Event()

    async def long_running():
        nonlocal n_started
        try:
            n_started += 1
            if n_started == 2:
                all_started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def run():
        for sid in ("a-1", "a-2"):
            listener._tasks[sid] = asyncio.create_task(long_running())
        await all_started.wait()  # both coroutines have begun awaiting
        await listener.cancel_all()
        assert listener._tasks == {}

    asyncio.run(run())
    assert len(cancelled) == 2


def test_background_task_exception_logged(caplog):
    """_log_task_failure logs the exception of a failed background task."""
    async def boom():
        raise RuntimeError("boom-stt")

    async def run():
        task = asyncio.create_task(boom())
        await asyncio.sleep(0)
        _log_task_failure(task, "audio track a-1")
        await task

    with caplog.at_level(logging.ERROR, logger="voice.audio_transcriber"):
        try:
            asyncio.run(run())
        except RuntimeError:
            pass  # expected: the raw task failure still propagates if awaited

    assert any(
        "boom-stt" in r.message for r in caplog.records
    )


def test_existing_rag_text_path_still_works():
    """voice.agent's RAG text path (error mapping / helpers) is intact."""
    from voice import agent as voice_agent
    from voice.rag_client import RagTimeoutError, RagMalformedResponseError

    assert callable(voice_agent.handle_rag_error)
    # Maps known RAG errors to the friendly Russian fallback (not empty).
    assert voice_agent.handle_rag_error(RagTimeoutError("t")) != ""
    assert voice_agent.handle_rag_error(RagMalformedResponseError("m")) != ""
    # The conversation/answer helpers still exist.
    assert callable(voice_agent._answer)


def _participant(identity="caller-1"):
    return SimpleNamespace(identity=identity)


def test_listener_on_final_receives_transcript_and_participant():
    """Phase D2.1: per-track FINAL_TRANSCRIPT is routed to the session
    `on_final(text, participant)` so the voice agent can answer the caller.

    Uses the same fake STT / deterministic VAD / synthetic frames as the other
    integration tests; the participant bound at subscribe time must reach the
    session-level on_final callback together with the transcript text.
    """
    stt = _make_stt()
    vad = DeterministicVAD()
    seen = []
    participant = _participant("voice-caller-1")

    listener = AudioTrackListener(
        stt_factory=lambda: stt,
        vad_factory=lambda: vad,
        audio_stream_factory=lambda *, track: _frame_source(count=3),
        on_final=lambda text, who: seen.append((text, who)),
    )

    track = FakeAudioTrack(sid="a-D2")

    async def run():
        listener.on_track_subscribed(track, participant=participant)
        # Let the processing task consume the synthetic frames and finish.
        await asyncio.sleep(0.2)

    asyncio.run(run())

    assert len(seen) == 1
    text, who = seen[0]
    assert text == "Калибровка катетера"
    # The participant that produced the audio is attributed to the transcript.
    assert who is participant


def test_listener_on_final_defaults_to_none():
    """Backward compatibility: AudioTrackListener without on_final works."""
    stt = _make_stt()
    vad = DeterministicVAD()
    listener = AudioTrackListener(
        stt_factory=lambda: stt,
        vad_factory=lambda: vad,
        audio_stream_factory=lambda *, track: _frame_source(count=2),
    )

    track = FakeAudioTrack(sid="a-default")

    async def run():
        listener.on_track_subscribed(track, participant=_participant())
        await asyncio.sleep(0.2)

    asyncio.run(run())


