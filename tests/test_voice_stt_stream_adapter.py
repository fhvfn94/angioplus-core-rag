# -*- coding: utf-8 -*-
"""Integration test: LiveKit StreamAdapter + Silero-compatible VAD + FasterWhisperSTT.

This proves the CURRENT livekit-agents==1.7.0 composition:

    synthetic rtc.AudioFrame(s)
        -> StreamAdapter (drives a VAD)
        -> VAD emits START_OF_SPEECH / END_OF_SPEECH
        -> on END_OF_SPEECH StreamAdapter calls FasterWhisperSTT.recognize()
        -> FINAL_TRANSCRIPT SpeechEvent

No real microphone, no /ask/RAG, no LiveKit Server. To keep the integration
test deterministic and free of neural-network dependence, a fake VAD is used
that emits deterministic speech-segment events from the pushed frames. The
faster-whisper model is also faked. This validates the integration *contract*
(StreamAdapter composes our offline STT after VAD end-of-turn), not Silero
accuracy -- which is covered by the VAD loader test separately.
"""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import agents, rtc
from livekit.agents.stt import StreamAdapter
from livekit.agents.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream

from voice.stt import FasterWhisperSTT


# ---------------------------------------------------------------------------
# Fake STT model (records transcribe kwargs, returns configurable segments)
# ---------------------------------------------------------------------------

class FakeWhisperModel:
    segments = [SimpleNamespace(text="Калибровка"), SimpleNamespace(text=" катетера")]
    calls = []

    def __init__(self, model, device, compute_type):
        self.model_arg = model
        self.device_arg = device
        self.compute_type_arg = compute_type
        type(self).constructed_args.append((model, device, compute_type))
        self.calls = []

    def transcribe(self, audio, language=None, initial_prompt=None,
                   beam_size=None, best_of=None, temperature=None, **kw):
        rec = {
            "audio_is_bytesio": isinstance(audio, io.BytesIO),
            "language": language,
            "initial_prompt": initial_prompt,
            "beam_size": beam_size,
            "best_of": best_of,
            "temperature": temperature,
        }
        self.calls.append(rec)
        return iter(type(self).segments), {"language": language}


# ---------------------------------------------------------------------------
# Deterministic fake VAD
# ---------------------------------------------------------------------------

class FakeVADStream(VADStream):
    """Emitted START on first frame, END on flush/end-of-input, carrying the
    frames received between segment boundaries."""

    def __init__(self, vad: VAD):
        super().__init__(vad)
        self._frames: list[rtc.AudioFrame] = []
        self._speaking = False

    async def _main_task(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                # Hard boundary: if we were speaking, close out the segment.
                if self._speaking and self._frames:
                    self._event_ch.send_nowait(self._make_event(VADEventType.END_OF_SPEECH))
                    self._speaking = False
                self._frames.clear()
                continue

            frame = item
            self._frames.append(frame)
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
        super().__init__(
            capabilities=VADCapabilities(update_interval=0.032)
        )

    @property
    def model(self) -> str:
        return "fake-vad"

    @property
    def provider(self) -> str:
        return "fake"

    def stream(self) -> VADStream:
        return FakeVADStream(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stt():
    return FasterWhisperSTT(
        language="ru",
        initial_prompt="Ангиография термины",
        _model_factory=FakeWhisperModel,
    )


def _mk_frame(n=320, sample_rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(-4000, 4000, size=n).astype(np.int16)
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=n,
    )


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeWhisperModel.constructed_args = []
    FakeWhisperModel.calls = []
    FakeWhisperModel.segments = [SimpleNamespace(text="Калибровка"), SimpleNamespace(text=" катетера")]
    yield


def test_stream_adapter_composes_stt_and_vad_into_final_transcript():
    stt = _make_stt()
    vad = DeterministicVAD()
    adapter = StreamAdapter(stt=stt, vad=vad)

    # StreamAdapter becomes a streaming STT capable of FINAL_TRANSCRIPT.
    assert adapter.capabilities.streaming is True
    assert adapter.wrapped_stt is stt

    events = asyncio.run(_drive_stream(adapter))

    types = [e.type for e in events]
    assert agents.stt.SpeechEventType.START_OF_SPEECH in types
    assert agents.stt.SpeechEventType.END_OF_SPEECH in types
    assert agents.stt.SpeechEventType.FINAL_TRANSCRIPT in types

    final = next(
        e for e in events
        if e.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT
    )
    # Our offline STT was invoked through the wrapped FasterWhisperSTT.
    assert final.alternatives[0].text == "Калибровка катетера"
    # And the STT model saw the merged audio as a BytesIO WAV.
    assert stt._model.calls[0]["audio_is_bytesio"] is True
    assert stt._model.calls[0]["beam_size"] == 1
    assert stt._model.calls[0]["temperature"] == 0


async def _drive_stream(adapter: StreamAdapter):
    """Drive the adapter stream entirely inside one event loop.

    The StreamAdapter/VADStream channels bind to the current event loop at
    construction time, so the stream must be created, fed and drained within a
    single running loop.
    """
    stream = adapter.stream()
    # Push a short synthetic utterance, then signal end of input which drives
    # the VAD to emit END_OF_SPEECH.
    for i in range(3):
        stream.push_frame(_mk_frame(seed=i))

    stream.end_input()

    return [ev async for ev in stream]
