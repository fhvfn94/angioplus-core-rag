# -*- coding: utf-8 -*-
"""Unit tests for the LiveKit faster-whisper STT layer (voice/stt.py).

These tests do NOT download or load a real Whisper model. A fake model is
injected via `FasterWhisperSTT(_model_factory=...)` so `voice.stt` is exercised
against the real LiveKit SDK / rtc.AudioFrame types without the heavy library.
"""
from __future__ import annotations

import asyncio
import io
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import agents, rtc

from voice import stt as voice_stt
from voice.stt import FasterWhisperSTT, WHISPER_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class FakeWhisperModel:
    """Records transcribe kwargs and returns configurable segments."""

    constructed = []  # reset per test
    segments = []  # reset per test
    raise_on_transcribe = None
    # When > 0, transcribe() returns a *lazy* generator that sleeps this long
    # per segment during iteration — modelling faster-whisper's lazy decoding.
    lazy_sleep = 0.0

    def __init__(self, model, device, compute_type):
        self.model_arg = model
        self.device_arg = device
        self.compute_type_arg = compute_type
        type(self).constructed.append((model, device, compute_type))
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
            "thread_id": threading.current_thread().ident,
        }
        self.calls.append(rec)
        if type(self).raise_on_transcribe is not None:
            raise type(self).raise_on_transcribe("boom")
        if type(self).lazy_sleep > 0:
            # A lazy generator: real "decoding" work happens only when the
            # consumer iterates it (as with faster-whisper's generate_segments).
            segs = type(self).segments

            def _lazy():
                for s in segs:
                    time.sleep(type(self).lazy_sleep)
                    yield s

            return _lazy(), {"language": language}
        return iter(type(self).segments), {"language": language}


def _make_stt(**kwargs) -> FasterWhisperSTT:
    return FasterWhisperSTT(_model_factory=FakeWhisperModel, **kwargs)


def _mk_frame(sample_rate=16000, channels=1, n=320, seed=0):
    if channels > 1:
        rng = np.random.default_rng(seed)
        data = rng.integers(-8000, 8000, size=n * channels).astype(np.int16)
    else:
        data = (np.arange(n, dtype=np.int16) * 2)
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=n,
    )


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeWhisperModel.constructed.clear()
    FakeWhisperModel.segments = [
        SimpleNamespace(text="Калибровка"),
        SimpleNamespace(text=" катетера"),
        SimpleNamespace(text="."),
    ]
    FakeWhisperModel.raise_on_transcribe = None
    FakeWhisperModel.lazy_sleep = 0.0
    yield


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_defaults_match_telegram_stt():
    stt = FasterWhisperSTT()
    # defaults mirror bot/services/stt.py
    assert not stt.is_model_loaded
    assert stt._model_name == "small"
    assert stt._device == "cpu"
    assert stt._compute_type == "int8"
    # Phase D1.5: the voice path defaults to Russian (no auto-detect) and a
    # concise AngioPlus vocabulary initial_prompt is active by default.
    assert stt._language == "ru"
    assert stt._initial_prompt and "катетер" in stt._initial_prompt


def test_default_voice_config_forces_russian():
    """The default (env) config forwards language='ru' to faster-whisper,
    skipping automatic language detection (Phase D1.5 TASK 1)."""
    stt = _make_stt()  # no language kwarg -> module default "ru"
    _recognize_once(stt)
    assert stt._model.calls[0]["language"] == "ru"


def test_default_initial_prompt_is_angioplus_vocab():
    """The default initial_prompt carries AngioPlus vocabulary (TASK 2)."""
    stt = _make_stt()  # no initial_prompt kwarg -> module default prompt
    _recognize_once(stt)
    call = stt._model.calls[0]["initial_prompt"]
    assert call
    for token in ("катетер", "μFR", "ангиография", "AngioPlus"):
        assert token in call


def test_initial_prompt_can_be_disabled():
    """initial_prompt=None disables conditioning (WHISPER_INITIAL_PROMPT empty
    maps to None in the constructor)."""
    stt = _make_stt(initial_prompt=None)
    _recognize_once(stt)
    assert stt._model.calls[0]["initial_prompt"] is None


def test_custom_constructor_overrides(fake=None):
    stt = FasterWhisperSTT(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        language="ru",
        initial_prompt="Специальный медицинский словарь",
    )
    assert stt.model == "large-v3"
    assert stt.provider == "faster-whisper"
    assert stt._device == "cuda"
    assert stt._compute_type == "float16"
    assert stt._language == "ru"
    assert stt._initial_prompt == "Специальный медицинский словарь"


def test_capabilities_are_offline_non_streaming():
    stt = _make_stt()
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False
    assert stt.capabilities.offline_recognize is True


# ---------------------------------------------------------------------------
# Lazy load + reuse
# ---------------------------------------------------------------------------

def test_model_is_lazy_and_not_loaded_at_construction():
    stt = _make_stt()
    assert stt.is_model_loaded is False
    assert FakeWhisperModel.constructed == []


def test_model_reused_across_recognitions():
    stt = _make_stt()
    loop = asyncio.new_event_loop()

    async def go():
        f = _mk_frame()
        await stt._recognize_impl(f, conn_options=agents.APIConnectOptions())
        await stt._recognize_impl(f, conn_options=agents.APIConnectOptions())

    try:
        loop.run_until_complete(go())
    finally:
        loop.close()

    # Only one model constructed, two transcribe calls on the same instance.
    assert FakeWhisperModel.constructed == [("small", "cpu", "int8")]
    model = stt._model
    assert model is not None
    assert len(model.calls) == 2
    assert stt.is_model_loaded is True


# ---------------------------------------------------------------------------
# Transcribe options forwarding
# ---------------------------------------------------------------------------

def _recognize_once(stt, frame=None, *, language=None):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            stt._recognize_impl(
                frame if frame is not None else _mk_frame(),
                language=language,
                conn_options=agents.APIConnectOptions(),
            )
        )
    finally:
        loop.close()


def test_language_from_config_is_forwarded():
    stt = _make_stt(language="ru", initial_prompt="prompt")
    _recognize_once(stt)
    model = stt._model
    assert model.calls[0]["language"] == "ru"


def test_explicit_language_argument_overrides_config():
    stt = _make_stt(language="ru")
    _recognize_once(stt, language="uk")
    assert stt._model.calls[0]["language"] == "uk"


def test_initial_prompt_is_forwarded():
    stt = _make_stt(initial_prompt="AngioPlus термины")
    _recognize_once(stt)
    assert stt._model.calls[0]["initial_prompt"] == "AngioPlus термины"


def test_baseline_decoding_options_are_used():
    stt = _make_stt()
    _recognize_once(stt)
    call = stt._model.calls[0]
    assert call["beam_size"] == 1
    assert call["best_of"] == 1
    assert call["temperature"] == 0


def test_audio_passed_as_inmemory_bytesio():
    stt = _make_stt()
    _recognize_once(stt)
    assert stt._model.calls[0]["audio_is_bytesio"] is True


# ---------------------------------------------------------------------------
# Transcript joining + FINAL_TRANSCRIPT structure
# ---------------------------------------------------------------------------

def test_segment_joining():
    stt = _make_stt()
    event = _recognize_once(stt)
    assert event.alternatives[0].text == "Калибровка катетера."


def test_returns_final_transcript_event():
    stt = _make_stt()
    event = _recognize_once(stt)
    assert event.type.name == "FINAL_TRANSCRIPT"
    assert len(event.alternatives) == 1
    assert event.request_id  # non-empty uuid
    assert event.alternatives[0].language == "auto" or event.alternatives[0].language


def test_empty_transcript_raises():
    stt = _make_stt()
    FakeWhisperModel.segments = []
    with pytest.raises(ValueError, match="empty transcription"):
        _recognize_once(stt)


def test_model_exception_propagates():
    stt = _make_stt()
    FakeWhisperModel.raise_on_transcribe = RuntimeError
    with pytest.raises(RuntimeError, match="boom"):
        _recognize_once(stt)


# ---------------------------------------------------------------------------
# Audio handling (normalization)
# ---------------------------------------------------------------------------

def test_mono16k_frame_passthrough_to_wav():
    f = _mk_frame(sample_rate=16000, channels=1, n=320)
    wav = voice_stt._buffer_to_mono16k_wav(f)
    assert wav[:4] == b"RIFF"  # valid WAV header


def test_list_of_frames_merged():
    f1 = _mk_frame(sample_rate=16000, channels=1, n=320)
    f2 = _mk_frame(sample_rate=16000, channels=1, n=320)
    wav = voice_stt._buffer_to_mono16k_wav([f1, f2])
    assert wav[:4] == b"RIFF"


def test_stereo_resampled_to_mono16k():
    f = _mk_frame(sample_rate=48000, channels=2, n=960)
    wav = voice_stt._buffer_to_mono16k_wav(f)
    assert wav[:4] == b"RIFF"
    # header frame rate
    import struct
    fmt = struct.unpack("<HHIIHH", wav[20:36])
    assert fmt[2] == WHISPER_SAMPLE_RATE  # sample rate field
    assert fmt[0] == 1  # mono channel


def test_recognize_impl_accepts_audioframe_and_list():
    stt = _make_stt()
    # single frame
    e1 = _recognize_once(stt, _mk_frame(sample_rate=16000, channels=1))
    # list of frames
    e2 = _recognize_once(stt, [_mk_frame(seed=1), _mk_frame(seed=2)])
    assert e1.alternatives[0].text == e2.alternatives[0].text == "Калибровка катетера."


# ---------------------------------------------------------------------------
# Blocking inference runs off the event loop thread
# ---------------------------------------------------------------------------

def test_inference_runs_off_event_loop_thread():
    stt = _make_stt()
    main_thread_id = threading.current_thread().ident
    worker_thread_ids = []

    orig_transcribe = FakeWhisperModel.transcribe

    def patched_transcribe(self, *a, **kw):
        worker_thread_ids.append(threading.current_thread().ident)
        return orig_transcribe(self, *a, **kw)

    FakeWhisperModel.transcribe = patched_transcribe
    try:
        _recognize_once(stt)
    finally:
        FakeWhisperModel.transcribe = orig_transcribe

    # The blocking model call must run in a worker (to_thread) thread, not the
    # test thread that created the event loop.
    assert worker_thread_ids, "transcribe was never called"
    assert worker_thread_ids[0] != main_thread_id


# ---------------------------------------------------------------------------
# Latency instrumentation helpers (Phase D1.5 TASK 4) — deterministic, no
# wall-clock assertions.
# ---------------------------------------------------------------------------

def test_realtime_factor_formula():
    from voice.stt import _realtime_factor
    # inference == audio -> realtime factor 1.0
    assert _realtime_factor(3.0, 3.0) == 1.0
    # faster than realtime
    assert _realtime_factor(1.5, 3.0) == 0.5
    # slower than realtime
    assert _realtime_factor(6.0, 3.0) == 2.0
    # zero/negative audio duration -> 0.0 (no divide-by-zero)
    assert _realtime_factor(5.0, 0.0) == 0.0
    assert _realtime_factor(5.0, -2.0) == 0.0
    assert _realtime_factor(0.0, 0.0) == 0.0


def test_wav_audio_duration_parsing():
    from voice.stt import _buffer_to_mono16k_wav, _wav_audio_duration
    # 320 samples at 16 kHz = 0.02 s
    frame = _mk_frame(sample_rate=16000, channels=1, n=320)
    wav = _buffer_to_mono16k_wav(frame)
    assert abs(_wav_audio_duration(wav) - 0.02) < 1e-6


def test_stt_completed_log_contains_latency_fields(caplog):
    """Sanity: the timing log line is emitted with the expected fields."""
    import logging
    stt = _make_stt()
    with caplog.at_level(logging.INFO, logger="voice.stt"):
        _recognize_once(stt)
    joined = "\n".join(r.message for r in caplog.records)
    assert "STT started:" in joined
    assert "STT completed:" in joined
    assert "audio_duration=" in joined
    assert "inference_time=" in joined
    assert "realtime_factor=" in joined


# ---------------------------------------------------------------------------
# Phase D1.5 follow-up regression tests: lazy segments generator timing
# ---------------------------------------------------------------------------

def _parse_latency(caplog):
    """Return (inference_time, realtime_factor) from the STT completed log."""
    for rec in caplog.records:
        m = rec.getMessage()
        if m.startswith("STT completed:"):
            it = rt = None
            for part in m.split():
                if part.startswith("inference_time="):
                    it = float(part.split("=")[1].rstrip("s"))
                elif part.startswith("realtime_factor="):
                    rt = float(part.split("=")[1])
            return it, rt
    raise AssertionError("STT completed log line not found")


def test_inference_time_includes_lazy_segment_iteration(caplog):
    """Regression: inference_time must include consuming the lazy segments
    generator (the real faster-whisper decodes lazily during iteration).

    The fake model's transcribe() returns a lazy generator that sleeps
    `lazy_sleep` per segment only during iteration. An inference timer that
    stopped right after transcribe() would report ~0.0s; the fixed timer must
    report >= the total iteration work.
    """
    import logging
    iter_sleep = 0.05
    FakeWhisperModel.lazy_sleep = iter_sleep
    FakeWhisperModel.segments = [
        SimpleNamespace(text="A"), SimpleNamespace(text="B"),
    ]
    stt = _make_stt()

    with caplog.at_level(logging.INFO, logger="voice.stt"):
        _recognize_once(stt)
    inference_time, realtime_factor = _parse_latency(caplog)

    # Two segments x iter_sleep; be generous so this is never flaky, yet it
    # would clearly fail if the timer stopped before iteration (~0.0s).
    lower_bound = iter_sleep * len(FakeWhisperModel.segments) * 0.5
    assert inference_time >= lower_bound, (
        "inference_time did not include lazy segment iteration"
    )
    # RTF relationship must hold (complete transcription wall time / duration).
    assert realtime_factor >= lower_bound / 0.02


def test_cold_model_load_includes_slow_construction(caplog):
    """Regression: cold model-load time is measured around the model factory,
    so a slow construction must be captured (not trimmed away)."""
    import logging

    build_sleep = 0.1

    def slow_factory(model, device, compute_type):
        time.sleep(build_sleep)
        return FakeWhisperModel(model, device, compute_type)

    stt = FasterWhisperSTT(_model_factory=slow_factory)

    with caplog.at_level(logging.INFO, logger="voice.stt"):
        _recognize_once(stt)

    cold_msg = None
    for rec in caplog.records:
        m = rec.getMessage()
        if m.startswith("STT model loaded (cold):"):
            cold_msg = m
            break
    assert cold_msg is not None, "cold model-load log line missing"
    seconds = float(cold_msg.split(": ")[-1].rstrip("s"))
    # Robust: must be at least the factory's own construction time.
    assert seconds >= build_sleep * 0.5
