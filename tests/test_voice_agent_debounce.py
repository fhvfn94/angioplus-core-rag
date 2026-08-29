# -*- coding: utf-8 -*-
"""Unit tests for the per-participant voice turn debouncer (voice/agent.py).

These tests exercise ``_VoiceTurnDebouncer`` directly so we can verify the
temporal turn-aggregation requirements without a running LiveKit job:

  - one FINAL_TRANSCRIPT -> exactly one RAG flush after the debounce window
  - two fragments inside the window -> one combined RAG flush
  - fragments separated beyond the window -> two RAG flushes
  - two participants aggregate independently
  - empty transcripts are ignored
  - cleanup cancels pending debounce timers (nothing flushed on shutdown)

The flush "RAG call" is a recording async callback; participants are fakes with
an ``identity`` attribute (mirroring rtc.RemoteParticipant), so no LiveKit
infrastructure is needed.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from voice.agent import VOICE_TURN_DEBOUNCE_SECONDS, _VoiceTurnDebouncer

# Tiny debounce window so tests are fast while still exercising real timers.
_DEBOUNCE = 0.05


class _FakeParticipant:
    def __init__(self, identity: str):
        self.identity = identity


class _Recorder:
    """Awaitable stand-in for ``_answer_and_publish``; records RAG calls."""

    def __init__(self):
        self.calls = []  # list of (text, participant)

    async def flush(self, text: str, participant) -> None:
        self.calls.append((text, participant))


def _make_debouncer(rec: _Recorder) -> _VoiceTurnDebouncer:
    # Use the default spawn (async loop task) with a tiny debounce window.
    return _VoiceTurnDebouncer(debounce_seconds=_DEBOUNCE)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# One FINAL_TRANSCRIPT -> one RAG call after the debounce
# ---------------------------------------------------------------------------

def test_single_fragment_flushes_once_after_debounce():
    rec = _Recorder()
    d = _make_debouncer(rec)
    participant = _FakeParticipant("alice")

    async def go():
        d.submit("кто делать если", participant, rec.flush)
        # Not flushed yet while a newer fragment may still land.
        assert d.pending_count() == 1
        await asyncio.sleep(_DEBOUNCE * 3)
        assert rec.calls == [("кто делать если", participant)]
        # State cleared after flush.
        assert d.pending_count() == 0

    _run(go())


# ---------------------------------------------------------------------------
# Two fragments inside the window -> one combined RAG call
# ---------------------------------------------------------------------------

def test_two_fragments_inside_window_combine_into_one_call():
    rec = _Recorder()
    d = _make_debouncer(rec)
    participant = _FakeParticipant("alice")

    async def go():
        d.submit("кто делать если", participant, rec.flush)
        d.submit("программа не может определить правильно сосуд",
                 participant, rec.flush)
        assert d.pending_count() == 1
        # Both fragments inside the debounce -> single combined RAG call.
        await asyncio.sleep(_DEBOUNCE * 3)
        assert len(rec.calls) == 1
        assert rec.calls[0][0] == (
            "кто делать если программа не может определить правильно сосуд"
        )
        assert rec.calls[0][1] is participant

    _run(go())


# ---------------------------------------------------------------------------
# Fragments beyond the debounce -> two RAG calls
# ---------------------------------------------------------------------------

def test_fragments_separated_beyond_debounce_flush_twice():
    rec = _Recorder()
    d = _make_debouncer(rec)
    participant = _FakeParticipant("alice")

    async def go():
        d.submit("оба вопроса", participant, rec.flush)
        # Enough time that the first turn would have flushed.
        await asyncio.sleep(_DEBOUNCE * 3)
        assert len(rec.calls) == 1
        d.submit("и второй", participant, rec.flush)
        await asyncio.sleep(_DEBOUNCE * 3)
        assert len(rec.calls) == 2
        assert rec.calls[0][0] == "оба вопроса"
        assert rec.calls[1][0] == "и второй"

    _run(go())


# ---------------------------------------------------------------------------
# Two participants aggregate independently
# ---------------------------------------------------------------------------

def test_participants_aggregate_independently():
    rec = _Recorder()
    d = _make_debouncer(rec)
    alice = _FakeParticipant("alice")
    bob = _FakeParticipant("bob")

    async def go():
        d.submit("вопрос алисы", alice, rec.flush)
        d.submit("вопрос боба самый", bob, rec.flush)
        d.submit("длинный", bob, rec.flush)
        assert d.pending_count() == 2  # separate buffers per participant

        # If alice's timer fires alone, only her text is flushed; bob's two
        # fragments must combine into one separate call.
        await asyncio.sleep(_DEBOUNCE * 3)
        assert len(rec.calls) == 2
        calls_by_text = {t: 0 for t, _ in rec.calls}
        calls_by_text["вопрос алисы"] += 1
        calls_by_text["вопрос боба самый длинный"] += 1
        assert calls_by_text == {
            "вопрос алисы": 1,
            "вопрос боба самый длинный": 1,
        }
        assert d.pending_count() == 0

    _run(go())


# ---------------------------------------------------------------------------
# Empty transcripts ignored
# ---------------------------------------------------------------------------

def test_empty_transcript_ignored():
    rec = _Recorder()
    d = _make_debouncer(rec)
    participant = _FakeParticipant("alice")

    async def go():
        d.submit("   ", participant, rec.flush)
        d.submit("", participant, rec.flush)
        d.submit(None, participant, rec.flush)
        assert d.pending_count() == 0
        await asyncio.sleep(_DEBOUNCE * 3)
        assert rec.calls == []

    _run(go())


# ---------------------------------------------------------------------------
# Cleanup cancels pending debounce timers (nothing flushed on shutdown)
# ---------------------------------------------------------------------------

def test_cleanup_cancels_pending_timers():
    rec = _Recorder()
    d = _make_debouncer(rec)
    participant = _FakeParticipant("alice")

    async def go():
        d.submit("никогда не должен отправиться", participant, rec.flush)
        assert d.pending_count() == 1
        d.flush_all()  # session shutdown
        assert d.pending_count() == 0
        # Wait well past the debounce: the cancelled timer must not flush.
        await asyncio.sleep(_DEBOUNCE * 5)
        assert rec.calls == []

    _run(go())


def test_constant_debounce_is_eight_tenths_of_a_second():
    assert VOICE_TURN_DEBOUNCE_SECONDS == 0.8
