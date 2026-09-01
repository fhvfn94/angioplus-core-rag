# -*- coding: utf-8 -*-
"""Human-run STT benchmark helper for AngioPlus Russian speech (Phase D1.5).

This helper evaluates the SAME faster-whisper configuration used by the
LiveKit voice path (voice/stt.py FasterWhisperSTT) against real WAV audio and
an expected phrase. It does NOT require LiveKit Server: it builds an
rtc.AudioFrame from the WAV and runs it through the STT hook directly.

It is NOT run by tests and does NOT record a microphone automatically.

Usage (human-run):
    # single file with an explicit expected phrase
    .\\.venv\\Scripts\\python.exe scripts\\stt_benchmark.py `
        --wav data\\calibration.wav `
        --expected "Как выполнить калибровку катетера?"

    # evaluate a whole recorded dataset: data/<id>.wav per phrase id
    .\\.venv\\Scripts\\python.exe scripts\\stt_benchmark.py `
        --dataset --audio-dir data\\recordings

    # lookup the expected phrase by id from the phrase set, run one wav
    .\\.venv\\Scripts\\python.exe scripts\\stt_benchmark.py `
        --wav data\\min_mufr.wav --id min_mufr

Output per file:
    expected, transcript, CER%, WER%, audio_duration, inference_time,
    realtime_factor (inference/audio, <1 = faster than realtime).

Normalization is deliberately conservative: lowercase, collapse whitespace,
strip punctuation. No fuzzy medical rewriting.

Local CER/WER use Levenshtein edit distance; no extra dependencies are added.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import wave
from pathlib import Path

import numpy as np

from livekit import agents, rtc

# Allow running directly as `python scripts/stt/stt_benchmark.py` from anywhere:
# put the repo root (parent of scripts/) on sys.path so `voice` and
# project modules are importable without installing the package.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from voice.stt import FasterWhisperSTT  # noqa: E402
from voice.stt_phrases import STT_PHRASES  # noqa: E402


# ---------------------------------------------------------------------------
# WAV -> LiveKit AudioFrame (frames are normalized by _buffer_to_mono16k_wav)
# ---------------------------------------------------------------------------

def load_wav_frame(path: str) -> rtc.AudioFrame:
    """Read a PCM WAV file into an rtc.AudioFrame (any rate/channels)."""
    with wave.open(path, "rb") as wav:
        params = wav.getparams()
        nchannels = params.nchannels
        sampwidth = params.sampwidth
        nframes = params.nframes
        raw = wav.readframes(nframes)

    if sampwidth != 2:
        raise SystemExit(
            f"Unsupported sample width {sampwidth*8}-bit in {path}; "
            "only 16-bit PCM WAV is supported."
        )

    data = np.frombuffer(raw, dtype=np.int16).tobytes()
    return rtc.AudioFrame(
        data=data,
        sample_rate=params.framerate,
        num_channels=nchannels,
        samples_per_channel=nframes,
    )


# ---------------------------------------------------------------------------
# Conservative normalization + CER/WER (local Levenshtein)
# ---------------------------------------------------------------------------

_TRANSLIT_PUNCT = str.maketrans("", "", ".,!?;:()«»\"'-–—…")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation (conservative)."""
    text = text.translate(_TRANSLIT_PUNCT)
    return " ".join(text.lower().split())


def _levenshtein(a: str, b: str) -> int:
    ra, rb = len(a), len(b)
    if ra == 0:
        return rb
    if rb == 0:
        return ra
    prev = list(range(rb + 1))
    for i in range(1, ra + 1):
        cur = [i]
        for j in range(1, rb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[rb]


def _cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _expected_by_id(phrase_id: str) -> str | None:
    for p in STT_PHRASES:
        if p["id"] == phrase_id:
            return p["text"]
    return None


async def _run_stt(stt: FasterWhisperSTT, frame: rtc.AudioFrame) -> str:
    event = await stt._recognize_impl(
        frame,
        conn_options=agents.APIConnectOptions(),
    )
    return event.alternatives[0].text if event.alternatives else ""


def benchmark_one(
    stt: FasterWhisperSTT,
    *,
    path: str,
    expected: str | None,
    phrase_id: str | None,
) -> dict:
    frame = load_wav_frame(path)
    transcript = asyncio.run(_run_stt(stt, frame))

    if expected is None and phrase_id is not None:
        expected = _expected_by_id(phrase_id)

    return {
        "id": phrase_id,
        "path": path,
        "expected": expected,
        "transcript": transcript,
        "cer": _cer(expected, transcript) if expected else None,
        "wer": _wer(expected, transcript) if expected else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--wav", action="append", help="WAV file(s) to transcribe")
    p.add_argument("--id", help="phrase id to look up the expected text")
    p.add_argument("--expected", help="explicit expected transcript")
    p.add_argument("--dataset", action="store_true",
                   help="evaluate all recorded data/<id>.wav in --audio-dir")
    p.add_argument("--audio-dir", default="data/recordings",
                   help="directory of <phrase_id>.wav files for --dataset")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    stt = FasterWhisperSTT()  # same configuration as the voice path

    targets = []
    if args.dataset:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.is_dir():
            raise SystemExit(f"audio dir not found: {audio_dir}")
        for phrase in STT_PHRASES:
            wav = audio_dir / f"{phrase['id']}.wav"
            if wav.is_file():
                targets.append((str(wav), phrase["text"], phrase["id"]))
        if not targets:
            raise SystemExit(
                f"No <id>.wav files found in {audio_dir}. "
                "Record the phrase set first (one file per id), then re-run."
            )
    elif args.wav:
        for wav_path in args.wav:
            expected = args.expected
            if expected is None and args.id:
                expected = _expected_by_id(args.id)
            targets.append((wav_path, expected, args.id))
    else:
        raise SystemExit("Provide --wav (and --id/--expected) or --dataset.")

    print(f"model={stt._model_name} device={stt._device} "
          f"compute_type={stt._compute_type} language={stt._language or 'auto'}")
    if stt._initial_prompt:
        print(f"initial_prompt={(stt._initial_prompt[:60] + '...') if len(stt._initial_prompt) > 60 else stt._initial_prompt}")

    for path, expected, phrase_id in targets:
        if not os.path.isfile(path):
            print(f"[skip] missing {path}")
            continue
        result = benchmark_one(stt, path=path, expected=expected, phrase_id=phrase_id)
        print("----")
        print(f"id        : {result['id']}")
        print(f"file      : {result['path']}")
        print(f"expected  : {result['expected']}")
        print(f"transcript: {result['transcript']}")
        if result["expected"]:
            print(f"CER       : {result['cer']:.3f} ({result['cer']*100:.1f}%)")
            print(f"WER       : {result['wer']:.3f} ({result['wer']*100:.1f}%)")
        else:
            print("CER/WER   : (no expected text provided)")


if __name__ == "__main__":
    main()
