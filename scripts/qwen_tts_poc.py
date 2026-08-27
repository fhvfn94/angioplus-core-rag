# -*- coding: utf-8 -*-
"""Phase D2.2 — isolated Qwen3-TTS 0.6B proof-of-concept (Russian speech -> WAV).

This is a HUMAN-RUN, ISOLATED PoC. It does NOT touch LiveKit, the voice
agent, Silero VAD, faster-whisper, RAG, Telegram or SIP. It only exercises the
OFFICIAL Qwen3-TTS Python API:

    text -> Qwen3-TTS-12Hz-0.6B-CustomVoice -> Russian speech -> WAV

and reports latency/resource measurements so the team can decide whether
Qwen3-TTS is suitable before integrating TTS into LiveKit.

Official model / package (verified against upstream, not assumed):
    model_id : Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice   (license: apache-2.0)
    package  : qwen-tts  (from qwen_tts import Qwen3TTSModel)
    API      : model.generate_custom_voice(text, language="Russian",
                                           speaker=<speaker>, instruct="")
    language : "Russian" (model supports ru among 10 languages)
    speakers : the 9 built-in CustomVoice speakers (no native Russian one;
               any speaker can speak Russian).

Streaming note (TASK 5): the high-level `generate_custom_voice()` Python API
is NON-streaming at this level — it returns a complete `(wavs, sr)` tuple.
True time-to-first-audio (TTFA) would require the lower-level Dual-Track
streaming path (DashScope / vLLM), which is out of scope for this isolated
PoC. Therefore TTFA is NOT measured here; that is reported honestly rather
than faked.

Requirements: see requirements-qwen-tts.txt. This script must be run from a
dedicated isolated environment (not the shared production .venv) because
qwen-tts pins transformers==4.57.3, conflicting with the project's
transformers==4.57.6.

Usage (human-run, in the isolated env):
    # single utterance -> output.wav
    python scripts/qwen_tts_poc.py --text "Как выполнить калибровку катетера?" --output output.wav

    # run the whole AngioPlus TTS phrase set -> one WAV per phrase id
    python scripts/qwen_tts_poc.py --dataset --out-dir data/tts_out

    # choose a built-in speaker (default: Vivian) and pick the device
    python scripts/qwen_tts_poc.py --text "Что показывает μFR?" --speaker Ryan --device cpu
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

# Make project modules importable when run as `python scripts/qwen_tts_poc.py`
# from the repo root.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from voice.tts_phrases import (  # noqa: E402
    TTS_PHRASES,
    get_ids,
    phrase_text,
    safe_stem,
)


# ---------------------------------------------------------------------------
# Official constants (verified against upstream QwenLM/Qwen3-TTS)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_LANGUAGE = "Russian"
DEFAULT_SPEAKER = "Vivian"

# The 9 built-in speakers of the CustomVoice checkpoint. The first is the
# default. There is no native-Russian timbre; the model can still speak
# Russian with any of these voices.
SUPPORTED_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--text", help="single text to synthesize")
    p.add_argument("--id", help="phrase id from voice/tts_phrases.py to use as --text")
    p.add_argument("--output", default="output.wav",
                   help="output WAV path (single mode)")
    p.add_argument("--dataset", action="store_true",
                   help="synthesize the whole AngioPlus TTS phrase set")
    p.add_argument("--out-dir", default="data/tts_out",
                   help="output directory for --dataset")
    p.add_argument("--model", default=DEFAULT_MODEL_ID,
                   help=f"official model id (default: {DEFAULT_MODEL_ID})")
    p.add_argument("--speaker", default=DEFAULT_SPEAKER,
                   help=f"built-in speaker, one of {SUPPORTED_SPEAKERS} "
                        f"(default: {DEFAULT_SPEAKER})")
    p.add_argument("--language", default=DEFAULT_LANGUAGE,
                   help=f"language (default: {DEFAULT_LANGUAGE})")
    p.add_argument("--device", default=None,
                   help="torch device: 'cpu' or 'cuda'. Default: auto-detect "
                        "(cuda if available else cpu).")
    p.add_argument("--list-speakers", action="store_true",
                   help="print the built-in speaker list and exit")
    return p


# ---------------------------------------------------------------------------
# Measurement helpers (pure, testable)
# ---------------------------------------------------------------------------

def realtime_factor(synthesis_time: float, audio_duration: float) -> Optional[float]:
    """RTF = synthesis_time / audio_duration (None when audio_duration <= 0)."""
    if not synthesis_time or synthesis_time < 0:
        return None
    if not audio_duration or audio_duration <= 0:
        return None
    return synthesis_time / audio_duration


def format_rtf(rtf: Optional[float]) -> str:
    if rtf is None:
        return "n/a"
    return f"{rtf:.3f} ({rtf:.2f}x audio)"


def get_process_ram_mb() -> int:
    """Resident set size of this process in MB (best effort)."""
    try:
        import resource  # POSIX only
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except (ImportError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        try:
            import psutil
            return int(psutil.Process().memory_info().rss / (1024 * 1024))
        except Exception:
            pass
    return 0


def get_gpu_vram_mb() -> int:
    """Allocated VRAM in MB across all CUDA devices (best effort, 0 if none)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        total = 0
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i)
            total += int(mem)
        return int(total / (1024 * 1024))
    except Exception:
        return 0


@dataclass
class EnvironmentReport:
    """Hardware / runtime facts reported by the PoC (TASK 6)."""
    python_version: str
    platform: str
    torch_version: str = "n/a"
    torch_cuda_available: bool = False
    torch_cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    selected_device: str = "cpu"


def detect_environment(force_device: Optional[str]) -> EnvironmentReport:
    """Detect torch/CUDA and decide the device to use (never assume GPU)."""
    report = EnvironmentReport(
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    try:
        import torch
    except ImportError:
        # torch is required to run the model; report without it.
        report.selected_device = force_device or "cpu"
        return report

    report.torch_version = torch.__version__
    report.torch_cuda_available = bool(torch.cuda.is_available())
    if report.torch_cuda_available:
        report.torch_cuda_version = torch.version.cuda
        report.gpu_name = torch.cuda.get_device_name(0)

    if force_device:
        report.selected_device = force_device.lower()
    else:
        report.selected_device = "cuda" if report.torch_cuda_available else "cpu"
    return report


# ---------------------------------------------------------------------------
# Qwen model glue (heavy model is imported lazily)
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: str):
    """Load the official Qwen3TTSModel. Imports qwen_tts lazily so the pure
    helpers in this module remain importable in tests without the package."""
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:  # pragma: no cover - covered separately
        raise SystemExit(
            "qwen_tts is not installed in this environment.\n"
            "This PoC must run from an isolated venv:\n"
            "  py -3.12 -m venv .venv-qwen\n"
            f"  .venv-qwen\\Scripts\\python.exe -m pip install -r "
            f"requirements-qwen-tts.txt\n"
            f"Skipping model load (raised: {exc})"
        )

    ret = {"dtype": "n/a", "device": device}
    kwargs = {"device_map": device}
    if device == "cpu":
        kwargs["dtype"] = torch.float32
        ret["dtype"] = "float32"
    else:
        kwargs["dtype"] = torch.bfloat16
        ret["dtype"] = "bfloat16"
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
    ret["model"] = model
    return ret


def generate_one(model, text: str, language: str, speaker: str) -> dict:
    """Generate audio for a single text with the official custom-voice API.
    Returns (wavs, sr) wrapped, plus a monotonic synthesis timing inside this
    call is measured by the caller; only the API result shape is returned here.
    """
    return model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct="",
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_single(env: EnvironmentReport, args) -> None:
    print(f"Loading model {args.model} on device={env.selected_device} ...")
    t_load0 = time.perf_counter()
    ret = load_model(args.model, env.selected_device)
    model = ret["model"]
    t_load1 = time.perf_counter()
    load_seconds = t_load1 - t_load0
    print(f"Model load time: {load_seconds:.3f} s")

    text = phrase_text(args.id) if args.id else args.text

    print(f"Generating ({language_header(args, text)}): {text!r}")
    t0 = time.perf_counter()
    wavs, sr = generate_one(model, text, args.language, args.speaker)
    t1 = time.perf_counter()
    synthesis_seconds = t1 - t0

    wav = wavs[0]
    audio_duration = _duration(wav, sr)
    rtf = realtime_factor(synthesis_seconds, audio_duration)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_wav(out, wav, sr)

    print("----")
    print(f"device         : {env.selected_device}")
    print(f"model          : {args.model}")
    print(f"speaker        : {args.speaker}")
    print(f"language       : {args.language}")
    print(f"sample_rate    : {sr}")
    print(f"load_time      : {load_seconds:.3f} s")
    print(f"synthesis_time : {synthesis_seconds:.3f} s")
    print(f"audio_duration : {audio_duration:.3f} s")
    print(f"realtime_factor: {format_rtf(rtf)}")
    print(f"process_ram_mb : {get_process_ram_mb()}")
    print(f"gpu_vram_mb    : {get_gpu_vram_mb()} (0 = none/not measured)")
    print(f"output         : {out}")
    _print_ttfa_note()


def run_dataset(env: EnvironmentReport, args) -> None:
    print(f"Loading model {args.model} on device={env.selected_device} ...")
    t_load0 = time.perf_counter()
    ret = load_model(args.model, env.selected_device)
    model = ret["model"]
    t_load1 = time.perf_counter()
    load_seconds = t_load1 - t_load0
    print(f"Model load time: {load_seconds:.3f} s\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for phrase in TTS_PHRASES:
        # Deterministic, safe output basename: <id>__<stem>.wav
        filename = f"{phrase['id']}__{safe_stem(phrase['text'])}.wav"
        out = out_dir / filename
        print(f"[{phrase['id']}] {phrase['text']!r}  -> {out.name}")
        t0 = time.perf_counter()
        wavs, sr = generate_one(model, phrase["text"], args.language, args.speaker)
        t1 = time.perf_counter()
        synthesis_seconds = t1 - t0
        wav = wavs[0]
        audio_duration = _duration(wav, sr)
        rtf = realtime_factor(synthesis_seconds, audio_duration)
        _save_wav(out, wav, sr)
        rows.append(dict(
            id=phrase["id"], kind=phrase["kind"], category=phrase["category"],
            synthesis=synthesis_seconds, duration=audio_duration, rtf=rtf,
            path=str(out),
        ))
        rtf_s = format_rtf(rtf)
        print(f"    synth={synthesis_seconds:.3f}s dur={audio_duration:.3f}s "
              f"rtf={rtf_s}")

    print("---- summary ----")
    print(f"device         : {env.selected_device}")
    print(f"model          : {args.model}")
    print(f"speaker        : {args.speaker}")
    print(f"language       : {args.language}")
    print(f"load_time      : {load_seconds:.3f} s")
    print(f"phrases        : {len(rows)}")
    print(f"process_ram_mb : {get_process_ram_mb()}")
    print(f"gpu_vram_mb    : {get_gpu_vram_mb()} (0 = none/not measured)")
    print(f"output_dir     : {out_dir}")
    _print_ttfa_note()


def language_header(args, text: str) -> str:
    return f"{args.language}/{args.speaker}"


def _duration(wav, sr: int) -> float:
    n = getattr(wav, "shape", None)
    n = n[0] if n is not None and len(n) > 0 else len(wav)
    if not sr or sr <= 0:
        return 0.0
    return n / sr


def _save_wav(path: Path, wav, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), wav, sr)


def _print_ttfa_note() -> None:
    print(
        "\nTTFA: NOT measured. The official high-level generate_custom_voice() "
        "API is non-streaming (returns a complete wavs,sr tuple), so a true "
        "time-to-first-audio cannot be obtained at this level. Streaming/TTFA "
        "would require the lower-level Dual-Track streaming path, out of scope "
        "for this isolated PoC."
    )


def main() -> None:
    args = _build_parser().parse_args()

    if args.list_speakers:
        print("Built-in speakers for " + DEFAULT_MODEL_ID + ":")
        for s in SUPPORTED_SPEAKERS:
            print("  -", s)
        return

    if args.speaker not in SUPPORTED_SPEAKERS:
        raise SystemExit(
            f"Unknown speaker {args.speaker!r}. Supported: {SUPPORTED_SPEAKERS}"
        )

    env = detect_environment(args.device)
    print(f"=== Qwen3-TTS PoC environment ===")
    print(f"python     : {env.python_version}")
    print(f"platform   : {env.platform}")
    print(f"torch      : {env.torch_version}")
    print(f"cuda       : {env.torch_cuda_available} "
          f"(v{env.torch_cuda_version or 'n/a'})")
    print(f"gpu        : {env.gpu_name or 'none detected'}")
    print(f"selected   : {env.selected_device}")

    if not args.text and not args.id and not args.dataset:
        raise SystemExit(
            "Nothing to synthesize. Provide --text/--id for single mode, or "
            "--dataset to run the phrase set."
        )

    if args.dataset:
        run_dataset(env, args)
    else:
        if not args.text and not args.id:
            raise SystemExit("--dataset requires --text or --id.")
        args.text = phrase_text(args.id) if args.id else args.text
        run_single(env, args)


if __name__ == "__main__":
    main()
