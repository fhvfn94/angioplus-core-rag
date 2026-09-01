# -*- coding: utf-8 -*-
"""Minimal HTTP TTS server for Qwen3-TTS 0.6B on a GPU (Phase D2.3).

This is the *remote* side that runs on a rented
NVIDIA GPU (e.g. a Vast.ai RTX 4090). It loads the official Qwen3-TTS
checkpoint ONCE at startup and serves a single non-streaming endpoint:

    POST /tts
      body: {"text": "...", "speaker": "Vivian", "language": "Russian"}
      response: audio/wav (mono 16-bit PCM) ready to be published into LiveKit

The local LiveKit voice worker calls this service via ``voice.tts_client``.
This file MUST run in the ISOLATED qwen venv (not the shared project venv)
because ``qwen-tts==0.1.1`` pins transformers==4.57.3, conflicting with the
project's pinned transformers==4.57.6.

Default settings (override via env):
    TTS_MODEL_ID   Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    TTS_PORT       8001
    TTS_HOST       0.0.0.0
    TTS_DEVICE     auto (cuda if available else cpu)
    TTS_DTYPE      auto (bfloat16 on cuda, float32 on cpu)

On CUDA we use ``torch.bfloat16`` + ``attn_implementation="sdpa"``. We do NOT
require FlashAttention (works with eager/SDPA on consumer GPUs without the
flash-attn build headache).

Usage on Vast.ai (inside the isolated venv):
    python scripts/tts/tts_server.py
    # or with uvicorn reload/workers
    uvicorn scripts.tts.tts_server:app --host 0.0.0.0 --port 8001

Minimal health check:
    curl http://<container-ip>:8001/tts \
      -H 'Content-Type: application/json' \
      -d '{"text":"Как выполнить калибровку катетера?","speaker":"Vivian","language":"Russian"}'
"""
from __future__ import annotations

import io
import logging
import os
import wave

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001

# The 9 built-in speakers of the CustomVoice checkpoint.
SUPPORTED_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="text to synthesize")
    speaker: str = Field("Vivian", description="built-in CustomVoice speaker")
    language: str = Field("Russian", description="language as the model expects")


# ---------------------------------------------------------------------------
# Model lifecycle (loaded once, not per request)
# ---------------------------------------------------------------------------

class ModelHolder:
    """Holds the loaded Qwen3TTSModel + its device/dtype so we only load once."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype_name: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype_name, self.device)
        self.model = None
        self.sample_rate: int | None = None  # cached from first generation

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device.lower()
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    @staticmethod
    def _resolve_dtype(dtype_name: str | None, device: str) -> str:
        if dtype_name:
            return dtype_name.lower()
        if device.startswith("cuda"):
            return "bfloat16"
        return "float32"

    def load(self) -> None:
        """Import torch/qwen_tts lazily and load the checkpoint once."""
        # Honor TTS_MODEL_ID env override (default is the official 0.6B model).
        self.model_id = os.getenv("TTS_MODEL_ID", "").strip() or self.model_id
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen_tts / torch not installed. Run this inside the isolated "
                "qwen venv created from requirements-tts-server.txt."
            ) from exc

        kwargs: dict = {"device_map": self.device}
        if self.device.startswith("cuda"):
            kwargs["dtype"] = torch.bfloat16
            # SDPA is reliable without the flash-attn build requirement.
            kwargs["attn_implementation"] = "sdpa"
        else:
            kwargs["dtype"] = torch.float32

        logger.info(
            "loading Qwen3-TTS model %s device=%s dtype=%s attn=sdpa(on cuda)",
            self.model_id, self.device, self.dtype,
        )
        self.model = Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)
        logger.info("Qwen3-TTS model loaded")

    def generate_wav(self, request: TTSRequest) -> bytes:
        """Run non-streaming generation and return a mono 16-bit PCM WAV blob."""
        if self.model is None:
            self.load()
        wavs, sr = self.model.generate_custom_voice(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            instruct="",
        )
        return self._encode_wav(wavs[0], sr)

    @staticmethod
    def _encode_wav(utterance, sr: int) -> bytes:
        """Convert the model's tensor utterance into mono int16 PCM WAV bytes.

        The model returns a (n_samples, 1) float tensor in roughly [-1, 1].
        We clamp, scale to int16 and write a WAV with stdlib ``wave`` to avoid
        a hard soundfile/scipy dependency in the server path.
        """
        import torch

        utt = utterance.detach().cpu()
        # flatten to a 1-D float tensor regardless of (N,1) / (1,N) shape.
        flat = utt.contiguous().view(-1)
        if flat.numel() == 0:
            raise ValueError("model returned empty audio")
        int16 = (torch.clamp(flat, -1.0, 1.0) * 32767.0).to(torch.int16)
        pcm = int16.numpy().tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(pcm)
        return buf.getvalue()


holder = ModelHolder()


def _get_holder() -> ModelHolder:
    if holder.model is None:
        holder.load()
    return holder


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AngioPlus Qwen3-TTS",
    version="0.1.0",
    description="Non-streaming Qwen3-TTS 0.6B HTTP service (Phase D2.3)",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": holder.model is not None,
        "device": holder.device,
        "dtype": holder.dtype,
    }


@app.post("/tts")
def tts(request: TTSRequest) -> Response:
    """Synthesize ``request.text`` to speech and return a mono WAV blob."""
    if request.speaker not in SUPPORTED_SPEAKERS:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"unknown speaker {request.speaker!r}; "
                   f"supported: {SUPPORTED_SPEAKERS}",
        )
    h = _get_holder()
    try:
        wav = h.generate_wav(request)
    except Exception as exc:  # noqa: BLE001 - surface a clean HTTP 500
        logger.exception("TTS generation failed")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}")
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts.wav"'},
    )


def main() -> None:
    import uvicorn

    host = os.getenv("TTS_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port = int(os.getenv("TTS_PORT", str(DEFAULT_PORT)).strip() or DEFAULT_PORT)
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "starting Qwen3-TTS server on %s:%s (device=%s dtype=%s)",
        host, port, holder.device, holder.dtype,
    )
    # Load the model before uvicorn serves requests so the first /tts call is
    # already warm (one-time load at startup, not per request).
    _get_holder()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
