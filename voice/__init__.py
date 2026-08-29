# -*- coding: utf-8 -*-
"""LiveKit voice component boundary for AngioPlus Core RAG.

This package contains ONLY the thin client boundary between the future
LiveKit voice component and the existing RAG API. It must never import
app.main / app.* RAG implementation and must never access Qdrant directly.

Phase D2.3 adds two remote-Qwen3-TTS boundaries:
  - ``voice/tts_client.py``  - async HTTP client to the remote GPU TTS service,
  - ``voice/tts_publisher.py`` - publishes the returned PCM into the LiveKit
    room as a spoken track.
Both keep heavy GPU deps (torch / qwen_tts / transformers) OUT of this package:
they run only in the isolated scripts/tts_server.py on the remote GPU.
"""
