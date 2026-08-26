# -*- coding: utf-8 -*-
"""Regression: Silero plugin must NOT be registered from a background thread.

Phase D1.4 moved `from livekit.plugins import silero` to module scope in
voice/vad.py so `Plugin.register_plugin()` runs on the main thread during
normal import/startup. Previously `load_vad()` imported silero lazily inside
the function; when called from an audio-processing task (not the main thread)
this raised:

    RuntimeError: Plugins must be registered on the main thread

These tests verify that importing and using voice.vad from a worker thread does
not attempt plugin registration there.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from voice.vad import VADConfig, load_vad


def test_silero_plugin_registered_on_main_thread_at_import():
    """The plugin must already be registered (done at module import)."""
    from livekit.agents.plugin import Plugin

    registered = [
        p
        for p in Plugin.registered_plugins
        if "silero" in getattr(p, "_package", "")
        or "Silero" in type(p).__name__
    ]
    assert registered, "silero plugin must be registered from the main thread"


def test_load_vad_from_background_thread_does_not_register_plugin():
    """Calling load_vad() on a worker thread must not hit the main-thread check.

    Uses an injected fake loader so no ONNX model is downloaded, but the call
    runs on a non-main thread, exactly as the real audio-processing task does.
    """
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kind="vad")

    def run_in_background():
        # If voice.vad lazily imported/registered silero here, the plugin's
        # main-thread guard would raise RuntimeError. It must not.
        result = load_vad(VADConfig(), vad_loader=fake_loader)
        return result.kind

    with ThreadPoolExecutor(max_workers=1) as ex:
        background_result = ex.submit(run_in_background).result()

    assert background_result == "vad"
    assert captured["min_silence_duration"] == 0.55


def test_module_scope_silero_is_reused_in_background():
    """The already-imported module-scope silero is used (no re-import attempt)."""
    import threading

    import voice.vad

    # Confirm silero is bound at module scope (so no lazy import remains).
    assert hasattr(voice.vad, "silero")

    # Simulate the audio-processing path (worker thread) reusing load_vad's
    # default loader symbol without triggering registration.
    in_thread = {}

    def worker():
        in_thread["silero_name"] = voice.vad.silero.__name__
        in_thread["is_main"] = threading.current_thread() is threading.main_thread()

    with ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(worker).result()

    assert in_thread["silero_name"] == "livekit.plugins.silero"
    assert in_thread["is_main"] is False  # really ran on a worker thread
