# -*- coding: utf-8 -*-
"""Manual Windows-microphone -> LiveKit E2E helper (Phase D1.4).

This is a *human-run* helper. It is NOT launched automatically by tests. It:

  - connects to a local LiveKit room,
  - publishes the default Windows microphone as the local participant's audio
    track,
  - stays connected until Ctrl+C,
  - shuts down cleanly (unpublishes / disconnects).

Usage (run in a shell, then speak the phrase):
    $env:LIVEKIT_URL='ws://127.0.0.1:7880'
    $env:LIVEKIT_API_KEY='devkey'
    $env:LIVEKIT_API_SECRET='secret'
    $env:ROOM_NAME='voice-mic-e2e'
    .\.venv\Scripts\python.exe scripts\livekit_mic_e2e.py

Alternatively provide a pre-generated token with $env:LIVEKIT_TOKEN and the
script will not need the API key/secret.

Speak: "Как выполнить калибровку катетера?"
Press Ctrl+C to stop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

from livekit import rtc
from livekit.rtc import LocalAudioTrack, TrackKind
from livekit.rtc.media_devices import OutputPlayer

logger = logging.getLogger("mic-e2e")


def _env(name: str, default=""):
    return os.getenv(name, "").strip() or default


def make_token(room: str) -> str:
    """Build a join token for the local server from env credentials."""
    token = _env("LIVEKIT_TOKEN")
    if token:
        return token
    from livekit.api import AccessToken, VideoGrants

    api_key = _env("LIVEKIT_API_KEY")
    api_secret = _env("LIVEKIT_API_SECRET")
    if not (api_key and api_secret):
        raise SystemExit(
            "Set LIVEKIT_TOKEN or (LIVEKIT_API_KEY + LIVEKIT_API_SECRET)."
        )
    at = AccessToken(api_key, api_secret)  # type: ignore[arg-type]
    at.with_grants(
        VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_publish_data=True,
            can_subscribe=True,
        )
    ).with_identity("mic-e2e-human").with_name("MicE2E")
    return at.to_jwt()


async def main() -> None:
    url = _env("LIVEKIT_URL", "ws://127.0.0.1:7880")
    room_name = _env("ROOM_NAME", "voice-mic-e2e")
    token = make_token(room_name)

    print(f"[mic-e2e] connecting to {url} room={room_name!r} ...")
    room = rtc.Room()
    await room.connect(url, token)
    print(
        f"[mic-e2e] connected as participant {room.local_participant.identity!r}"
    )

    # ----------------------------------------------------------------------
    # Playback: route remote audio tracks (e.g. the worker's tts-answer) to the
    # default Windows output device. OutputPlayer wraps sounddevice internally,
    # uses output_device=None (=> default device => realme Buds Air 5) and mixes
    # all added remote tracks together.
    # ----------------------------------------------------------------------
    player = OutputPlayer()
    await player.start()
    print("[mic-e2e] output playback started on the default Windows audio device")

    # Track bookkeeping so each remote track is only handled/played once.
    _play_tasks: dict[str, asyncio.Task] = {}

    def _on_track_subscribed(track, publication, participant) -> None:
        """Synchronous Room.on("track_subscribed"): play remote AUDIO tracks."""
        if getattr(track, "kind", None) != TrackKind.KIND_AUDIO:
            return
        if track.sid in _play_tasks:
            return
        _play_tasks[track.sid] = asyncio.create_task(
            _play_remote_audio(track)
        )

    def _on_track_unsubscribed(track, publication, participant) -> None:
        task = _play_tasks.pop(track.sid, None)
        if task is not None:
            task.cancel()

    async def _play_remote_audio(track) -> None:
        """Add the remote audio track to the OutputPlayer for playback."""
        try:
            await player.add_track(track)
            logger.info(
                "playing remote audio track %s on output device", track.sid
            )
            # OutputPlayer.add_track creates its own AudioStream routed into the
            # internal mixer. Keep the task alive until unsubscribe/shutdown.
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "playback for remote track %s failed: %r", track.sid, exc
            )
        finally:
            _play_tasks.pop(track.sid, None)

    # Subscribe to remote tracks. LiveKit auto-subscribes all remote tracks, so
    # this fires for tracks published either before or after we connected. Our
    # own published mic comes back as a LOCAL track subscription, not a remote
    # "track_subscribed", so we never play our own microphone back.
    room.on("track_subscribed", _on_track_subscribed)
    room.on("track_unsubscribed", _on_track_unsubscribed)

    # Media device capture -> rtc.AudioSource -> LocalAudioTrack -> publish.
    media_devices = rtc.media_devices.MediaDevices()
    default_input = media_devices.default_input_device()
    devices = media_devices.list_input_devices()
    if default_input is None and not devices:
        raise SystemExit(
            "No audio input device found. Cannot capture the Windows microphone."
        )
    print(
        f"[mic-e2e] default input device index={default_input} "
        f"({len(devices)} input device(s) enumerated)"
    )

    capture = media_devices.open_input()  # default device, mono 48k
    track = LocalAudioTrack.create_audio_track("microphone", capture.source)
    publication = await room.local_participant.publish_track(
        track, options=rtc.TrackPublishOptions()
    )
    print(
        f"[mic-e2e] published track sid={publication.sid} "
        f"kind={publication.kind} source={publication.source}"
    )

    print("\n[mic-e2e] mic publishing. LIVEKIT voice agent can now subscribe.\n")
    print("Speak: Как выполнить калибровку катетера?")
    print("(Press Ctrl+C to stop)\n")

    stop = asyncio.Event()

    def _on_sigint(*_args):
        stop.set()

    # On Windows, add_signal_handler(SIGINT) requires a Proactor loop; guard it.
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except (NotImplementedError, RuntimeError):
        # Fallback: rely on KeyboardInterrupt bubbled up to main().
        pass

    try:
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("[mic-e2e] shutting down ...")
        # Cancel any pending playback tasks / stop the output player.
        for task in list(_play_tasks.values()):
            task.cancel()
        try:
            await player.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("output player close failed: %s", exc)
        try:
            await room.local_participant.unpublish_track(publication.sid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("unpublish failed: %s", exc)
        try:
            await capture.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("capture close failed: %s", exc)
        await room.disconnect()
        print("[mic-e2e] done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

