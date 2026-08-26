# -*- coding: utf-8 -*-
"""Smallest real LiveKit RTC text participant for the Phase C end-to-end test.

Chain under test (this script does NOT call /ask directly):

    Local LiveKit Server (ws://127.0.0.1:7880, devkey/secret)
      -> Agent worker (voice/agent.py, already registered)
      -> LiveKit Room "e2e-room"
      -> this participant publishes a text data message
      -> voice/agent.py receives it -> RagClient -> /ask
      -> /ask answer -> voice/agent.py publishes it back into the room
      -> this participant prints QUESTION / ANSWER

Credentials:
    LIVEKIT_URL        e.g. ws://127.0.0.1:7880
    LIVEKIT_API_KEY    devkey
    LIVEKIT_API_SECRET secret

The current livekit.rtc SDK requires an access token (not raw API
credentials), so we mint one with the installed livekit.api.AccessToken.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import sys

from livekit import rtc
from livekit.api import AccessToken, VideoGrants

ROOM = "e2e-room"
IDENTITY = "tester"
QUESTION = "Как выполнить калибровку катетера?"
# Generous timeout: real RAG round-trip can take several seconds.
REPLY_TIMEOUT_SECONDS = 120.0
# How long to wait for the agent participant to join the room before the
# question is published (deterministic readiness, not a fixed sleep).
AGENT_READY_TIMEOUT_SECONDS = 30.0


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _configure_utf8_console() -> None:
    """Force UTF-8 stdout/stderr so Cyrillic + emoji print correctly on
    Windows (the default cp1251 console cannot encode those characters)."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # not a text stream in this environment
    # Also set the environment/lang so any subprocess honours UTF-8.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


async def _wait_for_agent_ready(room: rtc.Room) -> None:
    """Wait until the agent's participant is actually present in the room.

    Dispatch of the agent job happens when the room is created/joined, so we
    must not publish until the agent has connected and registered its
    data_received handler. We detect that deterministically via the Room's
    participant state/events (no fixed sleep).
    """
    connected = asyncio.Event()

    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        print(f"[e2e] participant connected: {participant.identity}")
        connected.set()

    room.on("participant_connected", on_participant_connected)

    deadline = asyncio.get_event_loop().time() + AGENT_READY_TIMEOUT_SECONDS
    # Satisfy readiness immediately if the agent already joined.
    while not connected.is_set():
        if room.remote_participants:  # any remote participant == the agent here
            print(
                "[e2e] agent present in room: "
                f"{list(room.remote_participants)[:1]}"
            )
            connected.set()
            break
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise SystemExit(
                "[e2e] FAIL: agent did not join the room within "
                f"{AGENT_READY_TIMEOUT_SECONDS}s; aborting before publish"
            )
        await asyncio.sleep(0.25)


async def main() -> int:
    _configure_utf8_console()

    url = _env("LIVEKIT_URL")
    api_key = _env("LIVEKIT_API_KEY")
    api_secret = _env("LIVEKIT_API_SECRET")

    # 1) Mint a join token for this participant (devkey/secret pair).
    token = (
        AccessToken(api_key=api_key, api_secret=api_secret)
        .with_identity(IDENTITY)
        .with_ttl(datetime.timedelta(minutes=5))
        .with_grants(VideoGrants(room_join=True, room=ROOM))
        .to_jwt()
    )

    room = rtc.Room()
    # The reply is awaited; set when the agent's data message arrives.
    reply_holder: dict[str, bytes | None] = {"reply": None}

    def on_data_received(packet: rtc.DataPacket) -> None:
        reply_holder["reply"] = bytes(packet.data)

    def on_disconnected() -> None:
        reply_holder["reply"] = reply_holder["reply"]  # no-op; room gone

    room.on("data_received", on_data_received)
    room.on("disconnected", on_disconnected)

    await room.connect(url, token)
    print(f"[e2e] connected to {url} room={ROOM} identity={IDENTITY}")

    # 2) Deterministic readiness: wait until the agent participant is in the
    #    room before publishing the question.
    await _wait_for_agent_ready(room)

    # 3) Publish the text question as a data message to the room.
    #    NOTE: publish_data is a coroutine in the current livekit.rtc SDK.
    await room.local_participant.publish_data(
        QUESTION.encode("utf-8"), reliable=True, topic="phase-c-text"
    )
    print(f"[e2e] published question ({len(QUESTION.encode('utf-8'))} bytes)")

    # 4) Wait for the agent's reply (goes through LiveKit -> voice/agent.py
    #    -> RagClient -> /ask -> back into the room).
    deadline = asyncio.get_event_loop().time() + REPLY_TIMEOUT_SECONDS
    while reply_holder["reply"] is None:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        await asyncio.sleep(0.25)

    await room.disconnect()

    if reply_holder["reply"] is None:
        print("[e2e] FAIL: no reply received within timeout")
        return 1

    answer = reply_holder["reply"].decode("utf-8", errors="replace")
    print("QUESTION:")
    print(QUESTION)
    print("ANSWER:")
    print(answer)
    print("[e2e] PASS: reply received via LiveKit Room -> voice/agent.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
