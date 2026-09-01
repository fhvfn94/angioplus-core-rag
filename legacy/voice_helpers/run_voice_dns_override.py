import socket
import runpy

TTS_HOST = "eva-possibly-hdtv-discover.trycloudflare.com"
TTS_IP = "104.16.230.132"

_original_getaddrinfo = socket.getaddrinfo

def _getaddrinfo(host, port, *args, **kwargs):
    if host == TTS_HOST:
        host = TTS_IP
    return _original_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = _getaddrinfo

import asyncio

_original_loop_getaddrinfo = asyncio.BaseEventLoop.getaddrinfo

async def _loop_getaddrinfo(self, host, port, *args, **kwargs):
    if host == TTS_HOST:
        host = TTS_IP
    return await _original_loop_getaddrinfo(self, host, port, *args, **kwargs)

asyncio.BaseEventLoop.getaddrinfo = _loop_getaddrinfo

print(f"DNS override active: {TTS_HOST} -> {TTS_IP}")

runpy.run_module("voice.agent", run_name="__main__")
