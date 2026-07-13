"""Standalone connectivity smoke test — no LiveKit dependency.

Verifies your `HAKIM_API_KEY` and network path can reach the realtime TTS
endpoint before you wire anything into LiveKit. Useful as the first
troubleshooting step when the agent reports connection errors: run this
first to isolate "Hakim reachability" from "LiveKit plugin wiring".

Usage:
    HAKIM_API_KEY=hk_live_... HAKIM_VOICE=cmokbc2b1001pvu39wmj61b7h \
        python examples/smoke_test.py "مرحبا بالعالم"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import websockets


async def main() -> None:
    api_key = os.environ["HAKIM_API_KEY"]
    voice = os.environ.get("HAKIM_VOICE", "cmokbc2b1001pvu39wmj61b7h")
    text = sys.argv[1] if len(sys.argv) > 1 else "مرحبا بالعالم"
    region = os.environ.get("HAKIM_REGION", "auto")
    host = {
        "auto": "api.tryhakim.ai",
        "de": "de.api.tryhakim.ai",
        "uae": "uae.api.tryhakim.ai",
        "ksa": "ksa.api.tryhakim.ai",
    }.get(region, "api.tryhakim.ai")
    url = f"wss://{host}/v1/audio/speech/stream"

    total_bytes = 0
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"}) as ws:
        await ws.send(
            json.dumps({"type": "session.update", "session": {"model": "hakim-fast-v1", "voice": voice}})
        )
        await ws.send(json.dumps({"type": "speech.create", "input": text, "request_id": "smoke_test"}))

        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                total_bytes += len(raw)
                continue
            event = json.loads(raw)
            print(f"<- {event['type']}: {event}")
            if event["type"] == "speech.done":
                break
            if event["type"] == "error":
                raise SystemExit(f"Hakim returned an error: {event}")

        await ws.send(json.dumps({"type": "session.close"}))

    print(f"\nOK — received {total_bytes} bytes of PCM-S16LE @ 24kHz audio ({total_bytes / 2 / 24000:.2f}s).")


if __name__ == "__main__":
    asyncio.run(main())
