"""Runnable example: a LiveKit voice agent using Hakim for STT + TTS.

This wires `HakimSTT` / `HakimTTS` (talking directly to Hakim's public
WebSocket API — see `../hakim_livekit/`) into a standard LiveKit
`AgentSession`. Swap in your own LLM plugin; VAD/turn-detection come from
`livekit-plugins-silero`, which is unrelated to Hakim.

Setup:
    cd integrations/livekit
    pip install -r requirements.txt livekit-plugins-silero livekit-plugins-openai
    cp .env.example .env   # fill in HAKIM_API_KEY, LIVEKIT_*, HAKIM_VOICE
    python examples/voice_agent.py dev

See ../README.md for the full walkthrough (auth, regions, troubleshooting).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero

from hakim_livekit import HakimSTT, HakimTTS

load_dotenv()

logger = logging.getLogger("hakim-livekit-example")


def prewarm(proc: JobProcess) -> None:
    # Load the VAD model once per worker process, not per session.
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=HakimSTT(
            language="ar",
            region=os.environ.get("HAKIMREGION", "auto"),
        ),
        # Swap for any LLM plugin — Hakim doesn't provide one.
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=HakimTTS(
            voice=os.environ["HAKIM_VOICE"],
            model="hakim-fast-v1",
            region=os.environ.get("HAKIMREGION", "auto"),
        ),
    )

    await session.start(
        agent=Agent(
            instructions=(
                "You are a helpful Arabic-speaking voice assistant. "
                "Keep responses short and conversational."
            )
        ),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(instructions="Greet the caller in Arabic and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
