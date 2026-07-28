"""Hakim TTS plugin for LiveKit Agents.

Speaks the public `WSS /v1/audio/speech/stream` API directly (see
https://tryhakim.ai/docs, operation `audio_speech_stream_ws`) using the
plain `websockets` library, and the public `POST /v1/audio/speech` API for
one-shot synthesis (`aiohttp`).

Wire contract implemented here (streaming):
  - Client -> server: `session.update`, `speech.create`, `session.close`.
  - Server -> client: `session.created`, `speech.started`, binary
    PCM-S16LE @ 24 kHz mono chunks, `speech.done`, `session.usage`, `error`.
  - `response_format` / `sample_rate` are locked to `pcm` / `24000` on the
    realtime surface — the plugin's `sample_rate` is fixed to match.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import aiohttp
import websockets
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._common import HakimStreamError, Region, auth_headers, resolve_api_key, resolve_http_url, resolve_ws_url

# The realtime WS surface pins response_format=pcm / sample_rate=24000
# (engine-native rate). Keep the plugin's declared sample_rate in lockstep
# so `.synthesize()` (HTTP) and `.stream()` (WS) always agree with what the
# AgentSession pipeline was configured for.
SAMPLE_RATE = 24000
DEFAULT_MODEL = "hakim-fast-v1"


@dataclass
class _TTSOptions:
    model: str
    voice: str
    cfg: float
    voice_prompt: str | None
    region: Region
    base_url: str | None


class HakimTTS(tts.TTS):
    """LiveKit Agents TTS plugin backed by Hakim. Use `.stream()` for
    sentence-by-sentence LLM pipelining (the common `AgentSession` case) or
    `.synthesize()` for one-shot text-to-speech.
    """

    def __init__(
        self,
        *,
        voice: str,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: str = DEFAULT_MODEL,
        cfg: float = 3.0,
        voice_prompt: str | None = None,
        region: Region = "auto",
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            voice: Hakim voice id or slug (e.g. `"cmokbc2b1001pvu39wmj61b7h"`).
                Required — there is no bundled default voice.
            api_key: Hakim API key (`hk_live_...`). Falls back to the
                `HAKIM_API_KEY` env var.
            model: `"hakim-fast-v1"` (default, sub-120ms TTFB), `"hakim-v2"`
                (higher quality + non-verbal tags), or `"hakim-v3"`
                (adds free-form `voice_prompt` control).
            cfg: Classifier-free-guidance weight, `0.0`-`10.0` (default `3.0`).
            voice_prompt: Free-form voice-character description, only
                honoured on `model="hakim-v3"` — dropped elsewhere.
            region: `"auto" | "de" | "uae" | "ksa"`.
            base_url: Overrides the region table entirely (staging /
                self-hosted). Takes precedence over `region`.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._api_key = resolve_api_key(api_key if is_given(api_key) else None)
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            cfg=cfg,
            voice_prompt=voice_prompt,
            region=region,
            base_url=base_url if is_given(base_url) else None,
        )
        self._session = http_session

    @property
    def provider(self) -> str:
        return "Hakim"

    @property
    def model(self) -> str:
        return self._opts.model

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "HakimChunkedStream":
        return HakimChunkedStream(
            tts=self,
            input_text=text,
            opts=self._opts,
            api_key=self._api_key,
            conn_options=conn_options,
            http_session=self._ensure_session(),
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "HakimSynthesizeStream":
        return HakimSynthesizeStream(tts=self, opts=self._opts, api_key=self._api_key, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()


class HakimChunkedStream(tts.ChunkedStream):
    """One-shot synthesis via `POST /v1/audio/speech` (`response_format:
    "pcm"`). Used by `HakimTTS.synthesize()` — LiveKit falls back to this
    when a plain one-shot call is needed instead of the persistent socket.
    """

    def __init__(
        self,
        *,
        tts: HakimTTS,
        input_text: str,
        opts: _TTSOptions,
        api_key: str,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._conn_options = conn_options

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )

        payload: dict[str, object] = {
            "model": self._opts.model,
            "input": self.input_text,
            "voice": self._opts.voice,
            "response_format": "pcm",
            "sample_rate": SAMPLE_RATE,
            "cfg": self._opts.cfg,
        }
        if self._opts.voice_prompt:
            payload["voice_prompt"] = self._opts.voice_prompt

        url = resolve_http_url("/v1/audio/speech", region=self._opts.region, base_url=self._opts.base_url)
        try:
            async with self._session.post(
                url,
                json=payload,
                headers=auth_headers(self._api_key),
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise APIStatusError(
                        f"Hakim speech request failed: {body}",
                        status_code=resp.status,
                        request_id=resp.headers.get("X-Request-Id"),
                    )
                async for chunk in resp.content.iter_chunked(4096):
                    output_emitter.push(chunk)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(str(e)) from e

        output_emitter.flush()


class HakimSynthesizeStream(tts.SynthesizeStream):
    """Persistent-socket streaming synthesis via `WSS
    /v1/audio/speech/stream`. Each flushed input segment (LiveKit flushes
    per sentence during LLM token pipelining) becomes one `speech.create`
    request over the same connection — no per-utterance TCP/TLS handshake.
    """

    def __init__(
        self,
        *,
        tts: HakimTTS,
        opts: _TTSOptions,
        api_key: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._api_key = api_key
        # Hakim's `speech.create` is one discrete request per utterance
        # (unlike providers that stream text into one long-lived synthesis
        # context) — each flushed input segment maps to exactly one
        # speech.started/speech.done pair. `end_input()` on the emitter must
        # fire exactly once, after both "no more text is coming" AND "every
        # utterance we already sent has finished" are true — whichever of
        # the two tasks observes that last is responsible for calling it.
        self._pending_utterances = 0
        self._input_closed = False
        self._ended = False
        # `_input_task` and `_recv_task` start concurrently — nothing
        # otherwise stops `_input_task` from firing `speech.create` before
        # `_recv_task` has observed `session.created`. The proxy's upstream
        # GPU connection comes up asynchronously after the WS upgrade (see
        # tts-realtime-proxy.ts's handshake note); a `speech.create` that
        # wins that race gets accepted and billed but produces no audio
        # (`speech.done` arrives with `duration_ms: 0`, no `speech.started`,
        # no binary frames at all — confirmed via a live capture). Gating
        # the first send on this event closes the race.
        self._session_ready = asyncio.Event()
        # Set once `_input_closed` AND every sent utterance's `speech.done`
        # has actually arrived (mirrors `_maybe_end_input`'s own condition).
        # `_input_task` must wait on this before sending `session.close` —
        # for a short single-utterance reply, the input channel closes
        # almost immediately after the one `speech.create` goes out, so
        # without this wait `session.close` reaches the server while the
        # utterance is still mid-synthesis. The server treats that as an
        # abort: it bills the request but the socket closes before
        # `speech.started`/audio/`speech.done` ever go out — this was the
        # actual, 100%-reproducible cause of "no audio frames were pushed",
        # confirmed via a live capture (this bug, not the session_ready race
        # above, was the primary one — that race is real but secondary).
        self._all_utterances_done = asyncio.Event()

    def _maybe_end_input(self, output_emitter: tts.AudioEmitter) -> None:
        if self._input_closed and self._pending_utterances <= 0:
            if not self._ended:
                self._ended = True
                output_emitter.end_input()
            self._all_utterances_done.set()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        url = resolve_ws_url("/v1/audio/speech/stream", region=self._opts.region, base_url=self._opts.base_url)
        try:
            async with websockets.connect(url, additional_headers=auth_headers(self._api_key)) as ws:
                session_update: dict[str, object] = {
                    "model": self._opts.model,
                    "voice": self._opts.voice,
                    "cfg": self._opts.cfg,
                }
                if self._opts.voice_prompt:
                    session_update["voice_prompt"] = self._opts.voice_prompt
                await ws.send(json.dumps({"type": "session.update", "session": session_update}))

                input_task = asyncio.create_task(self._input_task(ws, output_emitter), name="hakim-tts-input")
                recv_task = asyncio.create_task(self._recv_task(ws, output_emitter), name="hakim-tts-recv")
                try:
                    await asyncio.gather(input_task, recv_task)
                finally:
                    await utils.aio.gracefully_cancel(input_task, recv_task)
        except websockets.exceptions.InvalidStatus as e:
            raise APIStatusError(f"Hakim TTS upgrade rejected: {e}") from e
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            raise APIConnectionError(str(e)) from e

    async def _input_task(self, ws: websockets.WebSocketClientProtocol, output_emitter: tts.AudioEmitter) -> None:
        # Wait for the session handshake to complete before sending the
        # first utterance — see the note on `_session_ready` in __init__.
        await self._session_ready.wait()

        pending: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                text = "".join(pending).strip()
                pending = []
                if text:
                    self._pending_utterances += 1
                    await ws.send(
                        json.dumps(
                            {
                                "type": "speech.create",
                                "input": text,
                                "request_id": utils.shortuuid(),
                            }
                        )
                    )
                continue
            pending.append(data)
        self._input_closed = True
        self._maybe_end_input(output_emitter)
        # Don't tell the server to close until every utterance we sent has
        # actually finished (speech.done received) — see the note on
        # `_all_utterances_done` in __init__.
        await self._all_utterances_done.wait()
        await ws.send(json.dumps({"type": "session.close"}))

    async def _recv_task(self, ws: websockets.WebSocketClientProtocol, output_emitter: tts.AudioEmitter) -> None:
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                output_emitter.push(raw)
                continue

            event = json.loads(raw)
            etype = event.get("type")

            if etype == "session.created":
                self._session_ready.set()
                continue
            elif etype in ("session.usage", None):
                continue
            elif etype == "speech.started":
                output_emitter.start_segment(segment_id=event.get("request_id") or utils.shortuuid())
            elif etype == "speech.done":
                output_emitter.end_segment()
                self._pending_utterances -= 1
                self._maybe_end_input(output_emitter)
                if self._ended:
                    return
            elif etype == "error":
                code = event.get("code", "unknown_error")
                message = event.get("message", "")
                retryable = bool(event.get("retryable", False))
                fatal = bool(event.get("fatal", False))
                if retryable and not fatal:
                    raise APIConnectionError(f"[{code}] {message}")
                raise HakimStreamError(code, message, retryable=retryable)
