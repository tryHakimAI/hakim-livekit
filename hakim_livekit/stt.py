"""Hakim STT plugin for LiveKit Agents.

Speaks the public `WSS /v1/audio/transcriptions/stream` API directly (see
https://tryhakim.ai/docs, operation `audio_transcriptions_stream`) using
the plain `websockets` library.

Wire contract implemented here:
  - Client -> server: `session.update`, `input_audio_buffer.append`,
    `input_audio_buffer.commit`, `session.close`.
  - Server -> client: `session.created`, `transcription.delta`
    (`is_final: false`), `transcription.done` (final text for the commit
    window), `error`.
  - Model is pinned to `hakim-arab-v2` — the only accepted value today.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, replace

import aiohttp
import websockets
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._common import HakimStreamError, Region, auth_headers, resolve_api_key, resolve_http_url, resolve_ws_url

STT_MODEL = "hakim-arab-v2"
# Recycle a pooled WS connection after this long regardless of use — mirrors
# the same constant in tts.py / livekit-agents' own inference TTS default.
_MAX_SESSION_DURATION = 300


@dataclass
class _STTOptions:
    language: str
    timestamps: str
    diarize: bool
    partials: bool
    input_audio_format: str
    input_sample_rate: int
    region: Region
    base_url: str | None


class HakimSTT(stt.STT):
    """LiveKit Agents STT plugin backed by Hakim's realtime transcription
    WebSocket. Use `HakimSTT().stream()` inside an `AgentSession` exactly
    like any other LiveKit STT plugin.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: str = "ar",
        timestamps: str = "segment",
        diarize: bool = False,
        partials: bool = True,
        input_audio_format: str = "pcm16",
        input_sample_rate: int = 16000,
        region: Region = "auto",
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            api_key: Hakim API key (`hk_live_...`). Falls back to the
                `HAKIM_API_KEY` env var.
            language: BCP-47-ish language hint (`"ar"`, `"en"`, ...). Pass
                per-call overrides via `stream(language=...)`.
            timestamps: `"word" | "segment" | "none"`.
            diarize: Speaker diarization (stereo call-recording use case;
                mono diarization is not yet supported).
            input_audio_format: `"pcm16" | "opus" | "mulaw"`. LiveKit
                delivers `rtc.AudioFrame`s as PCM16, so the default matches
                without any client-side transcoding.
            input_sample_rate: Must match the sample rate of the frames you
                push. LiveKit's default room audio is 48 kHz; resample to
                16 kHz upstream (or pass 48000 here) — either works, but
                keep the two in sync.
            region: `"auto" | "de" | "uae" | "ksa"` — pins the session to a
                specific regional endpoint. `"auto"` picks the closest
                healthy region.
            base_url: Overrides the region table entirely (staging /
                self-hosted). Takes precedence over `region`.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=True,
                aligned_transcript="word" if timestamps == "word" else False,
                offline_recognize=True,
            )
        )
        self._api_key = resolve_api_key(api_key if is_given(api_key) else None)
        self._opts = _STTOptions(
            language=language,
            timestamps=timestamps,
            diarize=diarize,
            partials=partials,
            input_audio_format=input_audio_format,
            input_sample_rate=input_sample_rate,
            region=region,
            base_url=base_url if is_given(base_url) else None,
        )
        self._session = http_session
        # Reused across turns so a fresh TCP/TLS/WS-upgrade handshake isn't
        # paid on every single utterance — see HakimSpeechStream._run().
        self._pool = utils.ConnectionPool[websockets.ClientConnection](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=_MAX_SESSION_DURATION,
        )

    @property
    def provider(self) -> str:
        return "Hakim"

    @property
    def model(self) -> str:
        return STT_MODEL

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _connect_ws(self, timeout: float) -> websockets.ClientConnection:
        url = resolve_ws_url(
            "/v1/audio/transcriptions/stream", region=self._opts.region, base_url=self._opts.base_url
        )
        return await asyncio.wait_for(
            websockets.connect(url, additional_headers=auth_headers(self._api_key)),
            timeout,
        )

    async def _close_ws(self, ws: websockets.ClientConnection) -> None:
        try:
            await ws.send(json.dumps({"type": "session.close"}))
        except websockets.exceptions.ConnectionClosed:
            pass
        await ws.close()

    def prewarm(self) -> None:
        """Open a pooled connection ahead of the first turn. LiveKit's
        `AgentSession` calls this automatically at session start."""
        self._pool.prewarm()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """One-shot batch transcription via `POST /v1/audio/transcriptions`.

        Most voice agents only ever use `.stream()`; this makes `HakimSTT`
        also work wherever LiveKit expects a plain `recognize()` call
        (`FallbackAdapter`, ad-hoc batch jobs, tests).
        """
        lang = language if is_given(language) else self._opts.language
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form = aiohttp.FormData()
        form.add_field("model", STT_MODEL)
        form.add_field("language", lang)
        form.add_field("timestamps", self._opts.timestamps)
        form.add_field("response_format", "json")
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")

        url = resolve_http_url(
            "/v1/audio/transcriptions", region=self._opts.region, base_url=self._opts.base_url
        )
        session = self._ensure_session()
        try:
            async with session.post(
                url,
                data=form,
                headers=auth_headers(self._api_key),
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise APIStatusError(
                        f"Hakim transcription request failed: {body}",
                        status_code=resp.status,
                        request_id=resp.headers.get("X-Request-Id"),
                    )
                data = await resp.json()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(str(e)) from e

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=data.get("request_id", ""),
            alternatives=[stt.SpeechData(language=lang, text=data.get("text", ""))],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "HakimSpeechStream":
        opts = replace(self._opts, language=language if is_given(language) else self._opts.language)
        return HakimSpeechStream(stt=self, opts=opts, api_key=self._api_key, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
        await self._pool.aclose()


class HakimSpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: HakimSTT,
        opts: _STTOptions,
        api_key: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.input_sample_rate)
        self._opts = opts
        self._api_key = api_key
        # `_send_task` and `_recv_task` start concurrently — see the matching
        # note on `HakimSynthesizeStream._session_ready` in tts.py. Gating
        # the first audio frame on `session.created` avoids feeding audio
        # into a session whose upstream connection isn't up yet.
        self._session_ready = asyncio.Event()
        # Hakim's commit/transcription.done is one discrete request per
        # commit window (mirrors HakimSynthesizeStream's per-utterance
        # speech.create/speech.done) — track how many commits are still
        # outstanding so `_recv_task` knows precisely when this stream's
        # work is done, instead of relying on the socket being closed.
        self._pending_commits = 0
        self._input_closed = False
        self._ended = False
        self._all_commits_done = asyncio.Event()

    def _maybe_finish(self) -> None:
        if self._input_closed and self._pending_commits <= 0:
            self._ended = True
            self._all_commits_done.set()

    async def _run(self) -> None:
        try:
            async with self._stt._pool.connection(timeout=self._conn_options.timeout) as ws:
                connection_reused = self._stt._pool.last_connection_reused
                self._report_connection_acquired(self._stt._pool.last_acquire_time, connection_reused)

                # Resent on every turn even when the underlying socket is
                # reused from the pool — cheap (one message on an
                # already-open connection) and keeps per-stream `language`
                # overrides (HakimSTT.stream(language=...)) correct, unlike
                # skipping it entirely on a reused connection. NOTE: the
                # server only ever emits `session.created` once, right
                # after the WS upgrade (see stt-realtime-proxy.ts's
                # emitSessionCreated — it has exactly one call site, at
                # initial connect; `session.update` never re-triggers it).
                # A reused connection has already seen that ack in a
                # previous turn, so don't wait for a second one that will
                # never arrive — that hung every reused turn forever,
                # silently dropping it (confirmed live: "need to speak
                # twice for the LLM to reply").
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "model": STT_MODEL,
                                "language": self._opts.language,
                                "timestamps": self._opts.timestamps,
                                "diarize": self._opts.diarize,
                                "partials": self._opts.partials,
                                "input_audio_format": self._opts.input_audio_format,
                                "input_sample_rate": self._opts.input_sample_rate,
                            },
                        }
                    )
                )
                if connection_reused:
                    self._session_ready.set()

                send_task = asyncio.create_task(self._send_task(ws), name="hakim-stt-send")
                recv_task = asyncio.create_task(self._recv_task(ws), name="hakim-stt-recv")
                try:
                    # `send_task` is the authoritative "this turn is done"
                    # signal (it returns only after `_all_commits_done`
                    # fires, including the zero-commit/empty-turn case).
                    # `recv_task` normally never finishes on its own — it's
                    # cancelled below once `send_task` completes. Racing on
                    # FIRST_COMPLETED (rather than gather) means a
                    # `recv_task` error surfaces immediately instead of only
                    # after `send_task` also happens to unblock.
                    done, _pending = await asyncio.wait(
                        [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        exc = task.exception()
                        if exc is not None:
                            raise exc
                finally:
                    await utils.aio.gracefully_cancel(send_task, recv_task)
        except websockets.exceptions.InvalidStatus as e:
            raise APIStatusError(f"Hakim STT upgrade rejected: {e}") from e
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            raise APIConnectionError(str(e)) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e

    async def _send_task(self, ws: websockets.ClientConnection) -> None:
        # Wait for the session handshake to complete before streaming any
        # audio — see `_session_ready` in __init__.
        await self._session_ready.wait()

        # Hakim recommends 50-250ms chunks; LiveKit delivers ~10-20ms
        # frames, so re-buffer into ~100ms frames before sending — one
        # WS text message per 100ms instead of one per 10-20ms frame.
        samples_100ms = self._opts.input_sample_rate // 10
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.input_sample_rate,
            num_channels=1,
            samples_per_channel=samples_100ms,
        )

        async def _send_frame(frame: rtc.AudioFrame) -> None:
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(bytes(frame.data)).decode("ascii"),
                    }
                )
            )

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                for frame in audio_bstream.flush():
                    await _send_frame(frame)
                self._pending_commits += 1
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                continue

            frame: rtc.AudioFrame = data
            for out_frame in audio_bstream.write(frame.data.tobytes()):
                await _send_frame(out_frame)

        self._input_closed = True
        self._maybe_finish()
        # Wait for every commit we sent to actually finish
        # (transcription.done received) before returning — see
        # `_all_commits_done` in __init__. Deliberately does NOT send
        # session.close: the connection goes back to the pool for the next
        # turn to reuse. session.close is only ever sent from `_close_ws`,
        # when the pool itself decides to actually tear a connection down.
        await self._all_commits_done.wait()

    async def _recv_task(self, ws: websockets.ClientConnection) -> None:
        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")

            if etype == "session.created":
                self._session_ready.set()
                continue
            if etype is None:
                continue

            if etype == "transcription.delta":
                text = event.get("text", "")
                if not text:
                    continue
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[stt.SpeechData(language=self._opts.language, text=text)],
                    )
                )
            elif etype == "transcription.done":
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                language=event.get("language", self._opts.language),
                                text=event.get("text", ""),
                            )
                        ],
                    )
                )
                self._pending_commits -= 1
                self._maybe_finish()
                if self._ended:
                    return
            elif etype == "error":
                code = event.get("code", "unknown_error")
                message = event.get("message", "")
                retryable = bool(event.get("retryable", False))
                # `_main_task` (base class) retries on `APIError` subclasses
                # and gives up on anything else — mirror that split here.
                if retryable:
                    raise APIConnectionError(f"[{code}] {message}")
                raise HakimStreamError(code, message, retryable=retryable)
