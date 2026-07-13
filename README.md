# Hakim + LiveKit Agents

Drop-in STT and TTS plugins that let a [LiveKit Agents](https://docs.livekit.io/agents/) voice
agent use Hakim for realtime Arabic-first speech-to-text and text-to-speech.

```
.
├── README.md                 ← you are here
├── requirements.txt
├── .env.example
├── hakim_livekit/
│   ├── __init__.py            exports HakimSTT, HakimTTS
│   ├── stt.py                 WSS /v1/audio/transcriptions/stream
│   ├── tts.py                 WSS /v1/audio/speech/stream + POST /v1/audio/speech
│   └── _common.py             shared auth/URL helpers
└── examples/
    ├── voice_agent.py          full LiveKit AgentSession example
    └── smoke_test.py           connectivity check, no LiveKit dependency
```

---

## 1. Architecture

```
┌─────────────┐   room audio    ┌──────────────────┐
│  LiveKit     │ ───────────────▶│  Your Agent      │
│  room/SFU    │                 │  (this process)  │
│              │◀─────────────── │                  │
└─────────────┘   synthesized    └─────┬──────┬─────┘
                    speech              │      │
                                        │      │
                          HakimSTT.stream()   HakimTTS.stream()
                                        │      │
                                        ▼      ▼
                        wss://api.tryhakim.ai/v1/audio/
                        transcriptions/stream   speech/stream
                                        │      │
                                        ▼      ▼
                                  Hakim inference platform
```

- `HakimSTT` opens one WebSocket per LiveKit `RecognizeStream` (i.e. per participant turn-taking
  session) to `WSS /v1/audio/transcriptions/stream`, forwards LiveKit's audio frames as
  `input_audio_buffer.append`, and turns `transcription.delta` / `transcription.done` frames into
  LiveKit `SpeechEvent`s.
- `HakimTTS` opens one WebSocket per LiveKit `SynthesizeStream` to `WSS /v1/audio/speech/stream`.
  Each sentence LiveKit flushes during LLM token streaming becomes one `speech.create` frame on
  the same socket — no new TCP/TLS handshake per utterance, which is what makes this faster than
  calling the batch `POST /v1/audio/speech` endpoint per sentence.
- Both plugins also implement the non-streaming fallback LiveKit expects (`recognize()` /
  `synthesize()`) against the batch HTTP endpoints, so they work anywhere a LiveKit STT/TTS plugin
  is expected, not just inside a live `AgentSession`.

---

## 2. Requirements

- Python ≥ 3.10 (matches LiveKit Agents' own requirement)
- A Hakim API key with `stt:write` and `tts:write` scopes (Dashboard → Settings → API keys)
- A LiveKit Cloud project or self-hosted LiveKit server
- `livekit-agents` ≥ 1.0 (install separately — this repo does not vendor it)

```bash
pip install -r requirements.txt
# the example agent also needs an LLM + VAD plugin of your choice:
pip install livekit-plugins-silero livekit-plugins-openai
```

---

## 3. Quickstart

```bash
cp .env.example .env
# fill in HAKIM_API_KEY, HAKIM_VOICE, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
```

Sanity-check connectivity first (isolates "Hakim reachability" from "LiveKit wiring" when
debugging):

```bash
python examples/smoke_test.py "مرحبا بالعالم"
```

Then run the full agent:

```bash
python examples/voice_agent.py dev
```

Connect a client to your LiveKit room (e.g. the [Agents Playground](https://agents-playground.livekit.io/))
and talk to it.

### Minimal usage

```python
from hakim_livekit import HakimSTT, HakimTTS

session = AgentSession(
    vad=silero.VAD.load(),
    stt=HakimSTT(language="ar"),
    llm=your_llm_plugin,
    tts=HakimTTS(voice="cmokbc2b1001pvu39wmj61b7h"),
)
```

`HAKIM_API_KEY` is read from the environment by default; pass `api_key=...` explicitly to either
constructor to override it.

---

## 4. Configuration reference

### `HakimSTT`

| Param                | Default          | Notes                                                                                          |
| -------------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| `api_key`            | `$HAKIM_API_KEY` |                                                                                                |
| `language`           | `"ar"`           | Per-call override via `stream(language=...)`.                                                  |
| `timestamps`         | `"segment"`      | `"word" \| "segment" \| "none"`.                                                               |
| `diarize`            | `False`          | Stereo call-recording use case only (see §6).                                                  |
| `partials`           | `True`           | Emit `INTERIM_TRANSCRIPT` events.                                                              |
| `input_audio_format` | `"pcm16"`        | `"pcm16" \| "opus" \| "mulaw"` — matches LiveKit's native frame format, no transcoding needed. |
| `input_sample_rate`  | `16000`          | Must match the sample rate of frames actually pushed.                                          |
| `region`             | `"auto"`         | `"auto" \| "de" \| "uae" \| "ksa"`.                                                            |
| `base_url`           | —                | Overrides `region` entirely (staging/self-hosted).                                             |

The model is always `hakim-arab-v2` — Hakim's Arabic-first acoustic profile, the only model this
endpoint accepts.

### `HakimTTS`

| Param          | Default           | Notes                                                     |
| -------------- | ----------------- | --------------------------------------------------------- |
| `voice`        | _(required)_      | Hakim voice id or slug — Dashboard → Voices.              |
| `api_key`      | `$HAKIM_API_KEY`  |                                                           |
| `model`        | `"hakim-fast-v1"` | `"hakim-fast-v1"` (sub-120ms TTFB)                        |
| `cfg`          | `3.0`             | Classifier-free-guidance weight, `0.0`–`10.0`.            |
| `voice_prompt` | `None`            | Free-form voice description; only honoured on `hakim-v3`. |
| `region`       | `"auto"`          | Same as above.                                            |
| `base_url`     | —                 | Same as above.                                            |

Sample rate is fixed at 24 kHz mono PCM — the realtime endpoint's engine-native rate; it isn't
configurable on this surface (use the batch HTTP API directly outside LiveKit if you need another
rate/container).

---

## 5. Regions

Hakim runs a primary plane plus regional deployments. Pin a region if your LiveKit infra is
colocated with one of them, to avoid an extra inter-region hop:

| `region=`          | Host                                       |
| ------------------ | ------------------------------------------ |
| `"auto"` (default) | `api.tryhakim.ai` — closest healthy region |
| `"de"`             | `de.api.tryhakim.ai` (Frankfurt)           |
| `"uae"`            | `uae.api.tryhakim.ai`                      |
| `"ksa"`            | `ksa.api.tryhakim.ai`                      |

`base_url` overrides the table entirely, for staging or self-hosted deployments.

---

## 6. Troubleshooting

| Symptom                                                | Likely cause                                   | Fix                                                                                                                                                                             |
| ------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: Hakim API key not set`                    | `HAKIM_API_KEY` missing/empty                  | Set the env var or pass `api_key=` explicitly.                                                                                                                                  |
| `websockets.exceptions.InvalidStatus` on connect (401) | Bad/revoked API key                            | Regenerate the key in the Hakim dashboard.                                                                                                                                      |
| ... (403)                                              | Key lacks `stt:write` / `tts:write` scope      | Check the key's scopes in the dashboard.                                                                                                                                        |
| ... (429)                                              | Rate limit hit                                 | Back off; check your plan's rate limit.                                                                                                                                         |
| `APIConnectionError` with `upstream_unavailable`       | Hakim's inference pool reports no healthy host | Transient — retry with backoff; LiveKit's built-in retry (`conn_options`) already does this.                                                                                    |
| Audio choppy / gaps between sentences                  | Each sentence opening a _new_ connection       | Confirm you're using `HakimTTS.stream()` (persistent socket), not calling `.synthesize()` per sentence yourself.                                                                |
| `binary_frames_not_supported` error frame              | Client sent a binary WS frame                  | Shouldn't happen through this plugin — only relevant if you're extending `_input_task`/`_send_task` yourself.                                                                   |
| Garbled/silent audio                                   | Sample rate mismatch                           | STT: confirm `input_sample_rate` matches what you actually push. TTS: the plugin always requests 24 kHz — don't resample its output before it reaches LiveKit's audio pipeline. |

Run `examples/smoke_test.py` first for any "no audio" / "no transcript" report — it isolates
whether the problem is network/auth (visible without LiveKit at all) or agent wiring.

Every Hakim response includes an `X-Request-Id` header (HTTP) or `request_id` field (WS frames) —
include it when filing a support ticket.

---

## License

MIT — see [LICENSE](./LICENSE).
