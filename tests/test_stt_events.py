"""Event-mapping tests for HakimSpeechStream._recv_task (no network).

Feeds the stream the server frames a real session produces — `transcription.delta`
fragments (finalized, or `tentative` replace-frames) then one `transcription.done` per
utterance — and checks the LiveKit events that come out.
"""

import asyncio
import json

from livekit.agents import stt

from hakim_livekit import HakimSTT

E = stt.SpeechEventType


class FakeWS:
    def __init__(self, frames):
        self._frames = [json.dumps(f) for f in frames]

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for f in self._frames:
            yield f


async def _run(frames):
    engine = HakimSTT(api_key="hk_test", language="ar")
    stream = engine.stream()
    try:
        stream._pending_commits = 0
        stream._input_closed = False
        # Recv until the fake socket is exhausted (or the stream signals it is finished).
        await stream._recv_task(FakeWS(frames))
        events = []
        while True:
            try:
                events.append(stream._event_ch.recv_nowait())
            except Exception:
                break
        return events
    finally:
        await stream.aclose()
        await engine.aclose()


def _summary(events):
    return [(e.type, e.alternatives[0].text if e.alternatives else None) for e in events]


def _types(frames):
    return [t for t, _ in _summary(asyncio.run(_run(frames)))]


def test_one_utterance_maps_to_start_interim_final_end():
    frames = [
        {"type": "session.created"},
        {"type": "transcription.delta", "text": "Hello, I'm calling about my", "is_final": False},
        {"type": "transcription.delta", "text": " bank account", "is_final": False},
        {"type": "transcription.done", "text": "Hello, I'm calling about my bank account"},
    ]
    assert _summary(asyncio.run(_run(frames))) == [
        (E.START_OF_SPEECH, None),
        (E.INTERIM_TRANSCRIPT, "Hello, I'm calling about my"),
        (E.INTERIM_TRANSCRIPT, "Hello, I'm calling about my bank account"),
        (E.FINAL_TRANSCRIPT, "Hello, I'm calling about my bank account"),
        (E.END_OF_SPEECH, None),
    ]


def test_second_utterance_starts_fresh():
    frames = [
        {"type": "transcription.delta", "text": "one", "is_final": False},
        {"type": "transcription.done", "text": "one"},
        {"type": "transcription.delta", "text": "two", "is_final": False},
        {"type": "transcription.done", "text": "two"},
    ]
    summary = _summary(asyncio.run(_run(frames)))
    assert [t for t, _ in summary] == [
        E.START_OF_SPEECH, E.INTERIM_TRANSCRIPT, E.FINAL_TRANSCRIPT, E.END_OF_SPEECH,
        E.START_OF_SPEECH, E.INTERIM_TRANSCRIPT, E.FINAL_TRANSCRIPT, E.END_OF_SPEECH,
    ]
    assert summary[5] == (E.INTERIM_TRANSCRIPT, "two")


def test_done_without_deltas_still_brackets_the_final():
    assert _summary(asyncio.run(_run([{"type": "transcription.done", "text": "short"}]))) == [
        (E.START_OF_SPEECH, None),
        (E.FINAL_TRANSCRIPT, "short"),
        (E.END_OF_SPEECH, None),
    ]


def test_empty_done_emits_no_speech_boundaries():
    assert _types([{"type": "transcription.done", "text": ""}]) == [E.FINAL_TRANSCRIPT]


def test_tentative_frames_replace_the_tail_and_finals_append():
    frames = [
        {"type": "transcription.delta", "text": "hel", "is_final": False, "tentative": True},
        {"type": "transcription.delta", "text": "hello wor", "is_final": False, "tentative": True},
        {"type": "transcription.delta", "text": "hello", "is_final": True},
        {"type": "transcription.delta", "text": " wor", "is_final": False, "tentative": True},
        {"type": "transcription.delta", "text": " world", "is_final": True},
        {"type": "transcription.done", "text": "hello world"},
    ]
    assert _summary(asyncio.run(_run(frames))) == [
        (E.START_OF_SPEECH, None),
        (E.INTERIM_TRANSCRIPT, "hel"),
        (E.INTERIM_TRANSCRIPT, "hello wor"),
        (E.INTERIM_TRANSCRIPT, "hello"),
        (E.INTERIM_TRANSCRIPT, "hello wor"),
        (E.INTERIM_TRANSCRIPT, "hello world"),
        (E.FINAL_TRANSCRIPT, "hello world"),
        (E.END_OF_SPEECH, None),
    ]


def test_empty_tentative_frame_clears_the_tail():
    frames = [
        {"type": "transcription.delta", "text": "abc", "is_final": True},
        {"type": "transcription.delta", "text": "xyz", "is_final": False, "tentative": True},
        {"type": "transcription.delta", "text": "", "is_final": False, "tentative": True},
        {"type": "transcription.done", "text": "abc"},
    ]
    interims = [t for k, t in _summary(asyncio.run(_run(frames))) if k == E.INTERIM_TRANSCRIPT]
    assert interims == ["abc", "abcxyz", "abc"]


def test_older_server_without_tentative_field_behaves_as_before():
    frames = [
        {"type": "transcription.delta", "text": "one ", "is_final": False},
        {"type": "transcription.delta", "text": "two", "is_final": False},
        {"type": "transcription.done", "text": "one two"},
    ]
    interims = [t for k, t in _summary(asyncio.run(_run(frames))) if k == E.INTERIM_TRANSCRIPT]
    assert interims == ["one ", "one two"]
