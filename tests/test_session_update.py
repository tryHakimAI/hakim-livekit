"""The `session.update` frame the plugin sends at the start of every turn (no network)."""

import asyncio

from hakim_livekit import HakimSTT


def _frame(stream_kwargs=None, **kwargs):
    async def go():
        engine = HakimSTT(api_key="hk_test", **kwargs)
        stream = engine.stream(**(stream_kwargs or {}))
        try:
            return stream._session_update_frame()
        finally:
            await stream.aclose()
            await engine.aclose()

    return asyncio.run(go())


def test_default_session_has_no_language_hints_key():
    frame = _frame(language="ar")
    assert frame["type"] == "session.update"
    assert frame["session"]["language"] == "ar"
    # Single-language sessions are byte-for-byte what they were before.
    assert "language_hints" not in frame["session"]


def test_language_hints_are_sent_when_set():
    frame = _frame(language="auto", language_hints=["ar", "en"])
    assert frame["session"]["language"] == "auto"
    assert frame["session"]["language_hints"] == ["ar", "en"]


def test_empty_hints_are_treated_as_unset():
    assert "language_hints" not in _frame(language_hints=[])["session"]
    assert "language_hints" not in _frame(language_hints=None)["session"]


def test_hints_are_copied_not_aliased():
    hints = ["ar", "en"]

    async def go():
        engine = HakimSTT(api_key="hk_test", language_hints=hints)
        hints.append("fr")
        stream = engine.stream()
        try:
            return stream._session_update_frame()["session"]["language_hints"]
        finally:
            await stream.aclose()
            await engine.aclose()

    assert asyncio.run(go()) == ["ar", "en"]


def test_per_stream_language_override_keeps_the_hints():
    frame = _frame(
        stream_kwargs={"language": "en"}, language="ar", language_hints=["ar", "en"]
    )
    assert frame["session"]["language"] == "en"
    assert frame["session"]["language_hints"] == ["ar", "en"]
