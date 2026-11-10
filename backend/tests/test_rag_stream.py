import pytest

from ake_backend.rag_stream import stream_answer


class Provider:
    async def stream(self, question, context):
        yield "Claim "
        yield "[a]."


@pytest.mark.asyncio
async def test_stream_provisional_then_validated_final():
    events = [
        event
        async for event in stream_answer(
            Provider(),
            "question",
            "[a] excerpt",
            [{"id": "a", "document_id": "d", "text": "excerpt"}],
            set(),
        )
    ]
    assert [event["event"] for event in events] == ["delta", "delta", "final"]
    assert events[-1]["data"]["answer"] == "Claim [a]."
    assert events[0]["data"]["provisional"] is True


@pytest.mark.asyncio
async def test_stream_invalid_citations_never_returns_final():
    class Bad:
        async def stream(self, question, context):
            yield "Made up [other]."

    events = [
        event
        async for event in stream_answer(
            Bad(),
            "question",
            "[a] excerpt",
            [{"id": "a", "document_id": "d", "text": "excerpt"}],
            set(),
        )
    ]
    assert events[-1]["event"] == "error"
    assert not any(event["event"] == "final" for event in events)


@pytest.mark.asyncio
async def test_stream_handles_abstention_and_limits():
    class Abstain:
        async def stream(self, question, context):
            yield "INSUFFICIENT_"
            yield "EVIDENCE"

    events = [
        event
        async for event in stream_answer(
            Abstain(),
            "question",
            "[a] excerpt",
            [{"id": "a", "document_id": "d", "text": "excerpt"}],
            set(),
        )
    ]
    assert events[-1]["event"] == "final"
    assert events[-1]["data"]["status"] == "insufficient_evidence"
    assert events[-1]["data"]["sources"] == []

    class TooLong:
        async def stream(self, question, context):
            yield "a" * 11

    events = [
        event
        async for event in stream_answer(
            TooLong(),
            "question",
            "[a] excerpt",
            [{"id": "a", "document_id": "d", "text": "excerpt"}],
            set(),
            max_chars=10,
        )
    ]
    assert events[-1]["event"] == "error"
    assert not any(event["event"] == "final" for event in events)


@pytest.mark.asyncio
async def test_unexpected_stream_error_does_not_leak_details():
    class Broken:
        async def stream(self, question, context):
            yield "partial"
            raise RuntimeError("secret provider detail")

    events = [
        event
        async for event in stream_answer(
            Broken(),
            "q",
            "[a] excerpt",
            [{"id": "a", "document_id": "d", "text": "excerpt"}],
            set(),
        )
    ]
    assert events[-1]["event"] == "error"
    assert "secret provider detail" not in str(events[-1])
    assert not any(event["event"] == "final" for event in events)
