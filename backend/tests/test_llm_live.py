"""Opt-in live provider smoke test; sends synthetic, non-private text."""

import os

import pytest

from ake_backend.llm import OpenAICompatibleProvider
from ake_backend.rag import validate_answer


@pytest.mark.asyncio
async def test_live_provider():
    if os.getenv("RUN_LLM_SMOKE") != "1":
        pytest.skip("Set RUN_LLM_SMOKE=1 and provider environment variables to make a real request")
    model = os.environ["LLM_MODEL"]
    assert model, "LLM_MODEL must be configured"
    provider = OpenAICompatibleProvider(
        os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1",
        os.getenv("LLM_API_KEY", ""),
        model,
    )
    response = await provider.complete(
        "What color is the fictional object?", "[synthetic] The fictional object is blue."
    )
    validate_answer(response, {"synthetic"})


@pytest.mark.asyncio
async def test_live_provider_stream():
    if os.getenv("RUN_LLM_SMOKE") != "1":
        pytest.skip("Set RUN_LLM_SMOKE=1 and provider environment variables to make a real request")
    model = os.environ["LLM_MODEL"]
    assert model, "LLM_MODEL must be configured"
    provider = OpenAICompatibleProvider(
        os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1",
        os.getenv("LLM_API_KEY", ""),
        model,
    )
    parts = [
        part
        async for part in provider.stream(
            "What color is the fictional object?", "[synthetic] The fictional object is blue."
        )
    ]
    assert "".join(parts).strip()
