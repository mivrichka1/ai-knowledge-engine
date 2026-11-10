"""Validate provisional streamed text before treating an answer as final."""

import json
import logging
from collections.abc import AsyncIterator

from .llm import LLMError
from .rag import cited_ids, validate_answer, validate_document_coverage


async def stream_answer(
    provider,
    question: str,
    context: str,
    sources: list[dict],
    selected_documents: set[str],
    max_chars: int = 30000,
) -> AsyncIterator[dict]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    answer = ""
    try:
        async for part in provider.stream(question, context):
            answer += part
            if len(answer) > max_chars:
                raise LLMError("Stream response is too long")
            yield {"event": "delta", "data": {"text": part, "provisional": True}}
        if answer.strip() == "INSUFFICIENT_EVIDENCE":
            yield {
                "event": "final",
                "data": {
                    "answer": "В доступных документах недостаточно информации, чтобы ответить на этот вопрос.",
                    "status": "insufficient_evidence",
                    "sources": [],
                },
            }
            return
        allowed = {source["id"] for source in sources}
        validate_answer(answer, allowed)
        validate_document_coverage(answer, sources, selected_documents)
        citations = cited_ids(answer)
        yield {
            "event": "final",
            "data": {
                "answer": answer,
                "status": "answered",
                "sources": [source for source in sources if source["id"] in citations],
            },
        }
    except Exception as exc:
        logging.getLogger("ake.rag").warning(
            json.dumps({"event": "stream_failed", "error_type": type(exc).__name__})
        )
        yield {
            "event": "error",
            "data": {
                "message": "Response could not be validated or the provider stream failed. Discard provisional text."
            },
        }
