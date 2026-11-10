import pytest

from ake_backend.rag import (
    InsufficientEvidence,
    InvalidCitation,
    build_context,
    validate_answer,
)


def test_context_uses_only_supplied_sources_and_respects_limit():
    sources = [{"id": "a", "text": "alpha beta"}, {"id": "b", "text": "gamma delta"}]
    context, ids = build_context(sources, max_chars=25)
    assert "[a]" in context
    assert ids == {"a"}
    assert len(context) <= 25


def test_answer_requires_valid_citations():
    assert validate_answer("A claim [a].", {"a"}) == "A claim [a]."
    with pytest.raises(InvalidCitation):
        validate_answer("Unknown [b].", {"a"})
    with pytest.raises(InvalidCitation):
        validate_answer("Claim without citation.", {"a"})
    with pytest.raises(InsufficientEvidence):
        validate_answer("anything", set())


def test_oversized_source_does_not_hide_all_evidence():
    context, identifiers = build_context(
        [{"id": "source-1", "text": "useful evidence " * 100}], max_chars=80
    )
    assert len(context) <= 80
    assert identifiers == {"source-1"}
    assert context.startswith("[source-1]\nuseful evidence")
    assert context.endswith("[excerpt truncated]")


def test_context_ignores_empty_sources():
    assert build_context([{"id": "a", "text": "   "}], max_chars=100) == ("", set())


def test_balanced_sources_interleave_documents_and_deduplicate():
    from ake_backend.rag import balanced_sources

    groups = [
        [{"id": "a1", "document_id": "a"}, {"id": "a2", "document_id": "a"}],
        [
            {"id": "b1", "document_id": "b"},
            {"id": "a1", "document_id": "a"},
            {"id": "b2", "document_id": "b"},
        ],
    ]
    assert [item["id"] for item in balanced_sources(groups, limit=3)] == ["a1", "b1", "a2"]
    assert balanced_sources([[], []], limit=2) == []
    with pytest.raises(ValueError):
        balanced_sources(groups, limit=0)


def test_selected_documents_must_each_be_cited():
    from ake_backend.rag import validate_document_coverage

    sources = [{"id": "a", "document_id": "doc-a"}, {"id": "b", "document_id": "doc-b"}]
    validate_document_coverage("First [a], second [b].", sources, {"doc-a", "doc-b"})
    with pytest.raises(InvalidCitation):
        validate_document_coverage("Only first [a].", sources, {"doc-a", "doc-b"})
    validate_document_coverage("Only first [a].", sources, set())


@pytest.mark.asyncio
async def test_retrieval_queries_each_selected_document():
    from ake_backend.rag import retrieve_for_documents

    calls = []

    async def retrieve(document_id):
        calls.append(document_id)
        return [{"id": f"{document_id}-1", "document_id": document_id}]

    results = await retrieve_for_documents(["first", "second"], retrieve, limit=8)
    assert calls == ["first", "second"]
    assert [r["id"] for r in results] == ["first-1", "second-1"]
