"""Schema for evidence-linked paper metadata extraction."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class EvidenceValue(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")
    value: T | None = None
    source_ids: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: EvidenceValue[str] | None = None
    authors: EvidenceValue[list[str]] | None = None
    publication_year: EvidenceValue[int] | None = None
    abstract: EvidenceValue[str] | None = None
    research_problem: EvidenceValue[str] | None = None
    methods: EvidenceValue[list[str]] | None = None
    datasets: EvidenceValue[list[str]] | None = None
    key_findings: EvidenceValue[list[str]] | None = None
    limitations: EvidenceValue[list[str]] | None = None


def validate_evidence(result: ExtractionResult, allowed_ids: set[str]) -> None:
    for field in type(result).model_fields:
        item = getattr(result, field)
        if item is not None and item.value is not None:
            if not item.source_ids or not set(item.source_ids) <= allowed_ids:
                raise ValueError("Extraction value has missing or unknown source IDs")


def select_extraction_sources(chunks: list[dict], limit: int = 15) -> list[dict]:
    """Select opening and section-like chunks in original order; heuristic only."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = list(range(min(2, len(chunks), limit)))
    terms = (
        "method",
        "approach",
        "dataset",
        "experiment",
        "result",
        "finding",
        "conclusion",
        "limitation",
        "метод",
        "результат",
        "ограничени",
    )
    for index, chunk in enumerate(chunks):
        if len(selected) >= limit:
            break
        if index not in selected and any(term in chunk["text"].casefold() for term in terms):
            selected.append(index)
    for index in range(len(chunks)):
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.append(index)
    return [chunks[index] for index in sorted(selected)]
