"""Context assembly and citation membership checking, not entailment verification."""

import re


class InsufficientEvidence(ValueError):
    pass


class InvalidCitation(ValueError):
    pass


def build_context(sources: list[dict], max_chars: int = 12000) -> tuple[str, set[str]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    parts: list[str] = []
    ids = set()
    used = 0
    for source in sources:
        identifier = source["id"]
        if not re.fullmatch(r"[A-Za-z0-9-]+", identifier):
            raise ValueError("Invalid source ID")
        if identifier in ids or not source["text"].strip():
            continue
        part = f"[{identifier}]\n{source['text']}"
        extra = len(part) + (2 if parts else 0)
        if used + extra > max_chars:
            continue
        parts.append(part)
        ids.add(identifier)
        used += extra
    if not parts:
        marker = "[excerpt truncated]"
        for source in sources:
            identifier = source["id"]
            prefix = f"[{identifier}]\n"
            available = max_chars - len(prefix) - len(marker) - 1
            if available > 0 and source["text"].strip():
                parts.append(prefix + source["text"][:available] + "\n" + marker)
                ids.add(identifier)
                break
    return "\n\n".join(parts), ids


def cited_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[([A-Za-z0-9-]+)\]", answer))


def validate_answer(answer: str, source_ids: set[str]) -> str:
    if not source_ids:
        raise InsufficientEvidence("No source material")
    references = cited_ids(answer)
    if not references or not references <= source_ids:
        raise InvalidCitation("Answer contains missing or unknown citations")
    return answer


def balanced_sources(groups: list[list[dict]], limit: int) -> list[dict]:
    """Interleave independently retrieved document lists, preserving each ranking."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    iterators = [iter(group) for group in groups]
    seen: set[str] = set()
    results: list[dict] = []
    while iterators and len(results) < limit:
        remaining = []
        for iterator in iterators:
            for item in iterator:
                if item["id"] not in seen:
                    seen.add(item["id"])
                    results.append(item)
                    remaining.append(iterator)
                    break
            if len(results) == limit:
                break
        iterators = remaining
    return results


def validate_document_coverage(
    answer: str, sources: list[dict], selected_documents: set[str]
) -> None:
    """Check each explicitly selected document has a cited chunk; not entailment."""
    references = cited_ids(answer)
    cited_documents = {item["document_id"] for item in sources if item["id"] in references}
    if not selected_documents.issubset(cited_documents):
        raise InvalidCitation("Answer does not cite every selected document")


async def retrieve_for_documents(document_ids, retrieve, limit: int = 8) -> list[dict]:
    """Retrieve separately for each selection, or once across the whole base."""
    groups = []
    for document_id in document_ids or [None]:
        groups.append(await retrieve(document_id))
    return balanced_sources(groups, limit)
