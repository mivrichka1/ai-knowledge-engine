"""Text normalization and fixed-size whitespace-token chunking.

Whitespace tokens are an approximation, not an embedding-model tokenizer.
Offsets refer to the exact input passed to fixed_chunks, after any normalization.
"""

import re
import unicodedata
from dataclasses import dataclass


class TextNormalizer:
    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
        lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class ChunkLimitError(ValueError):
    """Chunk count exceeds the configured limit."""


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int
    end: int
    token_count: int
    strategy: str = "fixed_whitespace"


def fixed_chunks(text: str, size: int = 512, overlap: int = 64) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and 0 <= overlap < size")
    tokens = list(re.finditer(r"\S+", text))
    result: list[Chunk] = []
    pos = 0
    while pos < len(tokens):
        stop = min(pos + size, len(tokens))
        start_offset, end_offset = tokens[pos].start(), tokens[stop - 1].end()
        result.append(
            Chunk(len(result), text[start_offset:end_offset], start_offset, end_offset, stop - pos)
        )
        if stop == len(tokens):
            break
        pos = stop - overlap
    return result


def tokenizer_chunks(
    text: str, tokenizer, size: int = 120, overlap: int = 16, max_chunks: int | None = None
) -> list[Chunk]:
    """Split using the embedding tokenizer's offsets, excluding special tokens."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and 0 <= overlap < size")
    encoded = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(encoded["input_ids"]):
        raise ValueError("Tokenizer returned inconsistent offsets")
    if max_chunks is not None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        step = size - overlap
        count = 0 if not offsets else 1 + max(0, (len(offsets) - size + step - 1) // step)
        if count > max_chunks:
            raise ChunkLimitError("Document chunk limit exceeded")
    chunks: list[Chunk] = []
    position = 0
    while position < len(offsets):
        stop = min(position + size, len(offsets))
        start, end = offsets[position][0], offsets[stop - 1][1]
        chunks.append(
            Chunk(len(chunks), text[start:end], start, end, stop - position, "model_tokenizer")
        )
        if stop == len(offsets):
            break
        position = stop - overlap
    return chunks


def recursive_chunks(text: str, max_chars: int = 1200) -> list[Chunk]:
    """Split by paragraphs, sentences, then whitespace; offsets refer to input.

    This experimental character-bounded strategy is not used for indexing and
    does not enforce an embedding model's token budget.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    pieces: list[tuple[int, int]] = []

    def split(start: int, end: int, level: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            return
        if end - start <= max_chars:
            pieces.append((start, end))
            return
        patterns = (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+")
        if level >= len(patterns):
            for offset in range(start, end, max_chars):
                pieces.append((offset, min(end, offset + max_chars)))
            return
        matches = list(re.finditer(patterns[level], text[start:end]))
        if not matches:
            split(start, end, level + 1)
            return
        cursor = start
        for match in matches:
            boundary = start + match.start()
            if boundary > cursor:
                split(cursor, boundary, level + 1)
            cursor = start + match.end()
        split(cursor, end, level + 1)

    split(0, len(text), 0)
    return [
        Chunk(
            i,
            text[start:end],
            start,
            end,
            len(re.findall(r"\S+", text[start:end])),
            "recursive_characters",
        )
        for i, (start, end) in enumerate(pieces)
    ]
