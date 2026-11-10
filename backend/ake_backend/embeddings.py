"""Embedding adapter and safety checks; never substitute fake embeddings."""

import math
from typing import Protocol


class EmbeddingInputTooLong(ValueError):
    """Input would otherwise be silently truncated by the embedding model."""


class EmbeddingProvider(Protocol):
    model_name: str
    revision: str | None
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(
    vectors: list[list[float]], expected_count: int, dimension: int
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ValueError("Unexpected embedding count")
    for vector in vectors:
        if len(vector) != dimension or not all(math.isfinite(float(x)) for x in vector):
            raise ValueError("Invalid embedding dimension or non-finite value")
        if not any(x != 0 for x in vector):
            raise ValueError("Zero vector cannot be used for cosine search")
    return vectors


class SentenceTransformerProvider:
    def __init__(self, model_name: str, dimension: int = 384, revision: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name, self.dimension, self.revision = model_name, dimension, revision
        self.model = SentenceTransformer(model_name, revision=revision, device="cpu")
        self.tokenizer = self.model.tokenizer
        if self.model.get_sentence_embedding_dimension() != dimension:
            raise ValueError("Configured dimension does not match embedding model")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        for start in range(0, len(texts), 128):
            tokenized = self.tokenizer(
                texts[start : start + 128], add_special_tokens=True, truncation=False
            )
            if any(len(ids) > self.model.max_seq_length for ids in tokenized["input_ids"]):
                raise EmbeddingInputTooLong(
                    "Text exceeds the embedding model context; reduce chunk size"
                )
        vectors = self.model.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        return validate_vectors(vectors, len(texts), self.dimension)
