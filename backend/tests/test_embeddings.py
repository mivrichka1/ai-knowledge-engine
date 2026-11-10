import math

import pytest

from ake_backend.embeddings import validate_vectors


def test_vector_validation():
    assert validate_vectors([[1.0, 0.0], [0.0, 1.0]], 2, 2) == [[1.0, 0.0], [0.0, 1.0]]
    for invalid in ([[1.0]], [[math.nan, 2.0]], [[0.0, 0.0]], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            validate_vectors(invalid, 2, 2)


def test_provider_rejects_chunks_that_would_be_truncated():
    from ake_backend.embeddings import SentenceTransformerProvider

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [[1] * 129 for _ in texts]}

    class Model:
        max_seq_length = 128

        def encode(self, *args, **kwargs):
            raise AssertionError("encoding must not silently truncate")

    provider = SentenceTransformerProvider.__new__(SentenceTransformerProvider)
    provider.tokenizer = Tokenizer()
    provider.model = Model()
    provider.dimension = 384
    with pytest.raises(ValueError, match="context"):
        provider.encode(["too many tokens"])
