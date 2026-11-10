import re

from ake_backend.text import tokenizer_chunks


class Tokenizer:
    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {
            "offset_mapping": [m.span() for m in matches],
            "input_ids": list(range(len(matches))),
        }


def test_model_token_chunks_retain_offsets_and_overlap():
    text = "один two  three четыре five"
    chunks = tokenizer_chunks(text, Tokenizer(), size=3, overlap=1)
    assert [c.text for c in chunks] == ["один two  three", "three четыре five"]
    assert [c.token_count for c in chunks] == [3, 3]
    assert all(text[c.start : c.end] == c.text for c in chunks)


def test_tokenizer_chunk_limit_rejects_before_building_unbounded_list():
    import pytest

    from ake_backend.text import ChunkLimitError

    with pytest.raises(ChunkLimitError):
        tokenizer_chunks("one two three four five", Tokenizer(), size=1, overlap=0, max_chunks=3)


def test_tokenizer_chunk_limit_allows_exact_limit_and_short_last_chunk():
    chunks = tokenizer_chunks(
        "one two three four five", Tokenizer(), size=2, overlap=1, max_chunks=4
    )
    assert len(chunks) == 4
    assert chunks[-1].text == "four five"
