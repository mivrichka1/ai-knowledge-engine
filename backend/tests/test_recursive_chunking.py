import pytest

from ake_backend.text import recursive_chunks


def test_recursive_prefers_paragraph_boundaries_and_preserves_content():
    text = "# Heading\n\nFirst paragraph. Second sentence.\n\nFinal paragraph."
    chunks = recursive_chunks(text, max_chars=35)
    assert chunks[0].text == "# Heading"
    assert chunks[-1].text == "Final paragraph."
    assert all(c.text == text[c.start : c.end] and len(c.text) <= 35 for c in chunks)
    assert "".join(text[c.start : c.end] for c in chunks).replace(" ", "").replace(
        "\n", ""
    ) == text.replace(" ", "").replace("\n", "")


def test_recursive_long_unbroken_and_empty():
    assert recursive_chunks("", max_chars=4) == []
    assert [x.text for x in recursive_chunks("abcdefghij", max_chars=4)] == ["abcd", "efgh", "ij"]
    with pytest.raises(ValueError):
        recursive_chunks("abc", max_chars=0)
