import pytest
from pydantic import ValidationError

from ake_backend.config import Settings

SECRET = "test-only-secret-at-least-32-characters"


def test_chunk_settings_must_have_valid_overlap():
    settings = Settings(jwt_secret=SECRET, chunk_size=96, chunk_overlap=12)
    assert settings.chunk_size == 96 and settings.chunk_overlap == 12
    with pytest.raises(ValidationError):
        Settings(jwt_secret=SECRET, chunk_size=96, chunk_overlap=96)
    with pytest.raises(ValidationError):
        Settings(jwt_secret=SECRET, chunk_size=0, chunk_overlap=0)
