import pytest

from ake_backend.rate_limit import RateLimiter


def test_rate_limiter_rejects_then_resets():
    limiter = RateLimiter(limit=2, window_seconds=60)
    limiter.check("a", now=0)
    limiter.check("a", now=1)
    with pytest.raises(ValueError):
        limiter.check("a", now=2)
    limiter.check("b", now=2)
    limiter.check("a", now=61)
