"""Per-process bounded sliding-window limiter; not a distributed security control."""

from collections import OrderedDict, deque
from time import monotonic


class RateLimiter:
    def __init__(self, limit: int = 20, window_seconds: int = 60, max_keys: int = 10000):
        if limit <= 0 or window_seconds <= 0 or max_keys <= 0:
            raise ValueError("Rate limit settings must be positive")
        self.limit, self.window_seconds, self.max_keys = limit, window_seconds, max_keys
        self.entries: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str, now: float | None = None) -> None:
        now = monotonic() if now is None else now
        if key not in self.entries:
            if len(self.entries) >= self.max_keys:
                self.entries.popitem(last=False)
            self.entries[key] = deque()
        bucket = self.entries[key]
        self.entries.move_to_end(key)
        while bucket and bucket[0] <= now - self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.limit:
            raise ValueError("Rate limit exceeded")
        bucket.append(now)
