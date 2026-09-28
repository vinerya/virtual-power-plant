"""TTL + content-hash cache for price-feed responses.

Two layers:

1. In-memory dict keyed by (feed_name, start_iso, end_iso) -> (expires_at, points).
2. Optional file-system persistence: each entry written as JSON sidecar so cold
   processes don't refetch within the TTL.

Eviction policy
---------------
On every :meth:`get`, expired entries are lazily evicted. There is no
background thread. ``content_hash`` lets the cache deduplicate identical
responses across feeds (e.g. two distinct windows that happen to return
the same wholesale prices).

Test mode
---------
Call :meth:`disable_for_testing` to make all subsequent ``get`` calls
return ``None`` and ``set`` a no-op — useful in unit tests that should
never see stale data.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from .base import PricePoint


class FeedCache:
    """TTL cache for price-feed responses.

    Parameters
    ----------
    ttl_seconds : int
        Default time-to-live (default 300).
    persist_dir : Path | str | None
        Optional directory for JSON sidecar persistence. ``None`` disables.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        persist_dir: str | Path | None = None,
    ) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.persist_dir: Path | None = Path(persist_dir) if persist_dir else None
        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        # key -> (expires_at, points, content_hash)
        self._mem: dict[str, tuple[float, list[PricePoint], str]] = {}
        self._disabled = False
        self._now = time.time  # injectable for tests

    # ----- key + hash --------------------------------------------------

    @staticmethod
    def _key(feed_name: str, start: datetime, end: datetime) -> str:
        return f"{feed_name}|{start.isoformat()}|{end.isoformat()}"

    @staticmethod
    def _content_hash(points: list[PricePoint]) -> str:
        h = hashlib.sha256()
        for p in points:
            h.update(p.timestamp.isoformat().encode())
            h.update(f"{p.price_per_kwh:.10f}".encode())
        return h.hexdigest()

    # ----- public API --------------------------------------------------

    def disable_for_testing(self) -> None:
        """Disable the cache; subsequent get() returns None."""
        self._disabled = True
        self._mem.clear()

    def enable(self) -> None:
        self._disabled = False

    def get(self, feed_name: str, start: datetime, end: datetime) -> list[PricePoint] | None:
        """Return cached points or ``None`` if absent / expired."""
        if self._disabled:
            return None
        key = self._key(feed_name, start, end)
        entry = self._mem.get(key)
        if entry is None:
            # Try persistence
            entry = self._load_persisted(key)
            if entry is None:
                return None
            self._mem[key] = entry
        expires_at, points, _h = entry
        if self._now() >= expires_at:
            self._mem.pop(key, None)
            self._delete_persisted(key)
            return None
        return points

    def set(
        self,
        feed_name: str,
        start: datetime,
        end: datetime,
        points: list[PricePoint],
        ttl_seconds: int | None = None,
    ) -> None:
        if self._disabled:
            return
        ttl = self.ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        expires_at = self._now() + ttl
        key = self._key(feed_name, start, end)
        ch = self._content_hash(points)
        self._mem[key] = (expires_at, points, ch)
        self._persist(key, expires_at, points)

    def evict_expired(self) -> int:
        """Remove all expired entries; return count evicted."""
        now = self._now()
        keys = [k for k, (exp, _, _) in self._mem.items() if now >= exp]
        for k in keys:
            self._mem.pop(k, None)
            self._delete_persisted(k)
        return len(keys)

    # ----- persistence helpers ----------------------------------------

    def _persist_path(self, key: str) -> Path | None:
        if self.persist_dir is None:
            return None
        safe = hashlib.sha1(key.encode()).hexdigest()
        return self.persist_dir / f"{safe}.json"

    def _persist(self, key: str, expires_at: float, points: list[PricePoint]) -> None:
        path = self._persist_path(key)
        if path is None:
            return
        payload = {
            "expires_at": expires_at,
            "points": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "price_per_kwh": p.price_per_kwh,
                    "feed": p.feed,
                    "metadata": p.metadata,
                }
                for p in points
            ],
        }
        try:
            path.write_text(json.dumps(payload))
        except OSError:
            pass

    def _load_persisted(self, key: str) -> tuple[float, list[PricePoint], str] | None:
        path = self._persist_path(key)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            pts = [
                PricePoint(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    price_per_kwh=float(p["price_per_kwh"]),
                    feed=p.get("feed", ""),
                    metadata=p.get("metadata", {}),
                )
                for p in payload["points"]
            ]
            ch = self._content_hash(pts)
            return (float(payload["expires_at"]), pts, ch)
        except (OSError, ValueError, KeyError):
            return None

    def _delete_persisted(self, key: str) -> None:
        path = self._persist_path(key)
        if path is not None and path.exists():
            try:
                path.unlink()
            except OSError:
                pass


# Module-level default cache (opt-in for adapters).
default_cache = FeedCache(ttl_seconds=300)
