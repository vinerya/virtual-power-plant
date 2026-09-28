"""ISO-8601 parsing for API timestamps in tests.

The API serialises UTC as a trailing ``Z``; ``datetime.fromisoformat`` only
accepts that from Python 3.11, and the project supports 3.10.
"""

from __future__ import annotations

from datetime import datetime


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
