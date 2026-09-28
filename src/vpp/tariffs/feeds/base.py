"""Price feed contract.

A :class:`PriceFeed` is an async source of per-interval energy prices.
Adapters implement :meth:`fetch(start, end)` returning a list of
:class:`PricePoint`. All timestamps are tz-aware UTC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class PricePoint:
    """One energy-price observation.

    Attributes
    ----------
    timestamp : datetime
        Interval-start, tz-aware (UTC recommended).
    price_per_kwh : float
        Price in $/kWh. Wholesale LMPs in $/MWh are converted at adapter level.
    feed : str
        Originating feed name (matches :attr:`PriceFeed.name`).
    metadata : dict
        Free-form: market_run_id, node, currency, source_units, etc.
    """

    timestamp: datetime
    price_per_kwh: float
    feed: str = ""
    metadata: dict = field(default_factory=dict)


class PriceFeed(ABC):
    """Uniform async price-feed contract.

    Subclasses must set ``name`` and ``timezone`` and implement :meth:`fetch`.
    Implementations should be thread-safe and idempotent for a given
    (start, end) window — the :class:`FeedCache` relies on this.
    """

    name: str = ""
    timezone: str = "UTC"

    @abstractmethod
    async def fetch(self, start: datetime, end: datetime, **kwargs: Any) -> list[PricePoint]:
        """Return prices in the half-open interval ``[start, end)``."""
        raise NotImplementedError
