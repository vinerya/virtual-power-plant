"""Real-time price feeds (Milestone 4).

Pluggable adapter contract for live wholesale + retail energy prices.

Public API
----------
* :class:`PriceFeed` — abstract async base class with ``fetch(start, end)``.
* :class:`PricePoint` — uniform per-interval price datum.
* :class:`FeedCache` — TTL + content-hash dedup, optional FS persistence.
* Adapters: :class:`CAISOLMPFeed`, :class:`ComEdHourlyFeed`,
  :class:`OpenADRPriceFeed`, :class:`SyntheticFeed`.
"""
from .base import PriceFeed, PricePoint
from .cache import FeedCache
from .caiso_lmp import CAISOLMPFeed
from .comed_hourly import ComEdHourlyFeed
from .openadr_price import OpenADRPriceFeed
from .synthetic import SyntheticFeed

__all__ = [
    "PriceFeed",
    "PricePoint",
    "FeedCache",
    "CAISOLMPFeed",
    "ComEdHourlyFeed",
    "OpenADRPriceFeed",
    "SyntheticFeed",
]
