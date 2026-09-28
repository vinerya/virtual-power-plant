"""Calendar helpers for tariff engine: seasons + US federal holidays.

Holidays are hardcoded (observed dates 2020-2030) rather than depending on the
`holidays` PyPI package — keeps M1 zero-dep. Extend `US_FEDERAL_HOLIDAYS`
or swap to the `holidays` package later if a longer horizon is needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime

# Default seasons: Northern Hemisphere convention used by most US IOUs.
# Summer = May–September inclusive, Winter = the rest.
DEFAULT_SUMMER_MONTHS = frozenset({5, 6, 7, 8, 9})


# Hardcoded observed-date US federal holidays (2020-2030). Sourced from
# OPM's federal holiday calendar (https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/).
# In M1 we treat these as "weekend"/"off-peak" days for URDB schedule index 1
# (URDB schedules use the same weekday/weekend table for holidays by convention).
US_FEDERAL_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2024
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 2, 19),
        date(2024, 5, 27),
        date(2024, 6, 19),
        date(2024, 7, 4),
        date(2024, 9, 2),
        date(2024, 10, 14),
        date(2024, 11, 11),
        date(2024, 11, 28),
        date(2024, 12, 25),
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 10, 13),
        date(2025, 11, 11),
        date(2025, 11, 27),
        date(2025, 12, 25),
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 10, 12),
        date(2026, 11, 11),
        date(2026, 11, 26),
        date(2026, 12, 25),
    }
)


@dataclass(frozen=True)
class SeasonConfig:
    """Configurable season detection. Defaults to Northern Hemisphere."""

    summer_months: frozenset[int] = field(default_factory=lambda: DEFAULT_SUMMER_MONTHS)

    def is_summer(self, dt: datetime | date) -> bool:
        return dt.month in self.summer_months

    def season(self, dt: datetime | date) -> str:
        return "summer" if self.is_summer(dt) else "winter"


def is_us_holiday(d: date | datetime, extra: Iterable[date] = ()) -> bool:
    """True if `d` is a US federal holiday (observed)."""
    if isinstance(d, datetime):
        d = d.date()
    return d in US_FEDERAL_HOLIDAYS or d in set(extra)


def is_weekend_or_holiday(dt: datetime | date) -> bool:
    """URDB convention: weekday schedule applies Mon-Fri non-holiday;
    weekend schedule applies Sat/Sun and federal holidays."""
    if isinstance(dt, datetime):
        d = dt.date()
        wd = dt.weekday()
    else:
        d = dt
        wd = dt.weekday()
    return wd >= 5 or is_us_holiday(d)
