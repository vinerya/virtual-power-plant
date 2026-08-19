"""Live-price-feed override of TOU rates inside tariff_to_opt_params (M4)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from vpp.tariffs import (
    DemandCharge,
    MeterTrace,
    Tariff,
    TimeOfUseRate,
    TOUSchedule,
)
from vpp.tariffs.feeds import SyntheticFeed, PricePoint
from vpp.tariffs.optimization import tariff_to_opt_params

ALL_DAYS = (True,) * 7
ALL_MONTHS = frozenset(range(1, 13))


def _peak_tou(rate_off: float = 0.10, rate_peak: float = 0.30) -> Tariff:
    tou = TimeOfUseRate(
        periods={
            "off": [TOUSchedule(ALL_DAYS, (0, 16), ALL_MONTHS, rate_off),
                    TOUSchedule(ALL_DAYS, (21, 24), ALL_MONTHS, rate_off)],
            "peak": [TOUSchedule(ALL_DAYS, (16, 21), ALL_MONTHS, rate_peak)],
        }
    )
    return Tariff(name="peak", components=[tou])


@pytest.mark.asyncio
async def test_live_price_overrides_tou():
    """Synthetic prices for hours 16-21 above the TOU peak rate take precedence.

    Walk a 24-hour horizon at 1-hour resolution. The TOU peak (16-21) is
    $0.30/kWh. We feed live prices of $0.80/kWh during 16-21 and $0.05/kWh
    elsewhere. Resolved energy_buy[t] should match the live overrides.
    """
    tariff = _peak_tou(rate_off=0.10, rate_peak=0.30)
    horizon_start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    horizon_hours = 24

    # Build PricePoints for each hour: 16-21 = $0.80, else $0.05.
    overrides: List[PricePoint] = []
    for h in range(horizon_hours):
        ts = horizon_start + timedelta(hours=h)
        price = 0.80 if 16 <= h < 21 else 0.05
        overrides.append(PricePoint(timestamp=ts, price_per_kwh=price, feed="live"))

    params = tariff_to_opt_params(
        tariff=tariff,
        horizon_start=horizon_start,
        horizon_hours=horizon_hours,
        interval_minutes=60,
        nem="none",
        live_price_overrides=overrides,
    )

    assert len(params.energy_buy_per_kwh) == 24
    # Peak hours overridden to live $0.80, eclipsing TOU peak $0.30.
    for h in range(16, 21):
        assert params.energy_buy_per_kwh[h] == pytest.approx(0.80)
    # Off-peak hours overridden to live $0.05 (below TOU off $0.10).
    for h in list(range(0, 16)) + list(range(21, 24)):
        assert params.energy_buy_per_kwh[h] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_synthetic_feed_drives_overrides():
    """End-to-end: SyntheticFeed.fetch -> tariff_to_opt_params override."""
    tariff = _peak_tou()
    horizon_start = datetime(2024, 7, 2, tzinfo=timezone.utc)
    feed = SyntheticFeed(base=0.20, amplitude=0.0, peak_adder=0.50, noise_std=0.0)
    points = await feed.fetch(horizon_start, horizon_start + timedelta(hours=24))

    params = tariff_to_opt_params(
        tariff=tariff,
        horizon_start=horizon_start,
        horizon_hours=24,
        interval_minutes=60,
        nem="none",
        live_price_overrides=points,
    )

    # During 16-21 synthetic price is base + peak = 0.70.
    for h in range(16, 21):
        assert params.energy_buy_per_kwh[h] == pytest.approx(0.70, abs=1e-3)


@pytest.mark.asyncio
async def test_live_override_takes_precedence_over_tou_sell_rate_under_nem2():
    """A live price override supersedes the static tariff schedule for
    *both* buy and sell -- even when the TOU period being overridden has
    its own explicit (now-stale) URDB `sell` rate."""
    tou = TimeOfUseRate(
        periods={
            "off": [TOUSchedule(ALL_DAYS, (0, 16), ALL_MONTHS, 0.10, sell_rate=0.08),
                    TOUSchedule(ALL_DAYS, (21, 24), ALL_MONTHS, 0.10, sell_rate=0.08)],
            "peak": [TOUSchedule(ALL_DAYS, (16, 21), ALL_MONTHS, 0.30, sell_rate=0.25)],
        }
    )
    tariff = Tariff(name="peak-sell-aware", components=[tou])
    horizon_start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    horizon_hours = 24

    # Override only the peak window with a real-time price well above both
    # the TOU rate and its sell_rate.
    overrides: List[PricePoint] = [
        PricePoint(timestamp=horizon_start + timedelta(hours=h), price_per_kwh=0.80, feed="live")
        for h in range(16, 21)
    ]

    params = tariff_to_opt_params(
        tariff=tariff,
        horizon_start=horizon_start,
        horizon_hours=horizon_hours,
        interval_minutes=60,
        nem="nem2",
        live_price_overrides=overrides,
    )

    for h in range(16, 21):
        assert params.energy_buy_per_kwh[h] == pytest.approx(0.80)
        # Sell mirrors the live-overridden buy price, NOT the static 0.25 sell_rate.
        assert params.energy_sell_per_kwh[h] == pytest.approx(0.80)
    for h in list(range(0, 16)) + list(range(21, 24)):
        assert params.energy_buy_per_kwh[h] == pytest.approx(0.10)
        # Untouched steps still prefer the explicit sell_rate.
        assert params.energy_sell_per_kwh[h] == pytest.approx(0.08)
