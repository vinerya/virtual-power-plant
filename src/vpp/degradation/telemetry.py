"""Telemetry-driven battery degradation updater (M3).

Consumes a SOC/temperature trace window for a battery, runs the rainflow +
calendar models, persists the resulting SOH and cumulative-throughput, and
exposes a long-running async loop suitable for FastAPI background tasks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import (
    LFP_PRESET,
    NMC_PRESET,
    CalendarDegradation,
    RainflowDegradation,
)

logger = logging.getLogger(__name__)


@dataclass
class TelemetryWindow:
    """A SOC trace window for a single battery."""

    battery_id: str
    soc_trace: list[float]
    timestamps: list[datetime]
    temperatures_c: list[float] | None = None

    def dt_hours(self) -> float:
        """Average sample interval, in hours.  Defaults to 1 minute if undefined."""
        if len(self.timestamps) < 2:
            return 1.0 / 60.0
        deltas = [
            (self.timestamps[i] - self.timestamps[i - 1]).total_seconds() / 3600.0
            for i in range(1, len(self.timestamps))
        ]
        deltas = [d for d in deltas if d > 0]
        if not deltas:
            return 1.0 / 60.0
        return sum(deltas) / len(deltas)

    def mean_temperature_c(self) -> float:
        if not self.temperatures_c:
            return 25.0
        return sum(self.temperatures_c) / len(self.temperatures_c)


@dataclass
class SOHUpdate:
    """Result of applying a telemetry window to a battery."""

    battery_id: str
    previous_soh: float
    new_soh: float
    loss_fraction: float
    cycle_loss: float
    calendar_loss: float
    cumulative_throughput_kwh: float
    timestamp: datetime


# Heuristic C-rate for residential storage when nominal_energy_kwh is missing.
# A C/4 system has nominal_energy_kwh ~= rated_power_kw * 4.0 hours.
DEFAULT_C_RATE_HOURS = 4.0


def _capacity_kwh(obj) -> float:
    """Return the battery's nominal energy capacity in kWh.

    Prefers ``nominal_energy_kwh`` when present.  Falls back to a documented
    C/4 heuristic (``rated_power * 4.0``) for residential-storage-shaped
    fixtures that pre-date the M4 column.  Returns 1.0 only if both values
    are absent or zero, to avoid divide-by-zero downstream.
    """
    nominal = getattr(obj, "nominal_energy_kwh", None)
    if nominal is not None and float(nominal) > 0:
        return float(nominal)
    rated = float(getattr(obj, "rated_power", 0.0) or 0.0)
    if rated > 0:
        return rated * DEFAULT_C_RATE_HOURS
    return 1.0


def _preset_for(chemistry: str | None) -> dict:
    if chemistry and chemistry.lower() == "nmc":
        return NMC_PRESET
    return LFP_PRESET  # default LFP


class DegradationUpdater:
    """Apply degradation models from telemetry windows and persist SOH state.

    Windowing strategy: each call to :meth:`apply_window` consumes a single,
    *non-overlapping* window provided by the caller.  The caller is expected
    to feed contiguous, non-overlapping windows so that throughput and cycle
    counts are not double-counted.  The periodic loop pulls one fresh window
    per battery per tick from a user-supplied ``fetch_telemetry`` callable.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        default_chemistry: str = "lfp",
    ) -> None:
        self._session_factory = session_factory
        self._default_chemistry = default_chemistry

    async def apply_window(self, window: TelemetryWindow) -> SOHUpdate | None:
        """Run rainflow + calendar models on the window and update the DB."""
        if len(window.soc_trace) < 2:
            return None

        # Local import to avoid circular dependency at module load.
        from vpp.db.models import ResourceModel
        from vpp.db.repositories import BatteryDegradationRepository

        async with self._session_factory() as session:
            obj: ResourceModel | None = await session.get(ResourceModel, window.battery_id)
            if obj is None or obj.resource_type != "battery":
                return None

            chemistry = obj.chemistry or self._default_chemistry
            preset = _preset_for(chemistry)

            rainflow_model = RainflowDegradation(**preset["rainflow"])
            calendar_model = CalendarDegradation(**preset["calendar"])

            dt_h = window.dt_hours()
            temp_c = window.mean_temperature_c()

            cycle_loss = float(
                rainflow_model.predict_capacity_loss(window.soc_trace, dt_h, temperature_c=temp_c)
            )
            calendar_loss = float(
                calendar_model.predict_capacity_loss(window.soc_trace, dt_h, temperature_c=temp_c)
            )
            loss = cycle_loss + calendar_loss

            previous_soh = float(obj.state_of_health)
            new_soh = max(0.0, previous_soh - loss)

            # Throughput in kWh: |dSOC| * nominal_energy_kwh (true capacity).
            throughput_frac = sum(
                abs(window.soc_trace[i] - window.soc_trace[i - 1])
                for i in range(1, len(window.soc_trace))
            )
            capacity_kwh = _capacity_kwh(obj)
            new_throughput_kwh = float(obj.cumulative_throughput_kwh) + (
                throughput_frac * capacity_kwh
            )

            ts = window.timestamps[-1] if window.timestamps else datetime.now(timezone.utc)

            await BatteryDegradationRepository.update_battery_soh(
                session,
                battery_id=obj.id,
                soh=new_soh,
                cum_throughput_kwh=new_throughput_kwh,
                ts=ts,
                loss_fraction=loss,
            )
            await session.commit()

            return SOHUpdate(
                battery_id=obj.id,
                previous_soh=previous_soh,
                new_soh=new_soh,
                loss_fraction=loss,
                cycle_loss=cycle_loss,
                calendar_loss=calendar_loss,
                cumulative_throughput_kwh=new_throughput_kwh,
                timestamp=ts,
            )

    async def run_periodic(
        self,
        fetch_telemetry: Callable[[str], Awaitable[TelemetryWindow] | TelemetryWindow],
        battery_ids: list[str],
        interval_minutes: int = 60,
        max_ticks: int | None = None,
    ) -> None:
        """Long-running async loop that processes one window per battery per tick.

        ``fetch_telemetry`` is invoked once per battery per tick and may be
        sync or async.  Errors on a single battery are logged and skipped --
        the loop continues to the next battery / tick.
        """
        tick = 0
        while True:
            for bid in battery_ids:
                try:
                    res = fetch_telemetry(bid)
                    window = await res if asyncio.iscoroutine(res) else res
                    if window is not None:
                        await self.apply_window(window)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Degradation update failed for battery %s", bid)
            tick += 1
            if max_ticks is not None and tick >= max_ticks:
                return
            await asyncio.sleep(interval_minutes * 60)
