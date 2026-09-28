"""Meter trace data structure for tariff billing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class MeterTrace:
    """Time-series of metered import/export energy.

    Attributes
    ----------
    timestamps : list[datetime]
        Interval-start timestamps, must be timezone-aware (UTC recommended).
    import_kwh : list[float]
        kWh consumed from grid in each interval.
    export_kwh : list[float]
        kWh exported to grid in each interval. NEM export compensation is
        computed by callers (e.g. the bill simulator's `nem` parameter,
        tariffs.optimization's `nem` mode), not by MeterTrace itself.
    interval_minutes : int
        Length of each interval in minutes (e.g. 15, 60).
    tz : timezone
        Local timezone for season/hour-of-day classification.
    """

    timestamps: list[datetime]
    import_kwh: list[float]
    export_kwh: list[float] = field(default_factory=list)
    interval_minutes: int = 60
    tz: timezone = field(default_factory=lambda: timezone.utc)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if len(self.import_kwh) != n:
            raise ValueError("import_kwh length mismatch with timestamps")
        if not self.export_kwh:
            self.export_kwh = [0.0] * n
        if len(self.export_kwh) != n:
            raise ValueError("export_kwh length mismatch with timestamps")
        for ts in self.timestamps:
            if ts.tzinfo is None:
                raise ValueError("timestamps must be timezone-aware")

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def interval_hours(self) -> float:
        return self.interval_minutes / 60.0

    def local_time(self, idx: int) -> datetime:
        return self.timestamps[idx].astimezone(self.tz)

    def kw_at(self, idx: int) -> float:
        """Average kW during interval idx (net import; negative if export dominates)."""
        net = self.import_kwh[idx] - self.export_kwh[idx]
        return net / self.interval_hours

    def kwh_in_window(self, start: datetime, end: datetime) -> float:
        """Total imported kWh whose interval-start lies in [start, end)."""
        total = 0.0
        for ts, kwh in zip(self.timestamps, self.import_kwh):
            if start <= ts < end:
                total += kwh
        return total

    def peak_kw_in_window(self, start: datetime, end: datetime) -> float:
        """Peak average-kW interval (import only) in [start, end)."""
        peak = 0.0
        for i, ts in enumerate(self.timestamps):
            if start <= ts < end:
                kw = self.import_kwh[i] / self.interval_hours
                if kw > peak:
                    peak = kw
        return peak

    def iter_with_local(self) -> Iterator[tuple[int, datetime, float, float]]:
        """Yield (idx, local_dt, import_kwh, export_kwh)."""
        for i, ts in enumerate(self.timestamps):
            yield i, ts.astimezone(self.tz), self.import_kwh[i], self.export_kwh[i]

    @classmethod
    def constant_load(
        cls,
        kw: float,
        start: datetime,
        days: int,
        interval_minutes: int = 60,
        tz: timezone = timezone.utc,
    ) -> MeterTrace:
        """Build a synthetic constant-kW trace, useful for smoke tests."""
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        n = int(days * 24 * 60 / interval_minutes)
        step = timedelta(minutes=interval_minutes)
        timestamps = [start + i * step for i in range(n)]
        kwh_per_interval = kw * (interval_minutes / 60.0)
        return cls(
            timestamps=timestamps,
            import_kwh=[kwh_per_interval] * n,
            export_kwh=[0.0] * n,
            interval_minutes=interval_minutes,
            tz=tz,
        )
