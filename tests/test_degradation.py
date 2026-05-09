"""Tests for the standalone battery-degradation models (Milestone 1)."""

from __future__ import annotations

import sys
from typing import List

import pytest

from vpp.degradation import (
    LFP_PRESET,
    NMC_PRESET,
    CalendarDegradation,
    RainflowDegradation,
    ThroughputDegradation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sawtooth(low: float, high: float, n_cycles: int, samples_per_half: int = 10) -> List[float]:
    """Build a triangular SOC trace with ``n_cycles`` round trips between
    ``low`` and ``high``, starting and ending at ``low``."""
    trace: List[float] = [low]
    for _ in range(n_cycles):
        # Up
        for k in range(1, samples_per_half + 1):
            trace.append(low + (high - low) * k / samples_per_half)
        # Down
        for k in range(1, samples_per_half + 1):
            trace.append(high - (high - low) * k / samples_per_half)
    return trace


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def test_throughput_one_full_cycle() -> None:
    """A single 100 % DoD round-trip should give exactly
    ``(1 - eol_fraction) / cycles_to_eol`` of capacity loss."""
    model = ThroughputDegradation(
        cycles_to_eol=5000.0,
        eol_capacity_fraction=0.8,
        nominal_capacity_kwh=100.0,
    )
    # One full discharge + one full charge = one equivalent full cycle.
    soc = [1.0] + [1.0 - i / 50 for i in range(1, 51)] + [i / 50 for i in range(1, 51)]
    assert soc[0] == 1.0
    assert soc[50] == 0.0
    assert soc[-1] == 1.0
    loss = model.predict_capacity_loss(soc, dt_hours=0.1)
    expected = (1.0 - 0.8) / 5000.0
    assert loss == pytest.approx(expected, rel=1e-9)


def test_throughput_zero_when_idle() -> None:
    model = ThroughputDegradation(cycles_to_eol=5000.0, nominal_capacity_kwh=100.0)
    soc = [0.5] * 100
    assert model.predict_capacity_loss(soc, dt_hours=0.25) == 0.0


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def test_calendar_doubling_at_high_temp() -> None:
    """Arrhenius with Ea ~0.5 eV should accelerate roughly 2x at +10 K."""
    model = CalendarDegradation(
        calendar_life_years_at_25c=10.0,
        eol_capacity_fraction=0.8,
        arrhenius_activation_ev=0.5,
        soc_stress_coefficient=0.0,  # isolate temperature effect
    )
    # ~30 days at 1-hour resolution.
    soc = [0.5] * (24 * 30)
    loss_25 = model.predict_capacity_loss(soc, dt_hours=1.0, temperature_c=25.0)
    loss_35 = model.predict_capacity_loss(soc, dt_hours=1.0, temperature_c=35.0)
    ratio = loss_35 / loss_25
    # 0.5 eV from 25->35 degC gives ~1.88x in theory.
    assert ratio == pytest.approx(1.88, rel=0.05)


def test_calendar_high_soc_increases_loss() -> None:
    model = CalendarDegradation(
        calendar_life_years_at_25c=10.0,
        soc_stress_coefficient=0.5,
    )
    soc_low = [0.3] * 1000
    soc_high = [0.9] * 1000
    loss_low = model.predict_capacity_loss(soc_low, dt_hours=1.0)
    loss_high = model.predict_capacity_loss(soc_high, dt_hours=1.0)
    assert loss_high > loss_low > 0


# ---------------------------------------------------------------------------
# Rainflow
# ---------------------------------------------------------------------------


def test_rainflow_matches_known_trace() -> None:
    """A 10-cycle 50 % DoD sawtooth should rainflow-count to ~10 cycles
    (within 5 %), so the loss should equal 10 * (1 - eol) / N_eol(0.5)."""
    pytest.importorskip("rainflow")
    curve = {0.2: 30000.0, 0.5: 8000.0, 0.8: 3000.0, 1.0: 1500.0}
    model = RainflowDegradation(cycle_life_curve=curve, eol_capacity_fraction=0.8)
    soc = _sawtooth(low=0.25, high=0.75, n_cycles=10, samples_per_half=20)
    loss = model.predict_capacity_loss(soc, dt_hours=0.1)
    # Expected: 10 cycles at DoD=0.5 -> 10 * 0.2 / 8000 = 2.5e-4
    expected = 10.0 * (1.0 - 0.8) / 8000.0
    assert loss == pytest.approx(expected, rel=0.05)


def test_lfp_vs_nmc_presets() -> None:
    """Same SOC trace: LFP should show lower loss than NMC across all models."""
    pytest.importorskip("rainflow")
    soc = _sawtooth(low=0.1, high=0.9, n_cycles=20, samples_per_half=10)
    dt = 0.25

    lfp_tp = ThroughputDegradation(
        nominal_capacity_kwh=100.0, **LFP_PRESET["throughput"]
    )
    nmc_tp = ThroughputDegradation(
        nominal_capacity_kwh=100.0, **NMC_PRESET["throughput"]
    )
    assert lfp_tp.predict_capacity_loss(soc, dt) < nmc_tp.predict_capacity_loss(soc, dt)

    lfp_cal = CalendarDegradation(**LFP_PRESET["calendar"])
    nmc_cal = CalendarDegradation(**NMC_PRESET["calendar"])
    long_soc = [0.7] * (24 * 365)  # one year at 70 % SOC
    assert lfp_cal.predict_capacity_loss(long_soc, 1.0, 25.0) < nmc_cal.predict_capacity_loss(
        long_soc, 1.0, 25.0
    )

    lfp_rf = RainflowDegradation(**LFP_PRESET["rainflow"])
    nmc_rf = RainflowDegradation(**NMC_PRESET["rainflow"])
    assert lfp_rf.predict_capacity_loss(soc, dt) < nmc_rf.predict_capacity_loss(soc, dt)


def test_rainflow_unavailable_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``rainflow`` cannot be imported, construction must raise a clear
    :class:`ImportError`. The error must surface at construction (fail-fast),
    not at first ``predict_capacity_loss`` call."""
    # Force the import to fail by removing the module from sys.modules and
    # blocking its re-import via a meta_path finder.
    monkeypatch.delitem(sys.modules, "rainflow", raising=False)

    class _Blocker:
        @staticmethod
        def find_spec(name, path, target=None):  # type: ignore[no-untyped-def]
            if name == "rainflow":
                raise ImportError("blocked for test")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Blocker()] + sys.meta_path)

    # Reload the degradation models module under the blocked import.
    import importlib

    import vpp.degradation.models as deg_models

    monkeypatch.delitem(sys.modules, "vpp.degradation.models", raising=False)
    deg_models = importlib.import_module("vpp.degradation.models")

    with pytest.raises(ImportError) as excinfo:
        deg_models.RainflowDegradation(cycle_life_curve={0.5: 1000.0})

    msg = str(excinfo.value)
    assert "rainflow" in msg.lower()
    assert "degradation" in msg.lower() or "pip install" in msg.lower()


# ---------------------------------------------------------------------------
# Sanity / API
# ---------------------------------------------------------------------------


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        ThroughputDegradation(cycles_to_eol=-1)
    with pytest.raises(ValueError):
        CalendarDegradation(calendar_life_years_at_25c=0)
    pytest.importorskip("rainflow")
    with pytest.raises(ValueError):
        RainflowDegradation(cycle_life_curve={})

    model = ThroughputDegradation(cycles_to_eol=1000.0, nominal_capacity_kwh=10.0)
    with pytest.raises(ValueError):
        model.predict_capacity_loss([0.5, 1.5], dt_hours=1.0)
    with pytest.raises(ValueError):
        model.predict_capacity_loss([0.5, 0.6], dt_hours=0.0)
