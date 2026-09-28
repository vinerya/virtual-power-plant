"""Battery degradation model implementations.

This module provides three independent degradation models that estimate
fractional capacity loss of a Li-ion battery from a state-of-charge (SOC)
trajectory:

1. :class:`ThroughputDegradation` -- linear loss vs. cumulative kWh
   throughput, expressed in equivalent full cycles (EFC) to end-of-life.
2. :class:`CalendarDegradation` -- time-and-temperature-driven loss with
   an Arrhenius acceleration factor and an SOC-stress multiplier.
3. :class:`RainflowDegradation` -- depth-of-discharge-aware cycle counting
   using the ``rainflow`` PyPI package and a piecewise cycle-life curve.

All three return a *fractional* capacity loss in [0, 1] (e.g. ``0.001``
means 0.1 % of nameplate capacity has been lost).

References
----------
- NREL "Battery Lifetime Analysis and Simulation Tool" (BLAST), Smith et al.,
  https://www.nrel.gov/transportation/blast.html
- Wang et al., "Cycle-life model for graphite-LiFePO4 cells",
  J. Power Sources 196 (2011) 3942-3948.  (LFP cycle-life curve.)
- Schmalstieg et al., "A holistic aging model for Li(NiMnCo)O2 based 18650
  lithium-ion batteries", J. Power Sources 257 (2014) 325-334.  (NMC.)
- Vetter et al., "Ageing mechanisms in lithium-ion batteries",
  J. Power Sources 147 (2005) 269-281.  (Arrhenius / SOC stress.)
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Boltzmann constant in eV/K (Arrhenius with activation energy in eV)
_K_B_EV_PER_K = 8.617333262e-5
_T_REF_K = 298.15  # 25 degC reference


class DegradationModel(ABC):
    """Abstract base for fractional capacity-loss estimators.

    Subclasses implement :meth:`predict_capacity_loss` which consumes a SOC
    trajectory (values in [0, 1]) sampled uniformly at ``dt_hours`` and
    returns a fractional capacity loss in [0, 1].
    """

    @abstractmethod
    def predict_capacity_loss(
        self,
        soc_trace: Sequence[float],
        dt_hours: float,
        temperature_c: float = 25.0,
    ) -> float:
        """Return fractional capacity loss for the given SOC trajectory.

        Parameters
        ----------
        soc_trace : sequence of float
            SOC values in [0, 1], uniformly sampled.
        dt_hours : float
            Time step between consecutive SOC samples, in hours.
        temperature_c : float, optional
            Average cell temperature in degrees Celsius.

        Returns
        -------
        float
            Fractional capacity loss in [0, 1].
        """
        raise NotImplementedError

    @staticmethod
    def _validate_trace(soc_trace: Sequence[float], dt_hours: float) -> None:
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")
        if len(soc_trace) < 2:
            raise ValueError("soc_trace must have at least 2 samples")
        for v in soc_trace:
            if v < 0.0 or v > 1.0:
                raise ValueError("SOC values must lie in [0, 1]")


# ---------------------------------------------------------------------------
# Throughput-based degradation
# ---------------------------------------------------------------------------


@dataclass
class ThroughputDegradation(DegradationModel):
    """Linear capacity-loss model proportional to cumulative kWh throughput.

    The total energy throughput (charge + discharge in kWh) is converted
    to *equivalent full cycles* by dividing by ``2 * nominal_capacity_kwh``
    (one full cycle = one charge + one discharge of the full pack). The
    fractional capacity loss is then::

        loss = (efc / cycles_to_eol) * (1 - eol_capacity_fraction)

    Parameters
    ----------
    cycles_to_eol : float
        Equivalent full cycles to reach end-of-life. Typical values:
        3000-6000 for NMC, 4000-10000 for LFP (NREL BLAST).
    eol_capacity_fraction : float
        Capacity fraction at end-of-life (e.g. 0.8 = 80% remaining).
    nominal_capacity_kwh : float
        Battery nameplate energy capacity in kWh.
    """

    cycles_to_eol: float
    eol_capacity_fraction: float = 0.8
    nominal_capacity_kwh: float = 1.0

    def __post_init__(self) -> None:
        if self.cycles_to_eol <= 0:
            raise ValueError("cycles_to_eol must be positive")
        if not 0.0 < self.eol_capacity_fraction < 1.0:
            raise ValueError("eol_capacity_fraction must be in (0, 1)")
        if self.nominal_capacity_kwh <= 0:
            raise ValueError("nominal_capacity_kwh must be positive")

    def predict_capacity_loss(
        self,
        soc_trace: Sequence[float],
        dt_hours: float,
        temperature_c: float = 25.0,
    ) -> float:
        self._validate_trace(soc_trace, dt_hours)
        # Sum of |dSOC| over the trace gives total throughput as a
        # fraction of capacity (charge+discharge). One full cycle =
        # 2.0 (one full discharge + one full charge).
        throughput_fraction = 0.0
        for i in range(1, len(soc_trace)):
            throughput_fraction += abs(soc_trace[i] - soc_trace[i - 1])
        equivalent_full_cycles = throughput_fraction / 2.0
        loss_per_cycle = (1.0 - self.eol_capacity_fraction) / self.cycles_to_eol
        return equivalent_full_cycles * loss_per_cycle


# ---------------------------------------------------------------------------
# Calendar aging
# ---------------------------------------------------------------------------


@dataclass
class CalendarDegradation(DegradationModel):
    """Time-and-temperature-driven calendar aging.

    Loss accumulates linearly with time at the reference temperature
    (25 degC) and is accelerated by an Arrhenius factor::

        accel(T) = exp( Ea/kB * (1/T_ref - 1/T) )

    A linear SOC-stress multiplier captures the well-known fact that
    high SOC accelerates calendar aging (Vetter 2005, Schmalstieg 2014)::

        stress(soc) = 1 + soc_stress_coefficient * (mean_soc - 0.5)

    Parameters
    ----------
    calendar_life_years_at_25c : float
        Years of float-storage at 25 degC and 50 % SOC to reach
        ``eol_capacity_fraction`` capacity loss.
    eol_capacity_fraction : float
        Capacity fraction at end-of-life.
    arrhenius_activation_ev : float
        Activation energy in eV. ~0.5 eV gives roughly 2x acceleration
        per 10 K above 25 degC (typical for Li-ion calendar fade,
        Schmalstieg 2014).
    soc_stress_coefficient : float
        Linear coefficient on (mean_soc - 0.5). Set to 0 to disable.
    """

    calendar_life_years_at_25c: float
    eol_capacity_fraction: float = 0.8
    arrhenius_activation_ev: float = 0.5
    soc_stress_coefficient: float = 0.5

    def __post_init__(self) -> None:
        if self.calendar_life_years_at_25c <= 0:
            raise ValueError("calendar_life_years_at_25c must be positive")
        if not 0.0 < self.eol_capacity_fraction < 1.0:
            raise ValueError("eol_capacity_fraction must be in (0, 1)")
        if self.arrhenius_activation_ev < 0:
            raise ValueError("arrhenius_activation_ev must be non-negative")

    def _arrhenius(self, temperature_c: float) -> float:
        t_kelvin = temperature_c + 273.15
        if t_kelvin <= 0:
            raise ValueError("temperature must be above absolute zero")
        ea = self.arrhenius_activation_ev
        return math.exp((ea / _K_B_EV_PER_K) * (1.0 / _T_REF_K - 1.0 / t_kelvin))

    def predict_capacity_loss(
        self,
        soc_trace: Sequence[float],
        dt_hours: float,
        temperature_c: float = 25.0,
    ) -> float:
        self._validate_trace(soc_trace, dt_hours)
        mean_soc = sum(soc_trace) / len(soc_trace)
        elapsed_hours = (len(soc_trace) - 1) * dt_hours
        elapsed_years = elapsed_hours / (24.0 * 365.25)

        loss_per_year_ref = (1.0 - self.eol_capacity_fraction) / self.calendar_life_years_at_25c
        accel = self._arrhenius(temperature_c)
        stress = 1.0 + self.soc_stress_coefficient * (mean_soc - 0.5)
        # Clamp stress to a sensible non-negative range.
        stress = max(stress, 0.0)
        return loss_per_year_ref * elapsed_years * accel * stress


# ---------------------------------------------------------------------------
# Rainflow cycle counting
# ---------------------------------------------------------------------------


def _interpolate_cycle_life(curve: dict[float, float], dod: float) -> float:
    """Piecewise-linear interpolation of cycles-to-EOL vs. DoD.

    The curve is supplied as ``{depth_of_discharge: cycles_to_eol}``.
    Below the smallest DoD knot, the smallest-DoD value is used.
    Above the largest knot, the largest-DoD value is used.
    """
    if not curve:
        raise ValueError("cycle_life_curve must be non-empty")
    knots = sorted(curve.items())
    if dod <= knots[0][0]:
        return float(knots[0][1])
    if dod >= knots[-1][0]:
        return float(knots[-1][1])
    for (d0, n0), (d1, n1) in itertools.pairwise(knots):
        if d0 <= dod <= d1:
            frac = (dod - d0) / (d1 - d0)
            return float(n0 + frac * (n1 - n0))
    return float(knots[-1][1])


@dataclass
class RainflowDegradation(DegradationModel):
    """Rainflow-counting capacity-loss model with DoD-dependent cycle cost.

    Uses the ``rainflow`` PyPI package (ASTM E1049 implementation) to
    decompose the SOC trace into half- and full-cycles. Each cycle of
    depth ``d`` contributes::

        d_loss = count * (1 - eol_capacity_fraction) / N_eol(d)

    where ``count`` is 1.0 for full cycles and 0.5 for half cycles, and
    ``N_eol(d)`` is the cycle-life at that DoD interpolated from
    ``cycle_life_curve``.

    Parameters
    ----------
    cycle_life_curve : dict[float, float]
        ``{dod: cycles_to_eol}``. Knots will be sorted; piecewise-linear
        interpolation is used between them.
    eol_capacity_fraction : float
        Capacity fraction at end-of-life.

    Raises
    ------
    ImportError
        At construction time, if the ``rainflow`` package is not
        installed. Install via ``pip install virtual-power-plant[degradation]``.
    """

    cycle_life_curve: dict[float, float]
    eol_capacity_fraction: float = 0.8
    _rainflow_module: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.cycle_life_curve:
            raise ValueError("cycle_life_curve must be non-empty")
        if not 0.0 < self.eol_capacity_fraction < 1.0:
            raise ValueError("eol_capacity_fraction must be in (0, 1)")
        # Validate at construction so callers fail fast.
        try:
            import rainflow as _rf  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in tests
            raise ImportError(
                "RainflowDegradation requires the 'rainflow' package. "
                "Install it with `pip install rainflow` or via the "
                "optional extra `pip install virtual-power-plant[degradation]`."
            ) from exc
        self._rainflow_module = _rf

    def predict_capacity_loss(
        self,
        soc_trace: Sequence[float],
        dt_hours: float,
        temperature_c: float = 25.0,
    ) -> float:
        self._validate_trace(soc_trace, dt_hours)
        rf = self._rainflow_module
        loss_amplitude = 1.0 - self.eol_capacity_fraction
        total_loss = 0.0
        # rainflow.extract_cycles yields (range, mean, count, i_start, i_end)
        for cycle in rf.extract_cycles(list(soc_trace)):  # type: ignore[union-attr]
            rng = float(cycle[0])
            count = float(cycle[2])
            # SOC range is already a fraction in [0, 1], i.e. DoD of the cycle.
            dod = min(max(rng, 0.0), 1.0)
            if dod <= 0.0:
                continue
            n_eol = _interpolate_cycle_life(self.cycle_life_curve, dod)
            if n_eol <= 0:
                continue
            total_loss += count * loss_amplitude / n_eol
        return total_loss


# ---------------------------------------------------------------------------
# Chemistry presets
# ---------------------------------------------------------------------------


# LFP (LiFePO4) preset.
#
# Sources:
# - NREL BLAST LFP defaults: ~6000 EFC to 80 % capacity at 25 degC,
#   ~10-15 year calendar life at 25 degC / 50 % SOC.
# - Wang et al. 2011 cycle-life vs. DoD: roughly 30k cycles at 20 % DoD,
#   8k at 50 % DoD, 3k at 80 % DoD, 1.5k at 100 % DoD.
# LFP is well-known for being more robust at high SOC than NMC, so we use
# a smaller soc_stress_coefficient.
LFP_PRESET: dict[str, dict[str, float]] = {
    "throughput": {
        "cycles_to_eol": 6000.0,
        "eol_capacity_fraction": 0.8,
    },
    "calendar": {
        "calendar_life_years_at_25c": 15.0,
        "eol_capacity_fraction": 0.8,
        "arrhenius_activation_ev": 0.5,
        "soc_stress_coefficient": 0.3,
    },
    "rainflow": {
        "cycle_life_curve": {
            0.2: 30000.0,
            0.5: 8000.0,
            0.8: 3000.0,
            1.0: 1500.0,
        },
        "eol_capacity_fraction": 0.8,
    },
}


# NMC (LiNiMnCoO2) preset.
#
# Sources:
# - NREL BLAST NMC defaults / Schmalstieg 2014: ~3000-4000 EFC to 80 %,
#   ~8-10 year calendar life at 25 degC / 50 % SOC.
# - Schmalstieg 2014 reports stronger SOC-stress sensitivity than LFP and
#   activation energy ~0.5 eV.
NMC_PRESET: dict[str, dict[str, float]] = {
    "throughput": {
        "cycles_to_eol": 3000.0,
        "eol_capacity_fraction": 0.8,
    },
    "calendar": {
        "calendar_life_years_at_25c": 10.0,
        "eol_capacity_fraction": 0.8,
        "arrhenius_activation_ev": 0.5,
        "soc_stress_coefficient": 0.7,
    },
    "rainflow": {
        "cycle_life_curve": {
            0.2: 12000.0,
            0.5: 4000.0,
            0.8: 1500.0,
            1.0: 800.0,
        },
        "eol_capacity_fraction": 0.8,
    },
}
