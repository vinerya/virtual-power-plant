"""V2G scheduler — rule-based and optimisation-based EV charging schedules.

The rule-based scheduler is the production default.  The optimiser formulation
is available when ``pulp`` or ``cvxpy`` is installed and creates a linear
program that minimises cost while respecting departure-SOC constraints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from vpp.v2g.models import (
    EVBattery,
    EVFleet,
    FlexibilityWindow,
    ChargingSession,
    ScheduleStatus,
    EVConnectionState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule slot
# ---------------------------------------------------------------------------

@dataclass
class ScheduleSlot:
    """One time slot in a V2G schedule."""

    start_time: float
    end_time: float
    power_kw: float          # positive = charge, negative = discharge
    ev_id: str = ""
    price: float = 0.0       # $/kWh at this slot

    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time) / 3600

    @property
    def energy_kwh(self) -> float:
        return self.power_kw * self.duration_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "power_kw": round(self.power_kw, 2),
            "ev_id": self.ev_id,
            "price": self.price,
            "energy_kwh": round(self.energy_kwh, 3),
        }


@dataclass
class V2GScheduleResult:
    """Result of a scheduling run."""

    schedule: list[ScheduleSlot] = field(default_factory=list)
    total_cost: float = 0.0
    total_revenue: float = 0.0
    method: str = "rule_based"
    solve_time_ms: float = 0.0

    @property
    def net_cost(self) -> float:
        return self.total_cost - self.total_revenue

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": [s.to_dict() for s in self.schedule],
            "total_cost": round(self.total_cost, 4),
            "total_revenue": round(self.total_revenue, 4),
            "net_cost": round(self.net_cost, 4),
            "method": self.method,
            "solve_time_ms": round(self.solve_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class V2GScheduler:
    """Production-grade V2G scheduler.

    The **rule-based** path is always available and is the production default.
    It uses a simple "charge-when-cheap, discharge-when-expensive" heuristic
    that respects departure SOC constraints.

    The **optimised** path formulates an LP using ``pulp`` and is used when
    ``use_optimiser=True`` and prices are provided.
    """

    def __init__(self, slot_duration_minutes: int = 15) -> None:
        self.slot_duration_minutes = slot_duration_minutes
        self._slot_duration_hours = slot_duration_minutes / 60.0

    # -- Public API ----------------------------------------------------------

    def schedule_fleet(
        self,
        fleet: EVFleet,
        prices: list[float] | None = None,
        time_horizon_hours: float = 24.0,
        use_optimiser: bool = False,
    ) -> V2GScheduleResult:
        """Generate a V2G schedule for the entire fleet."""
        t0 = time.time()
        windows = fleet.get_flexibility_windows()

        if not windows:
            return V2GScheduleResult(method="no_flexibility")

        num_slots = int(time_horizon_hours / self._slot_duration_hours)
        if prices is None:
            prices = [0.10] * num_slots  # flat default

        # Ensure prices list matches horizon
        prices = (prices + [prices[-1]] * num_slots)[:num_slots]

        if use_optimiser:
            try:
                result = self._optimised_schedule(fleet, windows, prices, num_slots)
            except Exception:
                logger.warning("Optimiser failed, falling back to rule-based")
                result = self._rule_based_schedule(fleet, windows, prices, num_slots)
        else:
            result = self._rule_based_schedule(fleet, windows, prices, num_slots)

        result.solve_time_ms = (time.time() - t0) * 1000
        return result

    def schedule_single(
        self,
        ev: EVBattery,
        prices: list[float] | None = None,
        time_horizon_hours: float = 24.0,
    ) -> V2GScheduleResult:
        """Generate a schedule for a single EV."""
        fleet = EVFleet()
        fleet.add_vehicle(ev)
        return self.schedule_fleet(fleet, prices, time_horizon_hours)

    # -- Rule-based scheduler ------------------------------------------------

    def _rule_based_schedule(
        self,
        fleet: EVFleet,
        windows: list[FlexibilityWindow],
        prices: list[float],
        num_slots: int,
    ) -> V2GScheduleResult:
        """Simple rule-based scheduling.

        Strategy:
        1. Sort time slots by price.
        2. Assign charging to cheapest slots (must reach target SOC).
        3. Assign V2G discharge to most expensive slots (if flexible).
        """
        now = time.time()
        slot_hours = self._slot_duration_hours
        schedule: list[ScheduleSlot] = []
        total_cost = 0.0
        total_revenue = 0.0

        for window in windows:
            ev = fleet.get_vehicle(window.ev_id)
            if ev is None:
                continue

            # Number of slots in this EV's window
            window_slots = min(
                num_slots,
                int(window.duration_hours / slot_hours),
            )
            if window_slots <= 0:
                continue

            # Price-sorted slot indices (cheapest first)
            slot_indices = list(range(window_slots))
            slot_indices.sort(key=lambda i: prices[i] if i < len(prices) else 0)

            energy_needed = window.needed_energy_kwh
            energy_available = window.available_energy_kwh
            charged = 0.0
            discharged = 0.0

            # Phase 1: Charge in cheapest slots to meet target
            for idx in slot_indices:
                if charged >= energy_needed:
                    break
                slot_energy = min(
                    ev.max_charge_kw * slot_hours * ev.charge_efficiency,
                    energy_needed - charged,
                )
                if slot_energy <= 0:
                    continue

                power = slot_energy / (slot_hours * ev.charge_efficiency)
                slot_start = now + idx * slot_hours * 3600
                schedule.append(ScheduleSlot(
                    start_time=slot_start,
                    end_time=slot_start + slot_hours * 3600,
                    power_kw=power,
                    ev_id=ev.ev_id,
                    price=prices[idx] if idx < len(prices) else 0,
                ))
                cost = slot_energy * (prices[idx] if idx < len(prices) else 0)
                total_cost += cost
                charged += slot_energy

            # Phase 2: Discharge in most expensive slots (reverse order)
            if ev.v2g_capable and ev.has_flexibility:
                expensive_first = sorted(
                    range(window_slots),
                    key=lambda i: prices[i] if i < len(prices) else 0,
                    reverse=True,
                )
                for idx in expensive_first:
                    if discharged >= energy_available * 0.5:  # conservative: use max 50%
                        break
                    price = prices[idx] if idx < len(prices) else 0
                    # Only discharge if price > charge cost margin
                    avg_charge_price = total_cost / max(charged, 0.01)
                    if price <= avg_charge_price * 1.3:  # need 30% margin
                        continue

                    slot_energy = min(
                        ev.max_discharge_kw * slot_hours * ev.discharge_efficiency,
                        energy_available * 0.5 - discharged,
                    )
                    if slot_energy <= 0:
                        continue

                    power = -slot_energy / (slot_hours * ev.discharge_efficiency)
                    slot_start = now + idx * slot_hours * 3600
                    schedule.append(ScheduleSlot(
                        start_time=slot_start,
                        end_time=slot_start + slot_hours * 3600,
                        power_kw=power,
                        ev_id=ev.ev_id,
                        price=price,
                    ))
                    total_revenue += slot_energy * price
                    discharged += slot_energy

        schedule.sort(key=lambda s: (s.ev_id, s.start_time))

        return V2GScheduleResult(
            schedule=schedule,
            total_cost=total_cost,
            total_revenue=total_revenue,
            method="rule_based",
        )

    # -- Optimised scheduler (LP) --------------------------------------------

    def _optimised_schedule(
        self,
        fleet: EVFleet,
        windows: list[FlexibilityWindow],
        prices: list[float],
        num_slots: int,
    ) -> V2GScheduleResult:
        """LP-based V2G scheduling using PuLP.

        Decision variables: charge[ev, t] and discharge[ev, t] for each EV
        and time slot.

        Objective: minimise(cost) = sum over t of
            price[t] * (charge[ev,t] - discharge[ev,t]) * dt
            + degradation_cost * discharge[ev,t] * dt
        """
        import pulp

        now = time.time()
        slot_hours = self._slot_duration_hours
        prob = pulp.LpProblem("V2G_Schedule", pulp.LpMinimize)

        ev_ids = [w.ev_id for w in windows]
        evs = {w.ev_id: fleet.get_vehicle(w.ev_id) for w in windows}

        # Decision variables
        charge = {}
        discharge = {}
        for eid in ev_ids:
            ev = evs[eid]
            if ev is None:
                continue
            for t in range(num_slots):
                charge[eid, t] = pulp.LpVariable(f"chg_{eid}_{t}", 0, ev.max_charge_kw)
                discharge[eid, t] = pulp.LpVariable(f"dis_{eid}_{t}", 0, ev.max_discharge_kw if ev.v2g_capable else 0)

        # Objective
        obj_terms = []
        for eid in ev_ids:
            ev = evs[eid]
            if ev is None:
                continue
            deg = ev.degradation_cost_per_kwh
            for t in range(num_slots):
                p = prices[t] if t < len(prices) else 0
                obj_terms.append(p * charge[eid, t] * slot_hours)
                obj_terms.append(-p * discharge[eid, t] * slot_hours)
                obj_terms.append(deg * discharge[eid, t] * slot_hours)
        prob += pulp.lpSum(obj_terms)

        # Constraints per EV
        for w in windows:
            ev = evs[w.ev_id]
            if ev is None:
                continue
            eid = w.ev_id
            window_slots = min(num_slots, int(w.duration_hours / slot_hours))

            # SOC must reach target at departure
            net_energy = pulp.lpSum(
                (charge[eid, t] * ev.charge_efficiency - discharge[eid, t] / ev.discharge_efficiency) * slot_hours
                for t in range(window_slots)
            )
            prob += net_energy >= w.needed_energy_kwh, f"target_soc_{eid}"

            # SOC never below min at any point
            for t_end in range(1, window_slots + 1):
                cumulative = pulp.lpSum(
                    (charge[eid, t] * ev.charge_efficiency - discharge[eid, t] / ev.discharge_efficiency) * slot_hours
                    for t in range(t_end)
                )
                min_deficit = (ev.min_soc - ev.current_soc) * ev.capacity_kwh
                prob += cumulative >= min_deficit, f"min_soc_{eid}_{t_end}"

            # SOC never above 1.0
            for t_end in range(1, window_slots + 1):
                cumulative = pulp.lpSum(
                    (charge[eid, t] * ev.charge_efficiency - discharge[eid, t] / ev.discharge_efficiency) * slot_hours
                    for t in range(t_end)
                )
                max_headroom = (1.0 - ev.current_soc) * ev.capacity_kwh
                prob += cumulative <= max_headroom, f"max_soc_{eid}_{t_end}"

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.constants.LpStatusOptimal:
            raise RuntimeError(f"LP infeasible or unbounded (status={prob.status})")

        # Extract schedule
        schedule: list[ScheduleSlot] = []
        total_cost = 0.0
        total_revenue = 0.0

        for eid in ev_ids:
            ev = evs[eid]
            if ev is None:
                continue
            w = next(ww for ww in windows if ww.ev_id == eid)
            window_slots = min(num_slots, int(w.duration_hours / slot_hours))

            for t in range(window_slots):
                chg = charge[eid, t].varValue or 0.0
                dis = discharge[eid, t].varValue or 0.0

                if chg > 0.01:
                    slot_start = now + t * slot_hours * 3600
                    p = prices[t] if t < len(prices) else 0
                    schedule.append(ScheduleSlot(
                        start_time=slot_start,
                        end_time=slot_start + slot_hours * 3600,
                        power_kw=chg,
                        ev_id=eid,
                        price=p,
                    ))
                    total_cost += chg * slot_hours * p

                if dis > 0.01:
                    slot_start = now + t * slot_hours * 3600
                    p = prices[t] if t < len(prices) else 0
                    schedule.append(ScheduleSlot(
                        start_time=slot_start,
                        end_time=slot_start + slot_hours * 3600,
                        power_kw=-dis,
                        ev_id=eid,
                        price=p,
                    ))
                    total_revenue += dis * slot_hours * p

        schedule.sort(key=lambda s: (s.ev_id, s.start_time))

        return V2GScheduleResult(
            schedule=schedule,
            total_cost=total_cost,
            total_revenue=total_revenue,
            method="optimised_lp",
        )
