"""Pure translation of grid signals into fleet dispatch directives.

Everything here is side-effect free so the rules can be read (and tested)
in one place. Sign convention: ``target_kw`` is **export-positive** for the
whole fleet, the same as ``POST /api/v1/optimization/dispatch`` -- positive
means "deliver power to the grid / reduce net load", negative means "absorb".

OpenADR 2.0b (VEN), using the *current* interval's value of the event's
first signal (falling back to ``currentValue`` / the first interval):

=====================  ==================  =====================================
signalName             signalType          target_kw
=====================  ==================  =====================================
SIMPLE                 level               ``simple_level_fractions[level]`` x
                                           fleet export capability
LOAD_DISPATCH          delta               ``+value`` (shed / deliver value kW)
LOAD_DISPATCH          setpoint / level    ``-value`` (value = VEN net-load
                                           setpoint in kW; negative = export)
LOAD_CONTROL           x-loadControl-      ``-value`` (net-load setpoint, kW)
                       Setpoint
LOAD_CONTROL           other               ``+value`` (offset, kW)
LOAD_PERCENTAGE,       --                  not translated (needs a metered
ELECTRICITY_PRICE                          baseline / belongs to the price-
                                           driven MPC); recorded only
=====================  ==================  =====================================

OpenADR 2.0b carries no unit on the event signal itself, so values are
taken as kW (the common VTN configuration for these signals).

IEEE 2030.5 active DERControls, highest-priority program first; for each
field the first control that sets it wins:

* ``opModConnect=false`` / ``opModEnergize=false`` -> target 0 kW
* ``opModTargetW`` -> target = W / 1000 (positive = discharge/export)
* ``opModFixedW`` -> target = pct/100 x setMaxW
* ``opModMaxLimW`` -> export cap = pct/100 x setMaxW
* ``opModGenLimW`` -> export cap = W / 1000
* ``opModLoadLimW`` -> absorb cap = W / 1000

``setMaxW`` is ``dr_ieee2030_5_set_max_w`` when configured, else the fleet's
current export capability. When both protocols are active, an IEEE 2030.5
target wins over an OpenADR one, and IEEE 2030.5 limits always clamp the
result (they are utility grid-safety constraints). Operator caps
(``dr_max_export_kw`` / ``dr_max_import_kw``) are applied last.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_OPENADR = ("SIMPLE", "LOAD_DISPATCH", "LOAD_CONTROL")


@dataclass(frozen=True)
class DRPolicy:
    """Operator-configured rules for automatic DR response."""

    auto_response: bool = False
    auto_opt_in: bool = True
    tick_interval_s: float = 5.0
    redispatch_interval_s: float = 900.0
    max_export_kw: float | None = None
    max_import_kw: float | None = None
    simple_level_fractions: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)
    min_opt_in_fraction: float = 0.5
    include_v2g: bool = True
    set_max_w: float | None = None
    max_profile_periods: int = 48

    @classmethod
    def from_settings(cls, settings: Any) -> DRPolicy:
        return cls(
            auto_response=bool(getattr(settings, "dr_auto_response_enabled", False)),
            auto_opt_in=bool(getattr(settings, "openadr_auto_opt_in", True)),
            tick_interval_s=float(getattr(settings, "dr_tick_interval_s", 5.0)),
            redispatch_interval_s=float(getattr(settings, "dr_redispatch_interval_s", 900.0)),
            max_export_kw=getattr(settings, "dr_max_export_kw", None),
            max_import_kw=getattr(settings, "dr_max_import_kw", None),
            simple_level_fractions=tuple(
                float(f)
                for f in getattr(settings, "dr_simple_level_fractions", (0.0, 0.5, 0.75, 1.0))
            )
            or (0.0,),
            min_opt_in_fraction=float(getattr(settings, "dr_min_opt_in_fraction", 0.5)),
            include_v2g=bool(getattr(settings, "dr_include_v2g", True)),
            set_max_w=getattr(settings, "dr_ieee2030_5_set_max_w", None),
            max_profile_periods=int(getattr(settings, "v2g_max_profile_periods", 48)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_response": self.auto_response,
            "auto_opt_in": self.auto_opt_in,
            "tick_interval_s": self.tick_interval_s,
            "redispatch_interval_s": self.redispatch_interval_s,
            "max_export_kw": self.max_export_kw,
            "max_import_kw": self.max_import_kw,
            "simple_level_fractions": list(self.simple_level_fractions),
            "min_opt_in_fraction": self.min_opt_in_fraction,
            "include_v2g": self.include_v2g,
            "set_max_w": self.set_max_w,
        }


@dataclass(frozen=True)
class FleetCapability:
    """What the fleet can sustain over a window (kW, both non-negative)."""

    export_kw: float = 0.0
    import_kw: float = 0.0
    resources: int = 0
    vehicles: int = 0


@dataclass
class DRDirective:
    """A dispatch target (and/or limits) derived from grid signals."""

    protocol: str
    source_id: str
    revision: int = 0
    target_kw: float | None = None
    max_export_kw: float | None = None
    max_import_kw: float | None = None
    start: float = 0.0
    end: float = 0.0
    signal_type: str | None = None
    signal_level: float | None = None
    supported: bool = True
    reasons: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)

    def key(self) -> tuple[Any, ...]:
        """Identity for change detection (re-dispatch when it changes)."""
        target = None if self.target_kw is None else round(self.target_kw, 1)
        return (self.protocol, self.source_id, self.revision, target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "source_id": self.source_id,
            "revision": self.revision,
            "target_kw": self.target_kw,
            "max_export_kw": self.max_export_kw,
            "max_import_kw": self.max_import_kw,
            "start": self.start,
            "end": self.end,
            "signal_type": self.signal_type,
            "signal_level": self.signal_level,
            "supported": self.supported,
            "reason": self.reason,
            "sources": self.sources,
        }


# ---------------------------------------------------------------------------
# OpenADR
# ---------------------------------------------------------------------------


def current_signal_value(event: Any, now: float) -> float:
    """Value of the event's first signal in the interval containing *now*."""
    signals = (event.metadata or {}).get("signals") or []
    if signals:
        intervals = signals[0].get("intervals") or []
        for interval in intervals:
            start = float(interval.get("start_time") or 0.0)
            duration = float(interval.get("duration_seconds") or 0.0)
            value = interval.get("value")
            if value is not None and start <= now < start + duration:
                return float(value)
        current = signals[0].get("current_value")
        if current is not None:
            return float(current)
    return float(event.signal_level or 0.0)


def translate_openadr(
    event: Any, cap: FleetCapability, policy: DRPolicy, *, now: float | None = None
) -> DRDirective:
    """Translate one OpenADR :class:`~vpp.protocols.openadr.DREvent`."""
    at = time.time() if now is None else now
    signal_name = getattr(event.signal_type, "value", str(event.signal_type))
    signals = (event.metadata or {}).get("signals") or []
    raw_type = str(signals[0].get("signal_type") or "") if signals else ""
    level = current_signal_value(event, max(at, float(event.start_time)))
    directive = DRDirective(
        protocol="openadr",
        source_id=str(event.event_id),
        revision=int((event.metadata or {}).get("modification_number") or 0),
        start=float(event.start_time),
        end=float(event.end_time),
        signal_type=f"{signal_name}/{raw_type}" if raw_type else signal_name,
        signal_level=level,
        sources=[{"protocol": "openadr", "id": str(event.event_id)}],
    )
    if signal_name == "SIMPLE":
        fractions = policy.simple_level_fractions
        idx = max(0, min(len(fractions) - 1, round(level)))
        directive.target_kw = fractions[idx] * cap.export_kw
        directive.reasons.append(
            f"SIMPLE level {level:g} -> {fractions[idx]:.0%} of fleet export capability "
            f"({cap.export_kw:.1f} kW)"
        )
    elif signal_name == "LOAD_DISPATCH":
        if raw_type == "delta":
            directive.target_kw = level
            directive.reasons.append(f"LOAD_DISPATCH delta: shed {level:g} kW")
        else:
            directive.target_kw = -level
            directive.reasons.append(f"LOAD_DISPATCH setpoint: net load {level:g} kW")
    elif signal_name == "LOAD_CONTROL":
        if raw_type.lower() == "x-loadcontrolsetpoint":
            directive.target_kw = -level
            directive.reasons.append(f"LOAD_CONTROL setpoint: net load {level:g} kW")
        else:
            directive.target_kw = level
            directive.reasons.append(f"LOAD_CONTROL offset: shed {level:g} kW")
    else:
        directive.supported = False
        directive.reasons.append(
            f"{signal_name} signals are not translated into a dispatch target"
        )
    return directive


# ---------------------------------------------------------------------------
# IEEE 2030.5
# ---------------------------------------------------------------------------


def translate_ieee2030_5(
    controls: list[Any], cap: FleetCapability, policy: DRPolicy
) -> DRDirective | None:
    """Merge active :class:`~vpp.protocols.ieee2030_5.DERControl` objects."""
    if not controls:
        return None
    set_max_kw = policy.set_max_w / 1000.0 if policy.set_max_w else cap.export_kw
    d = DRDirective(
        protocol="ieee2030_5",
        source_id=",".join(str(c.control_id) for c in controls)[:255],
        revision=int(max((c.creation_time or 0.0) for c in controls)),
        start=min(float(c.start_time or 0.0) for c in controls),
        end=min(
            (float(c.end_time) for c in controls if c.start_time and c.duration_seconds),
            default=0.0,
        ),
        sources=[
            {"protocol": "ieee2030_5", "id": str(c.control_id), "program_id": c.program_id}
            for c in controls
        ],
    )
    for c in controls:
        if d.target_kw is None:
            if c.connect is False or c.energize is False:
                d.target_kw = 0.0
                d.signal_type = "opModConnect/Energize"
                d.reasons.append(f"{c.control_id}: cease to energize -> 0 kW")
            elif c.set_watts is not None:
                d.target_kw = float(c.set_watts) / 1000.0
                d.signal_type, d.signal_level = "opModTargetW", float(c.set_watts)
                d.reasons.append(f"{c.control_id}: opModTargetW {c.set_watts:g} W")
            elif c.fixed_w_pct is not None:
                d.target_kw = float(c.fixed_w_pct) / 100.0 * set_max_kw
                d.signal_type, d.signal_level = "opModFixedW", float(c.fixed_w_pct)
                d.reasons.append(
                    f"{c.control_id}: opModFixedW {c.fixed_w_pct:g}% of {set_max_kw:.1f} kW"
                )
        if c.max_limit_pct is not None:
            limit = float(c.max_limit_pct) / 100.0 * set_max_kw
            if d.max_export_kw is None:
                d.max_export_kw = limit
                d.reasons.append(
                    f"{c.control_id}: opModMaxLimW {c.max_limit_pct:g}% -> export <= "
                    f"{limit:.1f} kW"
                )
        if c.gen_limit_w is not None:
            limit = float(c.gen_limit_w) / 1000.0
            if d.max_export_kw is None or limit < d.max_export_kw:
                d.max_export_kw = limit
                d.reasons.append(f"{c.control_id}: opModGenLimW -> export <= {limit:.1f} kW")
        if c.load_limit_w is not None and d.max_import_kw is None:
            d.max_import_kw = float(c.load_limit_w) / 1000.0
            d.reasons.append(
                f"{c.control_id}: opModLoadLimW -> absorb <= {d.max_import_kw:.1f} kW"
            )
    if d.target_kw is None and d.max_export_kw is None and d.max_import_kw is None:
        d.supported = False
        d.reasons.append("active controls carry no active-power target or limit")
    return d


# ---------------------------------------------------------------------------
# Combination + safety limits
# ---------------------------------------------------------------------------


def combine(
    ieee: DRDirective | None, openadr: DRDirective | None, policy: DRPolicy
) -> DRDirective | None:
    """One effective directive: IEEE 2030.5 target > OpenADR target; all caps applied."""
    base: DRDirective | None = None
    if ieee is not None and ieee.target_kw is not None:
        base = ieee
    elif openadr is not None and openadr.target_kw is not None:
        base = openadr
        if ieee is not None:
            base.sources = base.sources + ieee.sources
    if base is None:
        return None

    target = float(base.target_kw or 0.0)
    limits = ieee if ieee is not None else base
    if limits.max_export_kw is not None and target > limits.max_export_kw:
        base.reasons.append(f"clamped to IEEE 2030.5 export limit {limits.max_export_kw:.1f} kW")
        target = limits.max_export_kw
    if limits.max_import_kw is not None and target < -limits.max_import_kw:
        base.reasons.append(f"clamped to IEEE 2030.5 absorb limit {limits.max_import_kw:.1f} kW")
        target = -limits.max_import_kw
    if policy.max_export_kw is not None and target > policy.max_export_kw:
        base.reasons.append(f"clamped to operator cap dr_max_export_kw={policy.max_export_kw:g}")
        target = float(policy.max_export_kw)
    if policy.max_import_kw is not None and target < -policy.max_import_kw:
        base.reasons.append(f"clamped to operator cap dr_max_import_kw={policy.max_import_kw:g}")
        target = -float(policy.max_import_kw)
    base.target_kw = target
    return base


def opt_decision(
    directive: DRDirective, cap: FleetCapability, policy: DRPolicy, *, test_event: bool = False
) -> tuple[str, str, str]:
    """Return ``(opt_type, action, reason)`` for a newly received OpenADR event."""
    fallback = "optIn" if policy.auto_opt_in else "optOut"
    if not policy.auto_response:
        return (
            fallback,
            "observed",
            "auto-response disabled (VPP_DR_AUTO_RESPONSE_ENABLED=false); "
            f"{fallback} per VPP_OPENADR_AUTO_OPT_IN; operator action required",
        )
    if not directive.supported:
        return fallback, "unsupported", directive.reason
    if test_event:
        return fallback, "test_event", "test event: acknowledged, never dispatched"
    requested = abs(float(directive.target_kw or 0.0))
    if requested <= 1e-6:
        return "optIn", "accepted", "event requests no power change"
    available = cap.export_kw if (directive.target_kw or 0.0) > 0 else cap.import_kw
    if policy.max_export_kw is not None and (directive.target_kw or 0.0) > 0:
        available = min(available, float(policy.max_export_kw))
    if policy.max_import_kw is not None and (directive.target_kw or 0.0) < 0:
        available = min(available, float(policy.max_import_kw))
    needed = policy.min_opt_in_fraction * requested
    if available > 0 and available + 1e-9 >= needed:
        return (
            "optIn",
            "accepted",
            f"fleet can deliver {min(available, requested):.1f} of {requested:.1f} kW",
        )
    return (
        "optOut",
        "declined",
        f"fleet capability {available:.1f} kW < {policy.min_opt_in_fraction:.0%} of the "
        f"requested {requested:.1f} kW",
    )


__all__ = [
    "SUPPORTED_OPENADR",
    "DRDirective",
    "DRPolicy",
    "FleetCapability",
    "combine",
    "current_signal_value",
    "opt_decision",
    "translate_ieee2030_5",
    "translate_openadr",
]
