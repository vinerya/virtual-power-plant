"""Site aggregation: capacity, live power, SOC, alert counts and health."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.db.models import ResourceModel, SiteModel
from vpp.portal.telemetry import latest_soc

#: SOH below which a battery marks its site "yellow" (matches the console).
SOH_WARN = 0.85

AlertCountProvider = Callable[[AsyncSession, Sequence[str]], Awaitable[dict[str, int]]]
_alert_count_provider: AlertCountProvider | None = None


def register_alert_count_provider(provider: AlertCountProvider | None) -> None:
    """Plug in the alert store: ``provider(session, source_ids) -> {source_id: n_active}``.

    Sites report ``active_alerts`` as the sum over their resource ids (and
    the site id itself). Until an alert store registers a provider this is
    reported as 0 -- there is no alert persistence to count from.
    """
    global _alert_count_provider
    _alert_count_provider = provider


def resource_capacity_kwh(r: ResourceModel) -> float | None:
    """Energy capacity of a battery resource, if known (never guessed)."""
    if r.nominal_energy_kwh:
        return float(r.nominal_energy_kwh)
    try:
        cfg = json.loads(r.config_json) if r.config_json else {}
    except ValueError:
        cfg = {}
    cap = cfg.get("capacity_kwh")
    return float(cap) if isinstance(cap, (int, float)) and cap > 0 else None


def score_health(*, active_alerts: int, soh_low: bool, offline: int) -> str:
    """Same thresholds as the console's client-side fallback (``web/lib/api/sites.ts``)."""
    if active_alerts >= 3 or offline >= 2:
        return "red"
    if active_alerts >= 1 or soh_low or offline >= 1:
        return "yellow"
    return "green"


async def summarize_sites(session: AsyncSession, sites: list[SiteModel]) -> list[dict[str, Any]]:
    if not sites:
        return []
    site_ids = [s.id for s in sites]
    resources = list(
        (
            await session.execute(
                select(ResourceModel)
                .where(ResourceModel.site_id.in_(site_ids))
                .order_by(ResourceModel.name)
            )
        ).scalars().all()
    )
    by_site: dict[str, list[ResourceModel]] = {sid: [] for sid in site_ids}
    for r in resources:
        by_site[r.site_id].append(r)

    socs = await latest_soc(session, [r.id for r in resources if r.resource_type == "battery"])

    alert_counts: dict[str, int] = {}
    if _alert_count_provider is not None:
        alert_counts = await _alert_count_provider(
            session, [r.id for r in resources] + site_ids
        )

    out: list[dict[str, Any]] = []
    for site in sites:
        members = by_site[site.id]
        online = sum(1 for r in members if r.online)
        cap_total = 0.0
        soc_weighted = 0.0
        soc_weight = 0.0
        for r in members:
            cap = resource_capacity_kwh(r) if r.resource_type == "battery" else None
            if cap is not None:
                cap_total += cap
            if r.id in socs:
                w = cap if cap is not None else 1.0
                soc_weighted += socs[r.id] * w
                soc_weight += w
        active = alert_counts.get(site.id, 0) + sum(alert_counts.get(r.id, 0) for r in members)
        soh_low = any(
            r.resource_type == "battery"
            and r.state_of_health is not None
            and r.state_of_health < SOH_WARN
            for r in members
        )
        out.append({
            "id": site.id,
            "name": site.name,
            "lat": site.lat,
            "lon": site.lon,
            "region": site.region,
            "address": site.address,
            "timezone": site.timezone or "UTC",
            "owner_id": site.owner_id,
            "resource_ids": [r.id for r in members],
            "total_resources": len(members),
            "online_count": online,
            "current_power": sum(r.current_power or 0.0 for r in members),
            "rated_power": sum(r.rated_power or 0.0 for r in members),
            "capacity_kwh": cap_total,
            "state_of_charge": (soc_weighted / soc_weight) if soc_weight else None,
            "active_alerts": active,
            "health": score_health(
                active_alerts=active, soh_low=soh_low, offline=len(members) - online
            ),
            "metadata": json.loads(site.metadata_json) if site.metadata_json else {},
            "created_at": site.created_at,
            "updated_at": site.updated_at,
        })
    return out
