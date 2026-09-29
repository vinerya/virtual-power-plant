"""Data retention: prune old rows from tables that otherwise grow forever.

Each :class:`RetentionRule` names a table, the ``VPP_*_RETENTION_DAYS``
setting that bounds it (``0`` = keep forever) and which rows count as old.
:func:`prune` applies every enabled rule once; the API runs it periodically
under the ``data-retention`` lease (so exactly one worker prunes, see
:mod:`vpp.cluster.lease`), and ``vpp prune`` runs it once from the CLI.

Rows are deleted in batches of ``VPP_RETENTION_BATCH_SIZE``, each in its own
transaction, so a large backlog never holds a long lock. Pruning is logged
(``vpp.retention``), not written to the audit log.

Not pruned here:

* ``orders`` / ``trades``: the trading venue rebuilds positions and P&L by
  replaying every trade, so deleting history would change them.
* ``meter_readings``, ``v2g_charging_sessions``, ``battery_soh_samples``,
  ``optimization_runs``, ``v2g_schedules``, ``v2g_flexibility_bids``:
  business / billing records or low-volume history.
* ``cluster_calls``, ``cluster_events``, ``shared_rate_limits``: short-lived
  coordination rows already purged by their owners (:mod:`vpp.cluster.rpc`,
  :mod:`vpp.cluster.relay`, :mod:`vpp.auth.shared_limits`).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, func, select

from vpp.cluster.lease import TaskRole
from vpp.cluster.node import utcnow
from vpp.db.models import (
    AlertModel,
    AuditLogModel,
    BatteryStateModel,
    DREventResponseModel,
    EventLogModel,
    ResourceTelemetryModel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy import ColumnElement
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

#: Default pause after startup before the first pass, so pruning does not
#: compete with the rest of startup.
INITIAL_DELAY_S = 60.0


@dataclass(frozen=True)
class RetentionRule:
    """One prunable table: rows matching ``older_than(cutoff)`` are deleted."""

    table: str
    setting: str  # Settings attribute holding the retention in days
    model: Any
    older_than: Callable[[datetime], ColumnElement[bool]]

    @property
    def env_var(self) -> str:
        return f"VPP_{self.setting.upper()}"


RULES: tuple[RetentionRule, ...] = (
    RetentionRule(
        "audit_log", "audit_retention_days", AuditLogModel, lambda c: AuditLogModel.ts < c
    ),
    RetentionRule(
        "event_log",
        "event_log_retention_days",
        EventLogModel,
        lambda c: EventLogModel.created_at < c,
    ),
    # Only resolved alerts; open, acknowledged and snoozed ones are kept.
    RetentionRule(
        "alerts",
        "alert_retention_days",
        AlertModel,
        lambda c: and_(
            AlertModel.status == "resolved",
            func.coalesce(AlertModel.resolved_at, AlertModel.last_fired_at) < c,
        ),
    ),
    RetentionRule(
        "dr_event_responses",
        "dr_response_retention_days",
        DREventResponseModel,
        lambda c: DREventResponseModel.created_at < c,
    ),
    RetentionRule(
        "battery_states",
        "telemetry_retention_days",
        BatteryStateModel,
        lambda c: BatteryStateModel.timestamp < c,
    ),
    RetentionRule(
        "resource_telemetry",
        "telemetry_retention_days",
        ResourceTelemetryModel,
        lambda c: ResourceTelemetryModel.timestamp < c,
    ),
)


@dataclass
class PruneResult:
    """Outcome of one rule: ``rows`` deleted (or, in a dry run, that would be)."""

    table: str
    env_var: str
    retention_days: int
    cutoff: datetime | None
    rows: int = 0

    @property
    def enabled(self) -> bool:
        return self.retention_days > 0


def retention_enabled(settings: Any) -> bool:
    """Whether any rule has a finite retention (else there is nothing to schedule)."""
    return any(int(getattr(settings, r.setting, 0) or 0) > 0 for r in RULES)


async def _count(
    factory: async_sessionmaker[AsyncSession], rule: RetentionRule, cutoff: datetime
) -> int:
    async with factory() as session:
        stmt = select(func.count()).select_from(rule.model).where(rule.older_than(cutoff))
        return int((await session.execute(stmt)).scalar_one())


async def _delete_batch(
    factory: async_sessionmaker[AsyncSession],
    rule: RetentionRule,
    cutoff: datetime,
    batch_size: int,
) -> int:
    """Delete at most *batch_size* old rows in one short transaction."""
    pk = rule.model.id
    ids = select(pk).where(rule.older_than(cutoff)).limit(batch_size).scalar_subquery()
    async with factory() as session:
        result = await session.execute(
            delete(rule.model).where(pk.in_(ids)).execution_options(synchronize_session=False)
        )
        await session.commit()
    return int(getattr(result, "rowcount", 0) or 0)


async def prune(
    factory: async_sessionmaker[AsyncSession],
    settings: Any,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    batch_size: int | None = None,
) -> list[PruneResult]:
    """Apply every retention rule once; one :class:`PruneResult` per rule.

    With ``dry_run`` rows are only counted. A rule whose retention is ``0``
    is skipped (reported with ``cutoff=None``).
    """
    now = now or utcnow()
    if batch_size is None:
        batch_size = int(getattr(settings, "retention_batch_size", 1000))
    size = max(1, batch_size)
    results: list[PruneResult] = []
    for rule in RULES:
        days = int(getattr(settings, rule.setting, 0) or 0)
        if days <= 0:
            results.append(PruneResult(rule.table, rule.env_var, days, None))
            continue
        cutoff = now - timedelta(days=days)
        res = PruneResult(rule.table, rule.env_var, days, cutoff)
        if dry_run:
            res.rows = await _count(factory, rule, cutoff)
        else:
            while True:
                deleted = await _delete_batch(factory, rule, cutoff, size)
                res.rows += deleted
                if deleted < size:
                    break
                await asyncio.sleep(0)  # let other work in this process run between batches
            if res.rows:
                logger.info(
                    "Retention: deleted %d row(s) from %s older than %s (%s=%d)",
                    res.rows,
                    rule.table,
                    cutoff.isoformat(),
                    rule.env_var,
                    days,
                )
        results.append(res)
    if not dry_run:
        total = sum(r.rows for r in results)
        logger.log(
            logging.INFO if total else logging.DEBUG,
            "Retention pass finished: %d row(s) deleted%s",
            total,
            "".join(f", {r.table}={r.rows}" for r in results if r.rows),
        )
    return results


async def run_retention_loop(
    factory: async_sessionmaker[AsyncSession],
    settings: Any,
    *,
    initial_delay_s: float = INITIAL_DELAY_S,
) -> None:
    """Forever: prune, then sleep ``VPP_RETENTION_INTERVAL_MINUTES``. Errors are logged."""
    interval_s = max(1, int(getattr(settings, "retention_interval_minutes", 60))) * 60
    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            await prune(factory, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Retention pass failed; retrying in %ds", interval_s)
        await asyncio.sleep(interval_s)


def retention_role(
    factory: async_sessionmaker[AsyncSession],
    settings: Any,
    *,
    initial_delay_s: float = INITIAL_DELAY_S,
) -> TaskRole:
    """Leader role for the ``data-retention`` lease."""
    return TaskRole(
        lambda: run_retention_loop(factory, settings, initial_delay_s=initial_delay_s),
        name="vpp-data-retention",
    )
