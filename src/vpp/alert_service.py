"""Persistent alerting: rules + fired alerts in the DB, evaluated on telemetry.

Wiring (done by the API lifespan, see :func:`vpp.api.app._lifespan`)::

    EventBus --RESOURCE_UPDATED--> AlertService (bounded queue + worker)
        -> AlertManager.check(metric, value, source=resource_id)
        -> persist / de-duplicate in the ``alerts`` table
        -> AlertManager.dispatch (log + optional signed webhook)
        -> WebSocket broadcast on the ``alerts`` channel
        -> ``vpp_alerts_fired_total`` metric

Design notes
------------
* Every numeric key of a ``RESOURCE_UPDATED`` event's ``data`` is a metric
  (``soc``, ``soh``, ``temperature``, ``current_power_kw``, ...).  ``soc``
  and ``soh`` are normalised to 0-1 (percentages are divided by 100).
* Rule state (cooldown, rate/anomaly history) is kept per ``(rule,
  resource)`` in memory by :class:`~vpp.alerts.AlertManager`; it resets on
  restart, which at worst re-fires a still-true condition once -- and that
  is absorbed by de-duplication below.
* De-duplication: while an alert for the same ``(rule, resource)`` is open
  (not resolved), re-triggers bump ``occurrences``/``value`` on the
  existing row instead of creating a new alert, and are not re-broadcast.
* Threshold rules with ``auto_resolve`` resolve their open alerts once a
  value is back in range.
* Only *newly fired* alerts are pushed on the ``alerts`` WebSocket channel
  (the console toasts every message there); lifecycle changes
  (ack / snooze / resolve) are picked up by the console's polling.
* Evaluation runs off the publisher's path: the EventBus callback only
  enqueues; a single worker task does DB work.  When the queue is full new
  telemetry is dropped for alerting (logged) rather than stalling ingestion.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import and_, or_, select

from vpp.alerts import Alert, AlertManager, AlertRule, AlertSeverity, RuleType
from vpp.db.models import AlertModel, AlertRuleModel, ResourceModel
from vpp.events.bus import Event, EventBus, EventType
from vpp.metrics import normalise_soc, record_alert_fired

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

ALERTS_CHANNEL = "alerts"
FRACTION_METRICS = frozenset({"soc", "soh"})
OPEN_STATUSES = ("active", "acknowledged", "snoozed")

DEFAULT_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "Battery SOC low",
        "description": "Battery state of charge below 10%.",
        "metric": "soc", "comparison": "<", "threshold": 0.10,
        "severity": "warning", "cooldown_s": 900.0,
    },
    {
        "name": "Battery over-temperature",
        "description": "Battery temperature above 50 degC.",
        "metric": "temperature", "comparison": ">", "threshold": 50.0,
        "severity": "critical", "cooldown_s": 300.0,
    },
    {
        "name": "Battery state of health degraded",
        "description": "Battery state of health below 80% (end-of-life threshold).",
        "metric": "soh", "comparison": "<", "threshold": 0.80,
        "severity": "warning", "cooldown_s": 86400.0,
    },
)


class Broadcaster(Protocol):
    async def broadcast(self, channel: str, data: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything here is stored in UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def api_severity(severity: str) -> str:
    """Map engine severities onto the console's info/warning/critical."""
    return "critical" if severity in ("error", "critical") else (
        severity if severity in ("info", "warning") else "warning"
    )


def effective_status(row: AlertModel, now: datetime | None = None) -> str:
    if row.status == "snoozed":
        until = as_utc(row.snoozed_until)
        if until is None or until <= (now or utcnow()):
            return "active"
    return row.status


def serialize_alert(row: AlertModel, now: datetime | None = None) -> dict[str, Any]:
    """Alert row -> ``AlertRead``-shaped dict (JSON-safe, ISO timestamps)."""
    def iso(dt: datetime | None) -> str | None:
        dt = as_utc(dt)
        return dt.isoformat() if dt else None

    status = effective_status(row, now)
    source_link = f"/assets/{row.source}" if row.source_kind == "resource" else None
    return {
        "id": row.id,
        "timestamp": iso(row.fired_at),
        "severity": api_severity(row.severity),
        "source": row.source,
        "source_kind": row.source_kind,
        "source_link": source_link,
        "title": row.title,
        "message": row.message,
        "status": status,
        "snoozed_until": iso(row.snoozed_until) if status == "snoozed" else None,
        "acknowledged_at": iso(row.acknowledged_at),
        "acknowledged_by": row.acknowledged_by,
        "resolved_at": iso(row.resolved_at),
        "rule_id": row.rule_id,
        "rule_name": row.rule_name,
        "metric": row.metric,
        "value": row.value,
        "threshold": row.threshold,
        "occurrences": row.occurrences,
        "last_fired_at": iso(row.last_fired_at),
    }


def rule_from_row(row: AlertRuleModel) -> AlertRule:
    """Build an engine rule keyed by the DB id (names are mutable)."""
    return AlertRule(
        name=row.id,
        rule_type=RuleType(row.rule_type),
        severity=AlertSeverity(row.severity),
        metric_name=row.metric,
        threshold=row.threshold,
        comparison=row.comparison,
        rate_window_s=row.rate_window_s,
        rate_threshold=row.rate_threshold,
        z_score_threshold=row.z_score_threshold,
        cooldown_s=row.cooldown_s,
        enabled=row.enabled,
        resource_id=row.resource_id,
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class AlertRepository:
    """DB access for alerts and alert rules."""

    @staticmethod
    def status_clause(status: str, now: datetime) -> Any:
        snooze_expired = or_(AlertModel.snoozed_until.is_(None), AlertModel.snoozed_until <= now)
        if status == "active":
            return or_(
                AlertModel.status == "active",
                and_(AlertModel.status == "snoozed", snooze_expired),
            )
        if status == "snoozed":
            return and_(AlertModel.status == "snoozed", AlertModel.snoozed_until > now)
        if status == "open":
            return AlertModel.status != "resolved"
        return AlertModel.status == status

    @staticmethod
    async def list_alerts(
        session: AsyncSession,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        severity: str | None = None,
        status: str | None = None,
        source: str | None = None,
        limit: int = 200,
        now: datetime | None = None,
    ) -> list[AlertModel]:
        now = now or utcnow()
        stmt = select(AlertModel)
        if since is not None:
            stmt = stmt.where(AlertModel.fired_at >= since)
        if until is not None:
            stmt = stmt.where(AlertModel.fired_at <= until)
        if severity:
            sev = ("critical", "error") if severity == "critical" else (severity,)
            stmt = stmt.where(AlertModel.severity.in_(sev))
        if status and status != "all":
            stmt = stmt.where(AlertRepository.status_clause(status, now))
        if source:
            stmt = stmt.where(AlertModel.source == source)
        stmt = stmt.order_by(AlertModel.fired_at.desc()).limit(limit)
        return list((await session.execute(stmt)).scalars().all())

    @staticmethod
    async def get(session: AsyncSession, alert_id: str) -> AlertModel | None:
        return await session.get(AlertModel, alert_id)

    @staticmethod
    async def find_open(
        session: AsyncSession, rule_id: str, source: str,
    ) -> list[AlertModel]:
        stmt = select(AlertModel).where(
            AlertModel.rule_id == rule_id,
            AlertModel.source == source,
            AlertModel.status.in_(OPEN_STATUSES),
        )
        return list((await session.execute(stmt)).scalars().all())

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def acknowledge(row: AlertModel, by: str | None, now: datetime | None = None) -> None:
        if row.status == "resolved":
            return
        now = now or utcnow()
        row.status = "acknowledged"
        row.acknowledged_at = now
        row.acknowledged_by = by
        row.snoozed_until = None

    @staticmethod
    def snooze(row: AlertModel, until: datetime) -> None:
        if row.status == "resolved":
            return
        row.status = "snoozed"
        row.snoozed_until = until

    @staticmethod
    def resolve(row: AlertModel, by: str | None, now: datetime | None = None) -> None:
        if row.status == "resolved":
            return
        row.status = "resolved"
        row.resolved_at = now or utcnow()
        row.resolved_by = by
        row.snoozed_until = None

    # -- rules ---------------------------------------------------------------

    @staticmethod
    async def list_rules(session: AsyncSession) -> list[AlertRuleModel]:
        stmt = select(AlertRuleModel).order_by(AlertRuleModel.name)
        return list((await session.execute(stmt)).scalars().all())

    @staticmethod
    async def get_rule(session: AsyncSession, rule_id: str) -> AlertRuleModel | None:
        return await session.get(AlertRuleModel, rule_id)

    @staticmethod
    async def get_rule_by_name(session: AsyncSession, name: str) -> AlertRuleModel | None:
        stmt = select(AlertRuleModel).where(AlertRuleModel.name == name)
        return (await session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def create_rule(session: AsyncSession, **fields: Any) -> AlertRuleModel:
        row = AlertRuleModel(**fields)
        session.add(row)
        await session.flush()
        await session.refresh(row)
        return row

    @staticmethod
    async def seed_default_rules(session: AsyncSession) -> int:
        """Insert :data:`DEFAULT_RULES` if no rule exists yet. Returns count added."""
        existing = (await session.execute(select(AlertRuleModel.id).limit(1))).first()
        if existing is not None:
            return 0
        for spec in DEFAULT_RULES:
            session.add(AlertRuleModel(rule_type="threshold", **spec))
        await session.flush()
        return len(DEFAULT_RULES)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


class AlertService:
    """Evaluates persisted rules against telemetry and persists fired alerts."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        broadcaster: Broadcaster | None = None,
        manager: AlertManager | None = None,
        queue_size: int = 1000,
    ) -> None:
        self._session_factory = session_factory
        self._broadcaster = broadcaster
        self.manager = manager or AlertManager()
        self._rule_meta: dict[str, AlertRuleModel] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._worker: asyncio.Task | None = None
        self._bus: EventBus | None = None
        self._subscription_id: str | None = None
        self._metrics_of_interest: frozenset[str] = frozenset()
        self.dropped_events = 0

    # -- lifecycle -------------------------------------------------------------

    async def start(self, bus: EventBus, *, seed_defaults: bool = False) -> None:
        if seed_defaults:
            async with self._session_factory() as session:
                added = await AlertRepository.seed_default_rules(session)
                await session.commit()
            if added:
                logger.info("Seeded %d default alert rules", added)
        await self.reload_rules()
        self._bus = bus
        self._subscription_id = bus.subscribe(
            self._on_event, event_types={EventType.RESOURCE_UPDATED},
        )
        self._worker = asyncio.create_task(self._run(), name="vpp-alert-evaluator")

    async def stop(self) -> None:
        if self._bus is not None and self._subscription_id is not None:
            self._bus.unsubscribe(self._subscription_id)
        self._subscription_id = None
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def reload_rules(self) -> None:
        """(Re)load all rules from the DB into the in-memory manager.

        Per-(rule, resource) cooldown state is kept for rules whose
        definition did not change.
        """
        async with self._session_factory() as session:
            rows = await AlertRepository.list_rules(session)
        new_ids = {r.id for r in rows}
        for rule in self.manager.rules:
            if rule.name not in new_ids:
                self.manager.remove_rule(rule.name)
        for row in rows:
            engine_rule = rule_from_row(row)
            current = self.manager.get_rule(row.id)
            if current is None or _rule_signature(current) != _rule_signature(engine_rule):
                self.manager.add_rule(engine_rule)
        self._rule_meta = {r.id: r for r in rows}
        self._metrics_of_interest = frozenset(self.manager.metric_names())

    async def drain(self) -> None:
        """Wait until every queued event has been processed (tests, shutdown)."""
        await self._queue.join()

    # -- event intake ----------------------------------------------------------

    async def _on_event(self, event: Event) -> None:
        data = event.data or {}
        if not self._metrics_of_interest.intersection(data):
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_events += 1
            logger.warning(
                "Alert evaluation queue full; dropped telemetry for %s (total dropped=%d)",
                data.get("resource_id"), self.dropped_events,
            )

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self.process_event(event)
            except Exception:
                logger.exception("Alert evaluation failed for event %s", event.event_id)
            finally:
                self._queue.task_done()

    # -- evaluation ------------------------------------------------------------

    async def process_event(self, event: Event) -> list[dict[str, Any]]:
        """Evaluate one telemetry event. Returns newly fired alerts (serialised)."""
        data = event.data or {}
        resource_id = data.get("resource_id") or data.get("id")
        if not resource_id:
            return []
        resource_id = str(resource_id)

        fired: list[tuple[Alert, AlertModel]] = []
        async with self._session_factory() as session:
            for metric in self._metrics_of_interest:
                value = _numeric(data.get(metric))
                if value is None:
                    continue
                if metric in FRACTION_METRICS:
                    value = normalise_soc(value)

                for alert in self.manager.check(metric, value, source=resource_id):
                    row, is_new = await self._persist(session, alert, metric, resource_id)
                    if is_new:
                        fired.append((alert, row))

                await self._auto_resolve(session, metric, value, resource_id)
            await session.commit()
            payloads = [(alert, serialize_alert(row)) for alert, row in fired]

        out: list[dict[str, Any]] = []
        for alert, payload in payloads:
            record_alert_fired(payload["severity"], payload["rule_name"] or payload["rule_id"] or "")
            await self.manager.dispatch(alert)
            if self._broadcaster is not None:
                try:
                    await self._broadcaster.broadcast(ALERTS_CHANNEL, payload)
                except Exception:
                    logger.exception("Failed to broadcast alert %s", payload["id"])
            out.append(payload)
        return out

    async def _persist(
        self, session: AsyncSession, alert: Alert, metric: str, resource_id: str,
    ) -> tuple[AlertModel, bool]:
        rule_id = alert.rule_name  # engine rule name == DB rule id
        meta = self._rule_meta.get(rule_id)
        now = utcnow()
        open_rows = await AlertRepository.find_open(session, rule_id, resource_id)
        if open_rows:
            row = open_rows[0]
            row.occurrences = (row.occurrences or 1) + 1
            row.value = alert.value
            row.last_fired_at = now
            return row, False

        resource_name = None
        resource = await session.get(ResourceModel, resource_id)
        if resource is not None:
            resource_name = resource.name
        title = meta.name if meta is not None else rule_id
        where = f"{resource_name} ({resource_id})" if resource_name else resource_id
        row = AlertModel(
            id=alert.alert_id,
            rule_id=rule_id if meta is not None else None,
            rule_name=title,
            severity=alert.severity.value,
            status="active",
            title=title,
            message=f"{alert.message} on {where}",
            source=resource_id,
            source_kind="resource",
            metric=metric,
            value=alert.value,
            threshold=alert.threshold,
            occurrences=1,
            fired_at=now,
            last_fired_at=now,
            metadata_json=json.dumps(alert.metadata, default=str),
        )
        session.add(row)
        await session.flush()
        # Keep the engine-side alert consistent with what was persisted.
        alert.message = row.message
        alert.metadata["title"] = title
        return row, True

    async def _auto_resolve(
        self, session: AsyncSession, metric: str, value: float, resource_id: str,
    ) -> None:
        for rule in self.manager.matching_rules(metric, source=resource_id):
            meta = self._rule_meta.get(rule.name)
            if meta is None or not meta.auto_resolve or rule.rule_type != RuleType.THRESHOLD:
                continue
            if rule.condition_met(value):
                continue
            for row in await AlertRepository.find_open(session, rule.name, resource_id):
                AlertRepository.resolve(row, by="auto")


def _rule_signature(rule: AlertRule) -> tuple[Any, ...]:
    return (
        rule.rule_type, rule.severity, rule.metric_name, rule.threshold, rule.comparison,
        rule.rate_window_s, rule.rate_threshold, rule.z_score_threshold, rule.cooldown_s,
        rule.enabled, rule.resource_id,
    )


# ---------------------------------------------------------------------------
# Process-wide accessor (set by the API lifespan)
# ---------------------------------------------------------------------------

_service: AlertService | None = None


def get_alert_service() -> AlertService | None:
    """The running service, or ``None`` when alert evaluation is disabled."""
    return _service


def set_alert_service(service: AlertService | None) -> None:
    global _service
    _service = service


def snooze_until_from(until: datetime | None, duration_ms: int | None, now: datetime) -> datetime:
    """Resolve a snooze request to an absolute UTC deadline (``until`` wins)."""
    if until is not None:
        return until.replace(tzinfo=timezone.utc) if until.tzinfo is None else until.astimezone(timezone.utc)
    return now + timedelta(milliseconds=duration_ms or 0)
