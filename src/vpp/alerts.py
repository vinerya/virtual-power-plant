"""Alert system — rule-based alerts with multiple notification channels.

Supports threshold, rate-of-change, and statistical anomaly detection rules.
Alerts are dispatched through pluggable channels (log, webhook, WebSocket).

This module is the in-memory evaluation engine.  Persistence, the REST API
and the EventBus/WebSocket wiring live in :mod:`vpp.alert_service`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert severity & state
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class RuleType(str, Enum):
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ANOMALY = "anomaly"


# ---------------------------------------------------------------------------
# Alert & Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """An alert instance produced by a rule."""

    alert_id: str = ""
    rule_name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    state: AlertState = AlertState.ACTIVE
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    resolved_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "state": self.state.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    name: str
    rule_type: RuleType
    severity: AlertSeverity = AlertSeverity.WARNING
    metric_name: str = ""
    threshold: float = 0.0
    comparison: str = ">"   # >, <, >=, <=, ==
    rate_window_s: float = 60.0  # for rate-of-change rules
    rate_threshold: float = 0.0  # max acceptable rate
    z_score_threshold: float = 3.0  # for anomaly rules
    cooldown_s: float = 300.0  # min time between alerts for this rule
    enabled: bool = True
    # Optional scope: only evaluate values coming from this source (e.g. a
    # resource id) when the manager is called with ``source=...``.
    resource_id: str | None = None

    _last_fired: float = 0.0
    _value_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def evaluate(self, value: float) -> Alert | None:
        """Evaluate the rule against a new value.  Returns an Alert or None."""
        if not self.enabled:
            return None

        now = time.time()
        if now - self._last_fired < self.cooldown_s:
            return None

        self._value_history.append((now, value))

        triggered = False
        message = ""

        if self.rule_type == RuleType.THRESHOLD:
            triggered, message = self._check_threshold(value)
        elif self.rule_type == RuleType.RATE_OF_CHANGE:
            triggered, message = self._check_rate_of_change(now)
        elif self.rule_type == RuleType.ANOMALY:
            triggered, message = self._check_anomaly(value)

        if triggered:
            self._last_fired = now
            return Alert(
                alert_id=uuid.uuid4().hex,
                rule_name=self.name,
                severity=self.severity,
                message=message,
                value=value,
                threshold=self.threshold,
                source=self.metric_name,
            )
        return None

    def condition_met(self, value: float) -> bool:
        """Whether a *threshold* rule's condition holds for ``value``.

        Stateless (ignores cooldown and history); used to auto-resolve open
        threshold alerts once the value is back in range.  Always ``False``
        for non-threshold rules.
        """
        if self.rule_type != RuleType.THRESHOLD:
            return False
        return self._check_threshold(value)[0]

    def fresh_copy(self) -> AlertRule:
        """Copy of this rule with independent cooldown / history state."""
        clone = dataclasses.replace(self)
        clone._last_fired = 0.0
        clone._value_history = deque(maxlen=100)
        return clone

    def _check_threshold(self, value: float) -> tuple[bool, str]:
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
        }
        op = ops.get(self.comparison, ops[">"])
        if op(value, self.threshold):
            return True, f"{self.metric_name} = {value:.2f} {self.comparison} {self.threshold:.2f}"
        return False, ""

    def _check_rate_of_change(self, now: float) -> tuple[bool, str]:
        history = list(self._value_history)
        cutoff = now - self.rate_window_s
        recent = [(t, v) for t, v in history if t >= cutoff]
        if len(recent) < 2:
            return False, ""
        first_t, first_v = recent[0]
        last_t, last_v = recent[-1]
        dt = last_t - first_t
        if dt <= 0:
            return False, ""
        rate = abs(last_v - first_v) / dt
        if rate > self.rate_threshold:
            return True, f"{self.metric_name} rate={rate:.3f}/s exceeds {self.rate_threshold:.3f}/s"
        return False, ""

    def _check_anomaly(self, value: float) -> tuple[bool, str]:
        if len(self._value_history) < 10:
            return False, ""
        values = [v for _, v in self._value_history]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        if std < 1e-9:
            return False, ""
        z_score = abs(value - mean) / std
        if z_score > self.z_score_threshold:
            return True, f"{self.metric_name} z-score={z_score:.2f} exceeds {self.z_score_threshold:.1f}"
        return False, ""


# ---------------------------------------------------------------------------
# Alert channels
# ---------------------------------------------------------------------------

class AlertChannel(ABC):
    """Abstract alert notification channel."""

    @abstractmethod
    async def send(self, alert: Alert) -> None:
        """Send an alert through this channel."""


class LogAlertChannel(AlertChannel):
    """Sends alerts to the Python logger."""

    async def send(self, alert: Alert) -> None:
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        logger.log(level, "ALERT [%s] %s: %s", alert.severity.value, alert.rule_name, alert.message)


class WebhookDeliveryError(RuntimeError):
    """Raised when a webhook could not be delivered after all retries."""


class WebhookAlertChannel(AlertChannel):
    """POSTs alerts as JSON to a webhook URL.

    Delivery semantics:

    * ``POST <url>`` with body ``{"alert": {...}, "sent_at": <unix ts>}``.
    * Network errors, timeouts, ``429`` and ``5xx`` responses are retried
      up to ``max_retries`` times with exponential backoff
      (``backoff_base_s * 2**attempt``, capped at ``backoff_max_s``).
      Other ``4xx`` responses are treated as permanent and not retried.
    * If ``secret`` is set, the request carries
      ``X-VPP-Timestamp: <unix seconds>`` and
      ``X-VPP-Signature: sha256=<hex>`` where the digest is
      ``HMAC-SHA256(secret, f"{timestamp}.{raw_body}")``.  Receivers should
      recompute it over the raw body and reject stale timestamps to prevent
      replay.
    * A final failure raises :class:`WebhookDeliveryError`;
      :class:`AlertManager` logs channel errors without affecting other
      channels.
    """

    SIGNATURE_HEADER = "X-VPP-Signature"
    TIMESTAMP_HEADER = "X-VPP-Timestamp"

    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        timeout_s: float = 5.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 10.0,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url
        self.secret = secret
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.headers = dict(headers or {})
        self._client = client

    @staticmethod
    def sign(secret: str, timestamp: str, body: bytes) -> str:
        """Return the ``sha256=<hex>`` signature for ``body``."""
        mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
        return "sha256=" + mac.hexdigest()

    def _build_request(self, alert: Alert) -> tuple[bytes, dict[str, str]]:
        payload = {"alert": alert.to_dict(), "sent_at": time.time()}
        body = json.dumps(payload, separators=(",", ":"), default=str).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "vpp-alerts/1",
            **self.headers,
        }
        if self.secret:
            ts = str(int(time.time()))
            headers[self.TIMESTAMP_HEADER] = ts
            headers[self.SIGNATURE_HEADER] = self.sign(self.secret, ts, body)
        return body, headers

    def _backoff(self, attempt: int) -> float:
        return min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))

    async def send(self, alert: Alert) -> None:
        body, headers = self._build_request(alert)
        if self._client is not None:
            await self._deliver(self._client, body, headers)
            return
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            await self._deliver(client, body, headers)

    async def _deliver(
        self, client: httpx.AsyncClient, body: bytes, headers: dict[str, str],
    ) -> None:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(
                    self.url, content=body, headers=headers, timeout=self.timeout_s,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    return
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code != 429 and resp.status_code < 500:
                    raise WebhookDeliveryError(
                        f"Webhook {self.url} rejected alert permanently: {last_error}"
                    )
            if attempt < self.max_retries:
                delay = self._backoff(attempt)
                logger.warning(
                    "Webhook delivery to %s failed (%s); retry %d/%d in %.2fs",
                    self.url, last_error, attempt + 1, self.max_retries, delay,
                )
                await asyncio.sleep(delay)
        raise WebhookDeliveryError(
            f"Webhook {self.url} failed after {self.max_retries + 1} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Manages alert rules, evaluation, and channel dispatch.

    Rules are evaluated per *source*: when ``source`` is passed to
    :meth:`check`/:meth:`evaluate` (e.g. a resource id), every rule keeps an
    independent cooldown/history per source, so one battery firing does not
    suppress the same rule for another battery.
    """

    def __init__(self, *, history_limit: int = 1000) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._scoped_rules: dict[tuple[str, str], AlertRule] = {}
        self._channels: list[AlertChannel] = [LogAlertChannel()]
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: deque[Alert] = deque(maxlen=history_limit)

    def add_rule(self, rule: AlertRule) -> None:
        self._rules[rule.name] = rule
        self._drop_scoped(rule.name)

    def remove_rule(self, name: str) -> None:
        self._rules.pop(name, None)
        self._drop_scoped(name)

    def clear_rules(self) -> None:
        self._rules.clear()
        self._scoped_rules.clear()

    def get_rule(self, name: str) -> AlertRule | None:
        return self._rules.get(name)

    @property
    def rules(self) -> list[AlertRule]:
        return list(self._rules.values())

    def metric_names(self) -> set[str]:
        """Metric names referenced by at least one enabled rule."""
        return {r.metric_name for r in self._rules.values() if r.enabled}

    def _drop_scoped(self, name: str) -> None:
        for key in [k for k in self._scoped_rules if k[0] == name]:
            del self._scoped_rules[key]

    def add_channel(self, channel: AlertChannel) -> None:
        self._channels.append(channel)

    @property
    def channels(self) -> list[AlertChannel]:
        return list(self._channels)

    def matching_rules(self, metric_name: str, source: str | None = None) -> list[AlertRule]:
        """Rule instances for a metric (per-source state when ``source`` is set)."""
        out: list[AlertRule] = []
        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue
            if source is None:
                out.append(rule)
                continue
            if rule.resource_id is not None and rule.resource_id != source:
                continue
            key = (rule.name, source)
            scoped = self._scoped_rules.get(key)
            if scoped is None:
                scoped = rule.fresh_copy()
                self._scoped_rules[key] = scoped
            out.append(scoped)
        return out

    def check(self, metric_name: str, value: float, source: str | None = None) -> list[Alert]:
        """Evaluate matching rules without recording or dispatching."""
        triggered: list[Alert] = []
        for rule in self.matching_rules(metric_name, source):
            alert = rule.evaluate(value)
            if alert is not None:
                if source is not None:
                    alert.source = source
                alert.metadata.setdefault("metric", metric_name)
                triggered.append(alert)
        return triggered

    async def dispatch(self, alert: Alert) -> None:
        """Send an alert to every channel; a failing channel never blocks others."""
        for channel in self._channels:
            try:
                await channel.send(alert)
            except Exception:
                logger.exception("Alert channel %s error", type(channel).__name__)

    async def evaluate(
        self, metric_name: str, value: float, source: str | None = None,
    ) -> list[Alert]:
        """Evaluate all rules matching a metric.  Returns triggered alerts.

        Triggered alerts are recorded as active, appended to (bounded)
        history and dispatched to all channels.
        """
        triggered = self.check(metric_name, value, source)
        for alert in triggered:
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)
            await self.dispatch(alert)
        return triggered

    def resolve(self, alert_id: str) -> bool:
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            return False
        alert.state = AlertState.RESOLVED
        alert.resolved_at = time.time()
        del self._active_alerts[alert_id]
        return True

    def acknowledge(self, alert_id: str) -> bool:
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            return False
        alert.state = AlertState.ACKNOWLEDGED
        return True

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        alerts = list(self._active_alerts.values())
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        return list(self._alert_history)[-limit:]

    @property
    def active_count(self) -> int:
        return len(self._active_alerts)
