"""Alert system — rule-based alerts with multiple notification channels.

Supports threshold, rate-of-change, and statistical anomaly detection rules.
Alerts are dispatched through pluggable channels (log, webhook, WebSocket).
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections import deque

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
                alert_id=f"{self.name}_{int(now)}",
                rule_name=self.name,
                severity=self.severity,
                message=message,
                value=value,
                threshold=self.threshold,
                source=self.metric_name,
            )
        return None

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


class WebhookAlertChannel(AlertChannel):
    """Sends alerts to a webhook URL (stub for production use)."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def send(self, alert: Alert) -> None:
        # In production, use httpx to POST the alert
        logger.info("Webhook alert to %s: %s", self.url, alert.message)


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Manages alert rules, evaluation, and channel dispatch."""

    def __init__(self) -> None:
        self._rules: dict[str, AlertRule] = {}
        self._channels: list[AlertChannel] = [LogAlertChannel()]
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        self._rules.pop(name, None)

    def add_channel(self, channel: AlertChannel) -> None:
        self._channels.append(channel)

    async def evaluate(self, metric_name: str, value: float) -> list[Alert]:
        """Evaluate all rules matching a metric.  Returns triggered alerts."""
        triggered: list[Alert] = []
        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue
            alert = rule.evaluate(value)
            if alert is not None:
                triggered.append(alert)
                self._active_alerts[alert.alert_id] = alert
                self._alert_history.append(alert)
                # Dispatch to channels
                for channel in self._channels:
                    try:
                        await channel.send(alert)
                    except Exception:
                        logger.exception("Alert channel error")
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
        return self._alert_history[-limit:]

    @property
    def active_count(self) -> int:
        return len(self._active_alerts)
