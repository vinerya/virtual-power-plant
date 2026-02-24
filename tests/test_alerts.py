"""Tests for the alert system."""

import pytest

from vpp.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertState,
    RuleType,
    LogAlertChannel,
)


class TestAlertRules:
    def test_threshold_above(self):
        rule = AlertRule(
            name="high_power",
            rule_type=RuleType.THRESHOLD,
            metric_name="power_kw",
            threshold=100.0,
            comparison=">",
            cooldown_s=0,
        )
        alert = rule.evaluate(150.0)
        assert alert is not None
        assert "150.00 > 100.00" in alert.message

    def test_threshold_below(self):
        rule = AlertRule(
            name="low_soc",
            rule_type=RuleType.THRESHOLD,
            metric_name="soc",
            threshold=0.2,
            comparison="<",
            cooldown_s=0,
        )
        assert rule.evaluate(0.1) is not None
        assert rule.evaluate(0.3) is None

    def test_threshold_cooldown(self):
        rule = AlertRule(
            name="test",
            rule_type=RuleType.THRESHOLD,
            metric_name="x",
            threshold=10,
            comparison=">",
            cooldown_s=9999,  # very long cooldown
        )
        assert rule.evaluate(20) is not None
        assert rule.evaluate(30) is None  # within cooldown

    def test_rate_of_change(self):
        rule = AlertRule(
            name="rapid_change",
            rule_type=RuleType.RATE_OF_CHANGE,
            metric_name="frequency",
            rate_window_s=10.0,
            rate_threshold=0.5,
            cooldown_s=0,
        )
        # Feed values
        import time
        rule._value_history.clear()
        now = time.time()
        rule._value_history.append((now - 5, 50.0))
        rule._value_history.append((now - 4, 50.5))
        rule._value_history.append((now - 3, 51.0))
        rule._value_history.append((now - 2, 52.0))
        rule._value_history.append((now - 1, 54.0))
        # Rate = (54 - 50) / 4 = 1.0/s > 0.5
        alert = rule.evaluate(55.0)
        assert alert is not None
        assert "rate" in alert.message.lower()

    def test_anomaly_detection(self):
        rule = AlertRule(
            name="anomaly",
            rule_type=RuleType.ANOMALY,
            metric_name="voltage",
            z_score_threshold=2.0,
            cooldown_s=0,
        )
        # Feed normal values
        import time
        rule._value_history.clear()
        now = time.time()
        for i in range(20):
            rule._value_history.append((now - 20 + i, 230.0 + (i % 3) * 0.5))
        # Now an outlier
        alert = rule.evaluate(300.0)
        assert alert is not None
        assert "z-score" in alert.message.lower()

    def test_disabled_rule(self):
        rule = AlertRule(
            name="disabled",
            rule_type=RuleType.THRESHOLD,
            metric_name="x",
            threshold=10,
            comparison=">",
            enabled=False,
        )
        assert rule.evaluate(999) is None


class TestAlertManager:
    @pytest.mark.asyncio
    async def test_evaluate_triggers_alert(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="test_rule",
            rule_type=RuleType.THRESHOLD,
            metric_name="power",
            threshold=100.0,
            comparison=">",
            cooldown_s=0,
            severity=AlertSeverity.WARNING,
        ))

        alerts = await mgr.evaluate("power", 150.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert mgr.active_count == 1

    @pytest.mark.asyncio
    async def test_no_trigger(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="test_rule",
            rule_type=RuleType.THRESHOLD,
            metric_name="power",
            threshold=100.0,
            comparison=">",
            cooldown_s=0,
        ))

        alerts = await mgr.evaluate("power", 50.0)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="rule1",
            rule_type=RuleType.THRESHOLD,
            metric_name="x",
            threshold=0,
            comparison=">",
            cooldown_s=0,
        ))
        alerts = await mgr.evaluate("x", 10.0)
        assert mgr.active_count == 1

        ok = mgr.resolve(alerts[0].alert_id)
        assert ok
        assert mgr.active_count == 0

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="rule1",
            rule_type=RuleType.THRESHOLD,
            metric_name="x",
            threshold=0,
            comparison=">",
            cooldown_s=0,
        ))
        alerts = await mgr.evaluate("x", 10.0)
        ok = mgr.acknowledge(alerts[0].alert_id)
        assert ok
        assert alerts[0].state == AlertState.ACKNOWLEDGED

    @pytest.mark.asyncio
    async def test_filter_by_severity(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="warn", rule_type=RuleType.THRESHOLD, metric_name="a",
            threshold=0, comparison=">", cooldown_s=0, severity=AlertSeverity.WARNING,
        ))
        mgr.add_rule(AlertRule(
            name="crit", rule_type=RuleType.THRESHOLD, metric_name="b",
            threshold=0, comparison=">", cooldown_s=0, severity=AlertSeverity.CRITICAL,
        ))

        await mgr.evaluate("a", 1.0)
        await mgr.evaluate("b", 1.0)

        critical = mgr.get_active_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical) == 1

    @pytest.mark.asyncio
    async def test_alert_history(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="r", rule_type=RuleType.THRESHOLD, metric_name="x",
            threshold=0, comparison=">", cooldown_s=0,
        ))
        await mgr.evaluate("x", 1.0)
        history = mgr.get_alert_history()
        assert len(history) >= 1
