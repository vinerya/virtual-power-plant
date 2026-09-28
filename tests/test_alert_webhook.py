"""WebhookAlertChannel: real HTTP delivery, retries/backoff and HMAC signing."""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest
import respx

from vpp import alerts as alerts_module
from vpp.alerts import (
    Alert,
    AlertManager,
    AlertRule,
    AlertSeverity,
    RuleType,
    WebhookAlertChannel,
    WebhookDeliveryError,
)

URL = "https://hooks.example.test/vpp"


@pytest.fixture
def no_sleep(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(alerts_module.asyncio, "sleep", fake_sleep)
    return delays


def _alert() -> Alert:
    return Alert(
        alert_id="a1", rule_name="soc_low", severity=AlertSeverity.CRITICAL,
        message="soc = 0.05 < 0.10", value=0.05, threshold=0.1, source="bat-1",
    )


@pytest.mark.asyncio
@respx.mock
async def test_posts_json_payload():
    route = respx.post(URL).mock(return_value=httpx.Response(204))
    await WebhookAlertChannel(URL).send(_alert())
    assert route.call_count == 1
    req = route.calls.last.request
    body = json.loads(req.content)
    assert body["alert"]["alert_id"] == "a1"
    assert body["alert"]["severity"] == "critical"
    assert "sent_at" in body
    assert req.headers["content-type"] == "application/json"
    assert WebhookAlertChannel.SIGNATURE_HEADER not in req.headers


@pytest.mark.asyncio
@respx.mock
async def test_hmac_signature_verifies():
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    await WebhookAlertChannel(URL, secret="topsecret").send(_alert())
    req = route.calls.last.request
    ts = req.headers[WebhookAlertChannel.TIMESTAMP_HEADER]
    expected = hmac.new(b"topsecret", ts.encode() + b"." + req.content, hashlib.sha256).hexdigest()
    assert req.headers[WebhookAlertChannel.SIGNATURE_HEADER] == f"sha256={expected}"


@pytest.mark.asyncio
@respx.mock
async def test_retries_5xx_then_succeeds_with_exponential_backoff(no_sleep):
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(503), httpx.Response(500), httpx.Response(200),
    ])
    await WebhookAlertChannel(URL, max_retries=3, backoff_base_s=0.5).send(_alert())
    assert route.call_count == 3
    assert no_sleep == [0.5, 1.0]


@pytest.mark.asyncio
@respx.mock
async def test_retries_network_errors_and_429(no_sleep):
    route = respx.post(URL).mock(side_effect=[
        httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.Response(429),
        httpx.Response(202),
    ])
    await WebhookAlertChannel(URL, max_retries=3).send(_alert())
    assert route.call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_gives_up_after_max_retries(no_sleep):
    route = respx.post(URL).mock(return_value=httpx.Response(502))
    with pytest.raises(WebhookDeliveryError, match="3 attempts"):
        await WebhookAlertChannel(URL, max_retries=2, backoff_max_s=0.75).send(_alert())
    assert route.call_count == 3
    assert no_sleep == [0.5, 0.75]  # capped


@pytest.mark.asyncio
@respx.mock
async def test_permanent_4xx_is_not_retried(no_sleep):
    route = respx.post(URL).mock(return_value=httpx.Response(400))
    with pytest.raises(WebhookDeliveryError, match="permanently"):
        await WebhookAlertChannel(URL, max_retries=3).send(_alert())
    assert route.call_count == 1
    assert no_sleep == []


@pytest.mark.asyncio
@respx.mock
async def test_uses_injected_client_and_custom_headers():
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient() as client:
        await WebhookAlertChannel(URL, client=client, headers={"X-Tenant": "t1"}).send(_alert())
    assert route.calls.last.request.headers["X-Tenant"] == "t1"


@pytest.mark.asyncio
@respx.mock
async def test_manager_isolates_failing_webhook(no_sleep):
    respx.post(URL).mock(return_value=httpx.Response(500))
    delivered: list[Alert] = []

    class Collect(alerts_module.AlertChannel):
        async def send(self, alert: Alert) -> None:
            delivered.append(alert)

    mgr = AlertManager()
    mgr.add_channel(WebhookAlertChannel(URL, max_retries=1))
    mgr.add_channel(Collect())
    mgr.add_rule(AlertRule(
        name="r", rule_type=RuleType.THRESHOLD, metric_name="x", threshold=0, cooldown_s=0,
    ))
    fired = await mgr.evaluate("x", 1.0)
    assert len(fired) == 1
    assert delivered == fired
