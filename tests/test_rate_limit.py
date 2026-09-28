"""Rate limiting is registered on the FastAPI app and actually enforced."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from vpp.api.app import create_app


@pytest.mark.asyncio
async def test_excess_requests_are_rate_limited():
    """A client that exceeds the configured requests-per-minute gets a 429."""
    app = create_app(rate_limit_enabled=True, rate_limit_requests_per_minute=3)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = [(await client.get("/health")).status_code for _ in range(5)]

    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


@pytest.mark.asyncio
async def test_rate_limiting_can_be_disabled():
    """With rate limiting off, no amount of requests gets throttled."""
    app = create_app(rate_limit_enabled=False, rate_limit_requests_per_minute=1)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = [(await client.get("/health")).status_code for _ in range(5)]

    assert statuses == [200, 200, 200, 200, 200]


# ---------------------------------------------------------------------------
# Client IP behind trusted proxies (VPP_TRUSTED_PROXIES)
# ---------------------------------------------------------------------------

WEB = "10.0.0.5"  # the (trusted) web console container


def _trusted(*cidrs: str):
    from vpp.client_ip import parse_trusted_proxies

    return parse_trusted_proxies(cidrs)


@pytest.mark.parametrize(
    ("peer", "xff", "real_ip", "trusted", "expected"),
    [
        # Default (nothing trusted): headers are ignored entirely.
        ("203.0.113.9", "1.1.1.1", "2.2.2.2", (), "203.0.113.9"),
        # Untrusted peer: spoofed headers are ignored.
        ("203.0.113.9", "1.1.1.1", "2.2.2.2", ("10.0.0.0/24",), "203.0.113.9"),
        # Trusted proxy: the forwarded client is used.
        (WEB, "198.51.100.7", None, ("10.0.0.5",), "198.51.100.7"),
        # A client-supplied XFF entry left of the real one is never reached.
        (WEB, "6.6.6.6, 198.51.100.7", None, ("10.0.0.5",), "198.51.100.7"),
        # Chains of trusted proxies are skipped from the right.
        (WEB, "198.51.100.7, 10.0.0.9", None, ("10.0.0.0/24",), "198.51.100.7"),
        # All hops trusted: the leftmost is the best we know.
        (WEB, "10.0.0.8, 10.0.0.9", None, ("10.0.0.0/24",), "10.0.0.8"),
        # X-Real-IP only when there is no X-Forwarded-For.
        (WEB, None, "198.51.100.8", ("10.0.0.5",), "198.51.100.8"),
        (WEB, "198.51.100.7", "6.6.6.6", ("10.0.0.5",), "198.51.100.7"),
        # Garbage stops the walk at the last trusted hop; bad X-Real-IP ignored.
        (WEB, "not-an-ip", None, ("10.0.0.5",), WEB),
        (WEB, "6.6.6.6, junk, 10.0.0.9", None, ("10.0.0.0/24",), "10.0.0.9"),
        (WEB, None, "junk", ("10.0.0.5",), WEB),
        # Ports and IPv6 forms.
        (WEB, "198.51.100.7:4711", None, ("10.0.0.5",), "198.51.100.7"),
        (WEB, "[2001:db8::1]:443", None, ("10.0.0.5",), "2001:db8::1"),
        ("::ffff:10.0.0.5", "198.51.100.7", None, ("10.0.0.5",), "198.51.100.7"),
        (None, "198.51.100.7", None, ("10.0.0.5",), "unknown"),
    ],
)
def test_resolve_client_ip(peer, xff, real_ip, trusted, expected):
    from vpp.client_ip import resolve_client_ip

    assert resolve_client_ip(peer, xff, real_ip, _trusted(*trusted)) == expected


def test_trusted_proxies_setting_is_validated():
    from pydantic import ValidationError

    from vpp.settings import Settings

    parsed = Settings(trusted_proxies=[" 10.0.0.5 ", "172.16.0.0/12"]).trusted_proxies
    assert parsed == ["10.0.0.5", "172.16.0.0/12"]
    with pytest.raises(ValidationError):
        Settings(trusted_proxies=["10.0.0.300"])


async def _statuses(app, peer: str, header_sets: list[dict[str, str]]) -> list[int]:
    transport = ASGITransport(app=app, client=(peer, 40000))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return [(await client.get("/health", headers=h)).status_code for h in header_sets]


def _app_trusting(monkeypatch, value: str | None):
    from vpp.settings import get_settings

    if value is None:
        monkeypatch.delenv("VPP_TRUSTED_PROXIES", raising=False)
    else:
        monkeypatch.setenv("VPP_TRUSTED_PROXIES", value)
    get_settings.cache_clear()
    try:
        return create_app(rate_limit_enabled=True, rate_limit_requests_per_minute=2)
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_trusted_proxy_gets_a_bucket_per_forwarded_client(monkeypatch):
    app = _app_trusting(monkeypatch, f'["{WEB}/32"]')
    alice = {"X-Forwarded-For": "198.51.100.1"}
    bob = {"X-Forwarded-For": "198.51.100.2"}
    statuses = await _statuses(app, WEB, [alice, alice, alice, bob, bob])
    assert statuses == [200, 200, 429, 200, 200]


@pytest.mark.asyncio
async def test_spoofed_headers_from_untrusted_peer_share_its_bucket(monkeypatch):
    """Rotating X-Forwarded-For / X-Real-IP does not escape the limit."""
    app = _app_trusting(monkeypatch, f'["{WEB}/32"]')
    spoofed = [
        {"X-Forwarded-For": f"198.51.100.{i}", "X-Real-IP": f"192.0.2.{i}"} for i in range(4)
    ]
    statuses = await _statuses(app, "203.0.113.50", spoofed)
    assert statuses == [200, 200, 429, 429]


@pytest.mark.asyncio
async def test_spoofed_prefix_through_trusted_proxy_does_not_escape(monkeypatch):
    """A client prepending fake hops is still keyed by the address the proxy saw."""
    app = _app_trusting(monkeypatch, f'["{WEB}/32"]')
    spoofed = [{"X-Forwarded-For": f"6.6.6.{i}, 198.51.100.9"} for i in range(4)]
    statuses = await _statuses(app, WEB, spoofed)
    assert statuses == [200, 200, 429, 429]


@pytest.mark.asyncio
async def test_headers_ignored_by_default(monkeypatch):
    app = _app_trusting(monkeypatch, None)
    rotating = [{"X-Forwarded-For": f"198.51.100.{i}"} for i in range(3)]
    statuses = await _statuses(app, WEB, rotating)
    assert statuses == [200, 200, 429]
