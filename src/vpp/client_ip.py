"""Resolve the real client IP behind trusted reverse proxies.

``X-Forwarded-For`` / ``X-Real-IP`` are plain request headers: any client can
send them. They are therefore honoured **only** when the TCP peer is one of
``VPP_TRUSTED_PROXIES`` (CIDRs or single addresses; empty by default, which
ignores the headers entirely and uses the peer address).

When the peer is trusted, ``X-Forwarded-For`` is walked from the right
(nearest hop first) past every trusted proxy; the first untrusted address is
the client. Entries further left were supplied by that client and are
ignored, so a spoofed ``X-Forwarded-For: 1.2.3.4`` only adds an entry the
walk never reaches. Without ``X-Forwarded-For``, a valid ``X-Real-IP`` from a
trusted peer is used. A malformed entry stops the walk at the last trusted
hop.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def parse_trusted_proxies(values: Iterable[str]) -> tuple[IPNetwork, ...]:
    """Parse CIDRs / addresses; raises ``ValueError`` on an invalid entry."""
    networks: list[IPNetwork] = []
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError as exc:
            raise ValueError(f"invalid trusted proxy {raw!r}: {exc}") from exc
    return tuple(networks)


def _parse_ip(value: str) -> IPAddress | None:
    """Parse ``1.2.3.4``, ``1.2.3.4:port``, ``::1`` or ``[::1]:port``."""
    value = value.strip().strip('"')
    if value.startswith("["):
        value = value[1:].split("]", 1)[0]
    elif value.count(":") == 1:
        value = value.split(":", 1)[0]
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _is_trusted(addr: IPAddress, trusted: Sequence[IPNetwork]) -> bool:
    return any(addr.version == net.version and addr in net for net in trusted)


def resolve_client_ip(
    peer: str | None,
    forwarded_for: str | None,
    real_ip: str | None,
    trusted: Sequence[IPNetwork],
) -> str:
    """Return the client address for rate limiting (see module docstring).

    ``forwarded_for`` is the (comma-joined) ``X-Forwarded-For`` value.
    """
    if peer is None:
        return "unknown"
    peer_addr = _parse_ip(peer)
    if not trusted or peer_addr is None or not _is_trusted(peer_addr, trusted):
        return peer

    if forwarded_for:
        client = peer_addr
        for entry in reversed(forwarded_for.split(",")):
            addr = _parse_ip(entry)
            if addr is None:
                break
            client = addr
            if not _is_trusted(addr, trusted):
                break
        return str(client)

    if real_ip:
        addr = _parse_ip(real_ip)
        if addr is not None:
            return str(addr)
    return peer
