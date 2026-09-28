"""Shared transport helpers for the HTTP-based protocol clients.

Used by the OpenADR 2.0b VEN and the IEEE 2030.5 client: TLS context
construction (including mutual TLS with a client certificate), exponential
backoff, and ISO 8601 duration / date-time parsing that both XML schemas
rely on.
"""

from __future__ import annotations

import re
import ssl
from datetime import datetime, timezone

__all__ = [
    "ProtocolTransportError",
    "backoff_delay",
    "build_ssl_context",
    "format_iso8601_duration",
    "parse_iso8601_datetime",
    "parse_iso8601_duration",
]


class ProtocolTransportError(ConnectionError):
    """Raised when a remote protocol endpoint is unreachable or misbehaves."""


def build_ssl_context(
    *,
    verify: bool = True,
    ca_path: str | None = None,
    cert_path: str | None = None,
    key_path: str | None = None,
    key_password: str | None = None,
    ciphers: str | None = None,
) -> ssl.SSLContext | bool:
    """Build a TLS context for httpx.

    Returns ``False`` when verification is disabled and no client
    certificate is configured (plain ``verify=False``), otherwise an
    :class:`ssl.SSLContext`. ``cert_path``/``key_path`` enable mutual TLS
    (required by IEEE 2030.5 and by most production OpenADR VTNs).
    """
    if not verify and not cert_path:
        return False
    ctx = ssl.create_default_context(cafile=ca_path) if verify else ssl.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    if cert_path:
        ctx.load_cert_chain(cert_path, keyfile=key_path, password=key_password)
    if ciphers:
        ctx.set_ciphers(ciphers)
    return ctx


def backoff_delay(attempt: int, *, base: float = 1.0, maximum: float = 300.0) -> float:
    """Exponential backoff delay for the *attempt*-th consecutive failure (1-based)."""
    if attempt <= 0:
        return 0.0
    return float(min(base * (2 ** (attempt - 1)), maximum))


_DURATION_RE = re.compile(
    r"^(?P<sign>[+-])?P"
    r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$"
)


def parse_iso8601_duration(value: str) -> float:
    """Parse an ISO 8601 / RFC 5545 duration (``PT1H30M``, ``P1D``, ``-PT5M``) to seconds.

    Year/month components are rejected: their length is calendar-dependent
    and neither OpenADR nor IEEE 2030.5 uses them for event durations.
    """
    text = (value or "").strip()
    match = _DURATION_RE.match(text)
    if not match or text in ("P", "PT", "-P", "+P") or text.endswith("T"):
        raise ValueError(f"Invalid ISO 8601 duration: {value!r}")
    parts = match.groupdict()
    seconds = (
        float(parts["weeks"] or 0) * 7 * 86400
        + float(parts["days"] or 0) * 86400
        + float(parts["hours"] or 0) * 3600
        + float(parts["minutes"] or 0) * 60
        + float(parts["seconds"] or 0)
    )
    return -seconds if parts["sign"] == "-" else seconds


def format_iso8601_duration(seconds: float) -> str:
    """Format seconds as an ISO 8601 duration (``PT3600S`` style is avoided)."""
    total = round(seconds)
    sign = "-" if total < 0 else ""
    total = abs(total)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    out = f"{sign}PT"
    if hours:
        out += f"{hours}H"
    if minutes:
        out += f"{minutes}M"
    if secs or out.endswith("T"):
        out += f"{secs}S"
    return out


_FRACTION_RE = re.compile(r"\.(\d+)")


def parse_iso8601_datetime(value: str) -> float:
    """Parse an ISO 8601 date-time (``2026-01-01T12:00:00Z``) to a Unix timestamp.

    Naive values are interpreted as UTC, matching the OpenADR profile's
    requirement that date-times be expressed in UTC.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError("Empty date-time")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    # Python 3.10's fromisoformat only accepts 3- or 6-digit fractions.
    text = _FRACTION_RE.sub(lambda m: "." + (m.group(1) + "000000")[:6], text)
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
