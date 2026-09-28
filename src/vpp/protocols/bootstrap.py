"""Start/stop the settings-driven grid protocol adapters (OCPP, OpenADR, IEEE 2030.5).

Each enabled adapter is registered in the shared :class:`ProtocolRegistry`
(so ``GET /api/v1/protocols`` shows it) and supervised by a task that
retries the initial connect with exponential backoff -- a VTN or utility
server that is down at API startup must not crash the API. Once connected,
each adapter's own poll loop handles transient failures.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from vpp.protocols._transport import backoff_delay

if TYPE_CHECKING:
    from vpp.protocols.base import ProtocolAdapter, ProtocolRegistry

logger = logging.getLogger(__name__)


def build_ocpp_adapter(settings: Any) -> ProtocolAdapter:
    from vpp.protocols.ocpp import OCPPAdapter

    adapter = OCPPAdapter()
    adapter.configure(
        central_system_enabled=True,
        heartbeat_interval_s=settings.ocpp_heartbeat_interval_s,
        call_timeout_s=settings.ocpp_call_timeout_s,
        auto_accept_boot=settings.ocpp_auto_accept_boot,
        allowed_charge_points=list(settings.ocpp_allowed_charge_points) or None,
        basic_auth_password=settings.ocpp_basic_auth_password,
        authorized_id_tags=settings.ocpp_authorized_id_tags,
        remote_id_tag=settings.ocpp_remote_id_tag,
    )
    return adapter


def build_openadr_adapter(settings: Any) -> ProtocolAdapter:
    from vpp.protocols.openadr import OpenADRAdapter

    adapter = OpenADRAdapter()
    config: dict[str, Any] = {
        "role": "ven",
        "vtn_url": settings.openadr_vtn_url,
        "ven_name": settings.openadr_ven_name,
        "ven_id": settings.openadr_ven_id,
        "auto_opt_in": settings.openadr_auto_opt_in,
        "timeout_s": settings.openadr_timeout_s,
        "verify_tls": settings.openadr_verify_tls,
        "ca_path": settings.openadr_ca_path,
        "cert_path": settings.openadr_cert_path,
        "key_path": settings.openadr_key_path,
    }
    if settings.openadr_poll_interval_s is not None:
        config["poll_interval_s"] = settings.openadr_poll_interval_s
    adapter.configure(**config)
    return adapter


def build_ieee2030_5_adapter(settings: Any) -> ProtocolAdapter:
    from vpp.protocols.ieee2030_5 import IEEE2030_5Adapter

    adapter = IEEE2030_5Adapter()
    config: dict[str, Any] = {
        "server_url": settings.ieee2030_5_server_url,
        "dcap_path": settings.ieee2030_5_dcap_path,
        "lfdi": settings.ieee2030_5_lfdi,
        "timeout_s": settings.ieee2030_5_timeout_s,
        "verify_tls": settings.ieee2030_5_verify_tls,
        "ca_path": settings.ieee2030_5_ca_path,
        "cert_path": settings.ieee2030_5_cert_path,
        "key_path": settings.ieee2030_5_key_path,
        "tls_ciphers": settings.ieee2030_5_tls_ciphers,
    }
    if settings.ieee2030_5_poll_interval_s is not None:
        config["poll_interval_s"] = settings.ieee2030_5_poll_interval_s
    adapter.configure(**config)
    return adapter


async def supervise_adapter(
    adapter: ProtocolAdapter,
    registry: ProtocolRegistry,
    *,
    base_delay_s: float = 5.0,
    max_delay_s: float = 300.0,
) -> None:
    """Register *adapter*, connect it (retrying with backoff), keep it until cancelled."""
    try:
        registry.register(adapter)
    except ValueError:
        logger.warning("Protocol adapter %s already registered; replacing it", adapter.name)
        registry.unregister(adapter.name)
        registry.register(adapter)

    attempt = 0
    try:
        while True:
            try:
                await adapter.connect()
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                delay = backoff_delay(attempt, base=base_delay_s, maximum=max_delay_s)
                logger.warning(
                    "Connecting protocol adapter %s failed (%s); retrying in %.0fs",
                    adapter.name,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        await asyncio.Event().wait()  # hold until shutdown cancels us
    finally:
        if registry.get(adapter.name) is adapter:
            registry.unregister(adapter.name)
        try:
            await adapter.disconnect()
        except Exception:
            logger.exception("Error disconnecting protocol adapter %s", adapter.name)


def start_protocol_adapters(settings: Any, registry: ProtocolRegistry) -> list[asyncio.Task]:
    """Spawn supervisor tasks for every protocol enabled in *settings*."""
    builders = (
        ("ocpp_enabled", build_ocpp_adapter),
        ("openadr_enabled", build_openadr_adapter),
        ("ieee2030_5_enabled", build_ieee2030_5_adapter),
    )
    tasks: list[asyncio.Task] = []
    for flag, builder in builders:
        if not getattr(settings, flag, False):
            continue
        adapter = builder(settings)
        tasks.append(
            asyncio.create_task(
                supervise_adapter(adapter, registry), name=f"vpp-protocol-{adapter.name}"
            )
        )
    return tasks


async def stop_protocol_adapters(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            try:
                await task
            except Exception:
                logger.exception(
                    "Protocol adapter task %s raised during shutdown", task.get_name()
                )
