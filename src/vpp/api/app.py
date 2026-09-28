"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from vpp.settings import get_settings
from vpp.db.engine import init_db, close_db, get_session_factory
from vpp.events import get_event_bus
from vpp.api.websocket import manager as websocket_manager, subscribe_event_bus_to_websocket

logger = logging.getLogger(__name__)


async def _placeholder_fetch_telemetry(battery_id: str):
    """Assemble a SOC window from the two most recent ``battery_states`` rows.

    Returns ``None`` for batteries with fewer than two recorded samples
    (never reported, or reported only once since the last tick). As of M5,
    ``battery_states`` is populated in production by
    :class:`~vpp.protocols.telemetry_ingestion.MQTTTelemetryIngestor` when
    MQTT ingestion is enabled (``VPP_MQTT_INGESTION_ENABLED``); with it
    disabled, this still degrades gracefully to a no-op, exercising the
    periodic loop's happy-path bookkeeping in development without crashing.
    """
    from vpp.db.models import BatteryStateModel
    from vpp.degradation.telemetry import TelemetryWindow

    try:
        factory = get_session_factory()
    except RuntimeError:
        return None

    async with factory() as session:
        stmt = (
            select(BatteryStateModel)
            .where(BatteryStateModel.resource_id == battery_id)
            .order_by(BatteryStateModel.timestamp.desc())
            .limit(2)
        )
        rows = list((await session.execute(stmt)).scalars().all())

    if len(rows) < 2:
        return None
    rows = list(reversed(rows))  # chronological order
    return TelemetryWindow(
        battery_id=battery_id,
        soc_trace=[r.soc / 100.0 if r.soc > 1.0 else r.soc for r in rows],
        timestamps=[r.timestamp or datetime.now(timezone.utc) for r in rows],
        temperatures_c=[r.temperature for r in rows],
    )


async def _degradation_periodic_loop(interval_minutes: int) -> None:
    """Long-running task that refreshes battery SOH on a fixed cadence.

    Iterates over all currently-registered battery resources every
    ``interval_minutes`` minutes, asks ``_placeholder_fetch_telemetry`` for
    a SOC window, and pushes the result through ``DegradationUpdater``.
    Exceptions on a single battery are logged and do not break the loop.
    """
    from vpp.db.models import ResourceModel
    from vpp.degradation.telemetry import DegradationUpdater

    factory = get_session_factory()
    updater = DegradationUpdater(session_factory=factory)

    while True:
        try:
            async with factory() as session:
                rows = list(
                    (
                        await session.execute(
                            select(ResourceModel.id).where(
                                ResourceModel.resource_type == "battery"
                            )
                        )
                    ).scalars().all()
                )
            for bid in rows:
                try:
                    window = await _placeholder_fetch_telemetry(bid)
                    if window is not None:
                        await updater.apply_window(window)
                except Exception:
                    logger.exception("Degradation tick failed for battery %s", bid)
        except Exception:
            logger.exception("Degradation periodic loop iteration failed")

        await asyncio.sleep(max(1, interval_minutes) * 60)


def _build_mqtt_adapter(settings):
    from vpp.protocols.mqtt import MQTTAdapter

    adapter = MQTTAdapter()
    adapter.configure(
        broker_host=settings.mqtt_broker_host,
        broker_port=settings.mqtt_broker_port,
        topic_prefix=settings.mqtt_topic_prefix,
        username=settings.mqtt_username,
        password=settings.mqtt_password,
    )
    return adapter


async def _mqtt_ingestion_loop(settings, *, retry_delay_seconds: float = 30.0) -> None:
    """Connect the MQTT adapter and ingest battery telemetry forever.

    Registers the adapter into the shared protocol registry (``GET
    /api/v1/protocols`` picks it up automatically). A failed initial
    connect (broker unreachable) is logged and retried on a fixed delay
    rather than crashing app startup -- the broker may come up after the
    API does. Once connected, paho's own background thread handles
    transport-level reconnection, so this outer loop only needs to cover
    the "never connected yet" case.
    """
    from vpp.api.routes.protocols import get_registry
    from vpp.protocols.telemetry_ingestion import MQTTTelemetryIngestor

    adapter = _build_mqtt_adapter(settings)
    registry = get_registry()
    try:
        registry.register(adapter)
    except ValueError:
        pass  # already registered (e.g. lifespan re-entered within one process, as in tests)

    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())

    try:
        while True:
            try:
                if not adapter.is_connected:
                    await adapter.connect()
                await ingestor.run()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "MQTT ingestion loop error; retrying in %.0fs", retry_delay_seconds
                )
                await asyncio.sleep(retry_delay_seconds)
    finally:
        registry.unregister(adapter.name)
        if adapter.is_connected:
            await adapter.disconnect()


async def _modbus_device_loop(
    resource_id: str,
    config: dict,
    *,
    retry_delay_seconds: float = 30.0,
    health_check_interval_seconds: float = 5.0,
) -> None:
    """Connect one Modbus-configured resource's adapter and keep it connected.

    ModbusAdapter.connect() starts its own internal polling task once
    connected, which already dispatches to subscribers -- this loop's only
    job is: connect (retrying on failure, e.g. device unreachable at
    startup), subscribe the persister, register into the shared protocol
    registry under a name unique to this resource (ModbusAdapter always
    constructs with name="modbus", which would collide across multiple
    devices in the same registry), and reconnect if the adapter ever drops.
    """
    from vpp.protocols.modbus import ModbusAdapter
    from vpp.protocols.modbus_ingestion import DEFAULT_POWER_REGISTER, ModbusResourcePersister
    from vpp.api.routes.protocols import get_registry

    adapter_config = {k: v for k, v in config.items() if k != "power_register"}
    power_register = config.get("power_register", DEFAULT_POWER_REGISTER)

    adapter = ModbusAdapter()
    adapter.name = f"modbus:{resource_id}"
    adapter.configure(**adapter_config)

    persister = ModbusResourcePersister(resource_id, get_session_factory(), power_register)
    adapter.subscribe("*", persister.handle_message)

    registry = get_registry()
    try:
        registry.register(adapter)
    except ValueError:
        pass  # already registered (e.g. lifespan re-entered within one process, as in tests)

    try:
        connected = False
        while True:
            try:
                if not connected:
                    await adapter.connect()
                    connected = True
                await asyncio.sleep(health_check_interval_seconds)
                if not adapter.is_connected:
                    connected = False
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Modbus connect failed for resource %s; retrying in %.0fs",
                    resource_id,
                    retry_delay_seconds,
                )
                connected = False
                await asyncio.sleep(retry_delay_seconds)
    finally:
        registry.unregister(adapter.name)
        if adapter.is_connected:
            await adapter.disconnect()


async def _modbus_ingestion_loop(settings) -> None:
    """Discover Modbus-configured resources and connect each one.

    Resources opt in via their own `metadata["modbus"]` -- see
    vpp.protocols.modbus_ingestion's module docstring. Discovery runs once
    at startup; a resource's Modbus config added after startup requires a
    restart to take effect (physical device fleets rarely change at
    runtime the way MQTT clients do, so this keeps the v1 simple).
    """
    import json

    from vpp.db.models import ResourceModel
    from vpp.protocols.modbus_ingestion import modbus_config_for_resource

    factory = get_session_factory()
    async with factory() as session:
        rows = list((await session.execute(select(ResourceModel))).scalars().all())

    device_tasks: list[asyncio.Task] = []
    for row in rows:
        metadata = json.loads(row.metadata_json) if row.metadata_json else {}
        config = modbus_config_for_resource(metadata)
        if config is None:
            continue
        device_tasks.append(
            asyncio.create_task(
                _modbus_device_loop(row.id, config),
                name=f"vpp-modbus-{row.id}",
            )
        )

    if not device_tasks:
        return

    try:
        await asyncio.gather(*device_tasks)
    except asyncio.CancelledError:
        for t in device_tasks:
            t.cancel()
        await asyncio.gather(*device_tasks, return_exceptions=True)
        raise


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    await init_db(
        settings.database_url,
        echo=settings.debug,
        use_alembic=settings.use_alembic,
    )

    event_bridge_sub_id = subscribe_event_bus_to_websocket(get_event_bus(), websocket_manager)

    task: Optional[asyncio.Task] = None
    if settings.degradation_updater_enabled:
        task = asyncio.create_task(
            _degradation_periodic_loop(
                settings.degradation_updater_interval_minutes
            ),
            name="vpp-degradation-updater",
        )
        app.state.degradation_task = task
    else:
        app.state.degradation_task = None

    mqtt_task: Optional[asyncio.Task] = None
    if settings.mqtt_ingestion_enabled:
        mqtt_task = asyncio.create_task(
            _mqtt_ingestion_loop(settings),
            name="vpp-mqtt-telemetry-ingestion",
        )
        app.state.mqtt_ingestion_task = mqtt_task
    else:
        app.state.mqtt_ingestion_task = None

    modbus_task: Optional[asyncio.Task] = None
    if settings.modbus_ingestion_enabled:
        modbus_task = asyncio.create_task(
            _modbus_ingestion_loop(settings),
            name="vpp-modbus-telemetry-ingestion",
        )
        app.state.modbus_ingestion_task = modbus_task
    else:
        app.state.modbus_ingestion_task = None

    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Degradation updater raised during shutdown")
        if mqtt_task is not None:
            mqtt_task.cancel()
            try:
                await mqtt_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("MQTT ingestion task raised during shutdown")
        if modbus_task is not None:
            modbus_task.cancel()
            try:
                await modbus_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Modbus ingestion task raised during shutdown")
        get_event_bus().unsubscribe(event_bridge_sub_id)
        await close_db()


def create_app(
    *,
    rate_limit_enabled: Optional[bool] = None,
    rate_limit_requests_per_minute: Optional[int] = None,
) -> FastAPI:
    """Build and return the configured FastAPI application.

    ``rate_limit_enabled``/``rate_limit_requests_per_minute`` override the
    corresponding settings values; pass explicitly in tests to avoid mutating
    the global cached ``Settings`` singleton.
    """
    settings = get_settings()

    app = FastAPI(
        title="Virtual Power Plant Platform",
        description=(
            "Production-ready API for managing distributed energy resources, "
            "optimization dispatch, multi-market trading, and grid protocol integration."
        ),
        version="2.0.0",
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- Middleware ----------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    enable_rate_limit = (
        settings.rate_limit_enabled if rate_limit_enabled is None else rate_limit_enabled
    )
    if enable_rate_limit:
        from vpp.auth.middleware import RateLimitMiddleware

        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=(
                rate_limit_requests_per_minute
                if rate_limit_requests_per_minute is not None
                else settings.rate_limit_requests_per_minute
            ),
        )

    # -- Routes -------------------------------------------------------------
    from .routes import (
        health, resources, optimization, trading, auth, config, protocols, v2g,
        tariffs, degradation,
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(resources.router)
    app.include_router(optimization.router)
    app.include_router(optimization.dispatches_router)
    app.include_router(trading.router)
    app.include_router(config.router)
    app.include_router(protocols.router)
    app.include_router(v2g.router)
    app.include_router(tariffs.router)
    app.include_router(degradation.router)

    # -- WebSocket ----------------------------------------------------------
    from .websocket import websocket_endpoint

    app.add_api_websocket_route("/ws", websocket_endpoint)

    return app
