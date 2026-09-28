"""Device control: Modbus setpoint writer + setpoint actuator safety rules.

The Modbus tests run against a real pymodbus TCP server on localhost, so the
adapter's wire-level calls (unit-id keyword, register encoding, read-back)
are exercised for real.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import warnings

import pytest
from _v2g_helpers import tmp_session_factory
from sqlalchemy import select

from vpp.control.actuator import SetpointActuator
from vpp.db.models import EventLogModel, ResourceModel
from vpp.events import EventType, get_event_bus
from vpp.optimization.planning import FleetAsset
from vpp.protocols.modbus import (
    ModbusAdapter,
    encode_value,
    sunspec_model_123_registers,
    sunspec_model_124_registers,
)
from vpp.protocols.modbus_control import (
    ControlConfigError,
    ModbusControlConfig,
    ModbusSetpointWriter,
    WriteResult,
)

# ---------------------------------------------------------------------------
# A real Modbus TCP server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ModbusServer:
    def __init__(self, initial: dict[int, int] | None = None, size: int = 512) -> None:
        from pymodbus.datastore import (
            ModbusDeviceContext,
            ModbusSequentialDataBlock,
            ModbusServerContext,
        )
        from pymodbus.server import ModbusTcpServer

        values = [0] * size
        for addr, v in (initial or {}).items():
            values[addr] = v & 0xFFFF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The data block's first address is 1 (pymodbus shifts wire addresses by one).
            block = ModbusSequentialDataBlock(1, values)
            ctx = ModbusServerContext(devices={1: ModbusDeviceContext(hr=block)}, single=False)
        self.port = _free_port()
        self._server = ModbusTcpServer(ctx, address=("127.0.0.1", self.port))
        self._task: asyncio.Task | None = None
        self._probe: ModbusAdapter | None = None

    async def start(self) -> ModbusServer:
        self._task = asyncio.create_task(self._server.serve_forever())
        for _ in range(50):
            await asyncio.sleep(0.02)
            with contextlib.suppress(OSError):
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                break
        self._probe = ModbusAdapter()
        self._probe.configure(host="127.0.0.1", port=self.port, poll_interval_s=0)
        await self._probe.connect()
        return self

    async def read(self, address: int, count: int = 1) -> list[int]:
        assert self._probe is not None
        return await self._probe.read_holding(address, count)

    async def stop(self) -> None:
        if self._probe is not None:
            await self._probe.disconnect()
        await self._server.shutdown()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task


@pytest.fixture
async def modbus_server():
    servers: list[ModbusServer] = []

    async def make(initial: dict[int, int] | None = None) -> ModbusServer:
        server = await ModbusServer(initial).start()
        servers.append(server)
        return server

    yield make
    for s in servers:
        await s.stop()


async def _adapter(port: int, **config) -> ModbusAdapter:
    adapter = ModbusAdapter()
    adapter.configure(host="127.0.0.1", port=port, poll_interval_s=0, **config)
    await adapter.connect()
    return adapter


def _writer(control: dict, adapter, reference_kw: float = 10.0) -> ModbusSetpointWriter:
    cfg = ModbusControlConfig.from_modbus_config({"control": {"enabled": True, **control}})
    assert cfg is not None

    async def provider():
        return adapter

    return ModbusSetpointWriter(cfg, provider, reference_kw=reference_kw)


# ---------------------------------------------------------------------------
# Encoding / register maps / config
# ---------------------------------------------------------------------------


def test_encode_value_types_and_ranges():
    assert encode_value(123, "uint16") == [123]
    assert encode_value(-1, "int16") == [0xFFFF]
    assert encode_value(-3000, "int32") == [0xFFFF, 0xF448]
    assert encode_value(70000, "uint32") == [1, 4464]
    assert encode_value(1.5, "float32") == [0x3FC0, 0]
    with pytest.raises(ValueError):
        encode_value(-1, "uint16")  # never wraps
    with pytest.raises(ValueError):
        encode_value(40000, "int16")


def test_sunspec_offsets_and_writable_flags():
    m123 = sunspec_model_123_registers(40236)
    assert m123["wmax_lim_pct"].address == 40241
    assert m123["wmax_lim_pct_rvrt_tms"].address == 40243
    assert m123["wmax_lim_ena"].address == 40245
    assert m123["wmax_lim_pct_sf"].address == 40259
    assert m123["wmax_lim_pct"].writable and not m123["wmax_lim_pct_sf"].writable
    m124 = sunspec_model_124_registers(100)
    assert (m124["stor_ctl_mod"].address, m124["out_w_rte"].address) == (105, 112)
    assert (m124["in_w_rte"].address, m124["in_out_w_rte_sf"].address) == (113, 125)


def test_control_config_parsing_and_validation():
    assert ModbusControlConfig.from_modbus_config({"host": "x"}) is None
    cfg = ModbusControlConfig.from_modbus_config(
        {
            "unit_id": 3,
            "control": {"profile": "sunspec_123", "model_base": 1, "revert_timeout_s": 60},
        }
    )
    assert cfg is not None and not cfg.enabled  # opt-in
    assert cfg.unit_id == 3 and cfg.keepalive_s == 30.0
    assert not cfg.unverified
    assert ModbusControlConfig.from_modbus_config(
        {"control": {"profile": "sunspec_124", "model_base": 1}}
    ).unverified
    for bad in (
        {"profile": "nope"},
        {"profile": "sunspec_123"},  # no model_base
        {"register": "x", "unit": "MW"},
        {"register": "x", "sign": "up"},
        {},  # neither register nor address
    ):
        with pytest.raises(ControlConfigError):
            ModbusControlConfig.from_modbus_config({"control": bad})


# ---------------------------------------------------------------------------
# Adapter + writer against a real Modbus server
# ---------------------------------------------------------------------------


async def test_poll_reads_with_current_pymodbus_and_profiles_stay_shared(modbus_server):
    from vpp.protocols.modbus import INVERTER_MAPS

    server = await modbus_server({3: 1234, 4: (-5) & 0xFFFF})
    before = set(INVERTER_MAPS["sma_sunnyboy"].registers)
    adapter = await _adapter(
        server.port,
        device_profile="sma_sunnyboy",
        unit_id=1,
        custom_registers={
            "power": {"address": 3, "register_type": "holding", "data_type": "uint16"},
            "temp": {"address": 4, "register_type": "holding", "data_type": "int16", "scale": 0.1},
        },
    )
    try:
        values = await adapter.poll_once()
        assert values["power"] == 1234.0
        assert values["temp"] == pytest.approx(-0.5)
        # custom registers never leak into the shared vendor profile
        assert set(INVERTER_MAPS["sma_sunnyboy"].registers) == before
    finally:
        await adapter.disconnect()


async def test_sunspec_123_curtailment_write_verify_and_release(modbus_server):
    base = 40
    server = await modbus_server({base: 123, base + 23: (-2) & 0xFFFF})  # SF = -2
    adapter = await _adapter(server.port)
    try:
        writer = _writer(
            {"profile": "sunspec_123", "model_base": base, "revert_timeout_s": 600}, adapter, 8.0
        )
        result = await writer.write(4.0)  # 50 % of 8 kW
        assert result.ok and result.verified is True, result.error
        assert await server.read(base + 5) == [5000]  # 50.00 % at SF -2
        assert await server.read(base + 7) == [600]
        assert await server.read(base + 9) == [1]
        assert [w["register"] for w in result.writes] == [
            "WMaxLimPct_RvrtTms",
            "WMaxLimPct",
            "WMaxLim_Ena",
        ]
        # Curtailment cannot go negative.
        assert (await writer.write(-3.0)).ok
        assert await server.read(base + 5) == [0]

        released = await writer.release()
        assert released.ok
        assert await server.read(base + 9) == [0]
    finally:
        await adapter.disconnect()


async def test_generic_signed_register_and_writable_guard(modbus_server):
    server = await modbus_server()
    adapter = await _adapter(
        server.port,
        custom_registers={
            "bat_setpoint": {
                "address": 100,
                "count": 2,
                "register_type": "holding",
                "data_type": "int32",
                "writable": True,
            },
            "readonly": {"address": 110},
        },
    )
    try:
        writer = _writer({"register": "bat_setpoint", "unit": "W"}, adapter)
        assert (await writer.write(-3.0)).ok  # charge 3 kW
        assert await server.read(100, 2) == [0xFFFF, 0xF448]  # -3000 W
        inverted = _writer(
            {"register": "bat_setpoint", "unit": "W", "sign": "import_positive"}, adapter
        )
        assert (await inverted.write(-3.0)).ok
        assert await server.read(100, 2) == [0, 3000]
        assert (await writer.release()).ok
        assert await server.read(100, 2) == [0, 0]

        guarded = await _writer({"register": "readonly"}, adapter).write(1.0)
        assert not guarded.ok and "not flagged writable" in guarded.error
        assert await server.read(110) == [0]
    finally:
        await adapter.disconnect()


async def test_generic_pct_with_scale_factor_register_and_enable(modbus_server):
    server = await modbus_server({7: (-1) & 0xFFFF})  # SF -1 -> 0.1 % per count
    adapter = await _adapter(server.port)
    try:
        writer = _writer(
            {
                "address": 20,
                "data_type": "uint16",
                "unit": "pct",
                "scale_factor_register": 7,
                "enable_register": 21,
            },
            adapter,
            reference_kw=5.0,
        )
        assert (await writer.write(2.5)).ok
        assert await server.read(20, 2) == [500, 1]  # 50.0 %, enabled
        assert (await writer.release()).ok
        assert await server.read(21) == [0]  # enable register cleared
    finally:
        await adapter.disconnect()


class FakeIO:
    """In-memory RegisterIO; ``corrupt`` makes read-back disagree."""

    def __init__(self, regs: dict[int, int] | None = None, corrupt: bool = False) -> None:
        self.regs = dict(regs or {})
        self.corrupt = corrupt
        self.connected = True
        self.writes: list[tuple[int, list[int]]] = []

    @property
    def is_connected(self) -> bool:
        return self.connected

    def register(self, name):
        return None

    async def write_registers(self, address, values, *, unit=None):
        self.writes.append((address, list(values)))
        for i, v in enumerate(values):
            self.regs[address + i] = v

    async def read_holding(self, address, count=1, *, unit=None):
        out = [self.regs.get(address + i, 0) for i in range(count)]
        return [v ^ 1 for v in out] if self.corrupt else out


async def test_sunspec_124_charge_discharge_plan_unverified_profile():
    io = FakeIO({25: 0})  # SF 0
    writer = _writer({"profile": "sunspec_124", "model_base": 0}, io, reference_kw=10.0)
    assert writer.config.unverified
    assert (await writer.write(4.0)).ok  # discharge 40 %
    assert (io.regs[12], io.regs[13], io.regs[5]) == (40, (-40) & 0xFFFF, 3)
    assert (await writer.write(-2.0)).ok  # charge 20 %
    assert (io.regs[12], io.regs[13]) == ((-20) & 0xFFFF, 20)
    assert (await writer.release()).ok
    assert io.regs[5] == 0


async def test_readback_mismatch_and_disconnected_device_fail():
    bad = _writer({"address": 1}, FakeIO(corrupt=True))
    result = await bad.write(1.0)
    assert not result.ok and result.verified is False and "read-back mismatch" in result.error
    unverified = _writer({"address": 1, "verify": False}, FakeIO(corrupt=True))
    assert (await unverified.write(1.0)).verified is None
    offline = FakeIO()
    offline.connected = False
    result = await _writer({"address": 1}, offline).write(1.0)
    assert not result.ok and "not connected" in result.error
    overflow = await _writer({"address": 1, "data_type": "int16"}, FakeIO()).write(100.0)
    assert not overflow.ok and "out of range" in overflow.error  # 100 kW = 100000 W


# ---------------------------------------------------------------------------
# Actuator safety rules (fake writer)
# ---------------------------------------------------------------------------


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []
        self.fail = False

    async def write(self, kw):
        self.calls.append(("write", kw))
        if self.fail:
            return WriteResult(False, error="device said no")
        return WriteResult(True, [{"register": "sp", "value": kw}], verified=True)

    async def release(self):
        self.calls.append(("release", None))
        if self.fail:
            return WriteResult(False, error="device said no")
        return WriteResult(True, verified=True)


class Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _asset(rid="b1", rtype="battery", rated=10.0, online=True, **control) -> FleetAsset:
    meta: dict = {}
    if control is not None:
        meta = {"modbus": {"host": "x", "control": {"enabled": True, "address": 1, **control}}}
    return FleetAsset(
        id=rid, name=rid, resource_type=rtype, rated_power_kw=rated, online=online, metadata=meta
    )


def _actuator(clock=None, factory=None, **kw):
    writers: dict[str, FakeWriter] = {}

    def factory_fn(rid, modbus, control, rated):
        writers[rid] = writers.get(rid) or FakeWriter()
        return writers[rid]

    act = SetpointActuator(
        enabled=kw.pop("enabled", True),
        writer_factory=factory_fn,
        clock=clock or Clock(),
        session_factory=factory,
        expiry_grace_s=kw.pop("grace", 0.0),
        **kw,
    )
    return act, writers


@pytest.fixture
def db(tmp_path):
    return tmp_session_factory(tmp_path / "ctl.db")


async def test_kill_switch_opt_in_and_offline(db):
    plain = FleetAsset(id="p", name="p", resource_type="battery", rated_power_kw=5)
    off, writers = _actuator(enabled=False, factory=db)
    [d] = await off.apply([_asset()], {"b1": 5.0}, source="t")
    assert d["status"] == "disabled" and "VPP_CONTROL_ENABLED" in d["reason"]
    assert writers == {}

    act, writers = _actuator(factory=db)
    out = await act.apply(
        [plain, _asset("b2", enabled=False), _asset("b3", online=False)],
        {"p": 1.0, "b2": 1.0, "b3": 1.0},
        source="t",
    )
    assert [d["status"] for d in out] == ["not_configured", "disabled", "offline"]
    assert writers == {}  # nothing touched a device


async def test_clamping_to_resource_and_control_limits(db):
    act, writers = _actuator(factory=db)
    a = _asset("b1", rated=10.0)
    [d] = await act.apply([a], {"b1": 25.0}, source="t")
    assert d["status"] == "accepted" and d["setpoint_kw"] == 10.0 and d["clamped"]
    capped = _asset("b2", rated=10.0, max_kw=4.0, min_kw=-2.0)
    [d] = await act.apply([capped], {"b2": -9.0}, source="t")
    assert d["setpoint_kw"] == -2.0
    pv = _asset("pv", rtype="solar", rated=6.0)
    [d] = await act.apply([pv], {"pv": -3.0}, source="t")
    assert d["setpoint_kw"] == 0.0  # generation cannot absorb
    limited = _asset("b3", rated=10.0)
    limited.max_discharge_kw = 3.0
    [d] = await act.apply([limited], {"b3": 8.0}, source="t")
    assert d["setpoint_kw"] == 3.0
    assert writers["b1"].calls == [("write", 10.0)]
    bad = _asset("b4", rated=10.0, min_kw=5.0, max_kw=1.0)
    [d] = await act.apply([bad], {"b4": 1.0}, source="t")
    assert d["status"] == "failed" and "empty setpoint range" in d["reason"]


async def test_deadband_rate_limit_and_deferred_write(db):
    clock = Clock()
    act, writers = _actuator(clock, db)
    a = _asset(deadband_kw=0.5, min_interval_s=10)
    assert (await act.apply([a], {"b1": 4.0}, source="t", ttl_s=60))[0]["status"] == "accepted"
    clock.t += 1
    [d] = await act.apply([a], {"b1": 4.3}, source="t", ttl_s=60)
    assert d["status"] == "unchanged"
    assert act.status()["active"][0]["expires_at"] == pytest.approx(clock.t + 60)
    [d] = await act.apply([a], {"b1": 6.0}, source="t", ttl_s=60)
    assert d["status"] == "deferred" and "rate limit" in d["reason"]
    assert writers["b1"].calls == [("write", 4.0)]
    assert await act.tick() == []  # still inside min_interval
    clock.t += 10
    [w] = await act.tick()
    assert w["action"] == "deferred_write" and w["status"] == "accepted"
    assert writers["b1"].calls[-1] == ("write", 6.0)


async def test_watchdog_expiry_releases_or_writes_safe_setpoint(db):
    clock = Clock()
    act, writers = _actuator(clock, db, grace=5.0)
    await act.apply(
        [_asset("b1"), _asset("b2", safe_setpoint_kw=0.0)],
        {"b1": 3.0, "b2": 3.0},
        source="t",
        ttl_s=60,
    )
    clock.t += 64
    assert await act.tick() == []
    clock.t += 1  # 60 s interval + 5 s grace
    out = {d["resource_id"]: d for d in await act.tick()}
    assert out["b1"]["action"] == "expire" and out["b1"]["mode"] == "release"
    assert out["b2"]["mode"] == "safe_setpoint" and out["b2"]["setpoint_kw"] == 0.0
    assert writers["b1"].calls[-1] == ("release", None)
    assert writers["b2"].calls[-1] == ("write", 0.0)
    assert act.status()["active"] == []


async def test_failed_release_is_retried_then_abandoned(db):
    clock = Clock()
    act, writers = _actuator(clock, db)
    await act.apply([_asset()], {"b1": 3.0}, source="t", ttl_s=1)
    writers["b1"].fail = True
    clock.t += 2
    for _attempt in range(3):
        [d] = await act.tick()
        assert d["status"] == "failed"
    assert "gave up" in d["reason"]
    assert act.status()["active"] == []


async def test_failed_write_keeps_no_state_and_keepalive_refreshes(db):
    clock = Clock()
    act, writers = _actuator(clock, db)
    await act.apply([_asset(keepalive_s=30)], {"b1": 2.0}, source="t", ttl_s=600)
    clock.t += 31
    [d] = await act.tick()
    assert d["action"] == "keepalive" and writers["b1"].calls[-1] == ("write", 2.0)

    writers["b2"] = FakeWriter()
    writers["b2"].fail = True
    [d] = await act.apply([_asset("b2")], {"b2": 1.0}, source="t")
    assert d["status"] == "failed" and d["reason"] == "device said no"
    assert [a["resource_id"] for a in act.status()["active"]] == ["b1"]


async def test_watchdog_never_writes_to_resource_gone_offline(db):
    async with db() as s:
        row = ResourceModel(name="bess", resource_type="battery", rated_power=10.0, online=True)
        s.add(row)
        await s.commit()
        rid = row.id
    clock = Clock()
    act, writers = _actuator(clock, db)
    await act.apply([_asset(rid, keepalive_s=5)], {rid: 2.0}, source="t", ttl_s=600)
    async with db() as s:
        (await s.get(ResourceModel, rid)).online = False
        await s.commit()
    clock.t += 10
    [d] = await act.tick()
    assert d["status"] == "offline" and d["action"] == "abandon"
    assert writers[rid].calls == [("write", 2.0)]  # no keepalive, no release write


async def test_simulated_shutdown_release_and_audit_trail(db):
    events: list = []

    async def on_event(e):
        events.append(e)

    sub = get_event_bus().subscribe(on_event, event_types={EventType.DEVICE_SETPOINT})
    try:
        act, writers = _actuator(factory=db)
        out = await act.apply(
            [_asset("sim", simulate=True), _asset("b1"), FleetAsset("p", "p", "battery", 5.0)],
            {"sim": 1.0, "b1": 2.0, "p": 1.0},
            source="unit",
            run_id="run-1",
        )
        assert [d["status"] for d in out] == ["simulated", "accepted", "not_configured"]
        assert "sim" not in writers
        released = await act.shutdown()
        assert {d["resource_id"]: d["status"] for d in released} == {
            "sim": "simulated",
            "b1": "accepted",
        }
        assert writers["b1"].calls[-1] == ("release", None)
    finally:
        get_event_bus().unsubscribe(sub)

    async with db() as s:
        rows = (await s.execute(select(EventLogModel))).scalars().all()
    details = [json.loads(r.details_json) for r in rows]
    assert all(r.event_type == "device_setpoint" for r in rows)
    assert sorted((d["resource_id"], d["action"]) for d in details) == [
        ("b1", "release"),
        ("b1", "setpoint"),
        ("sim", "release"),
        ("sim", "setpoint"),
    ]  # not_configured resources are not logged
    assert {d["run_id"] for d in details if d["action"] == "setpoint"} == {"run-1"}
    assert [e.data["source"] for e in events] == ["unit", "control.shutdown"]
    assert events[0].data["summary"] == {"simulated": 1, "accepted": 1, "not_configured": 1}


async def test_ev_assets_are_left_to_ocpp(db):
    act, _ = _actuator(factory=db)
    ev = _asset("ev:1")
    assert await act.apply([ev], {"ev:1": 3.0}, source="t") == []


async def test_actuator_default_writer_uses_private_modbus_connection(modbus_server, db):
    server = await modbus_server()
    modbus = {
        "mode": "tcp",
        "host": "127.0.0.1",
        "port": server.port,
        "device_profile": "none",
        "power_register": "ac_power",
        "custom_registers": {
            "sp": {"address": 30, "data_type": "int16", "writable": True},
        },
        "control": {"enabled": True, "register": "sp", "unit": "W", "min_interval_s": 0},
    }
    asset = FleetAsset("b1", "b1", "battery", 5.0, metadata={"modbus": modbus})
    act = SetpointActuator(enabled=True, session_factory=db)
    [d] = await act.apply([asset], {"b1": -2.5}, source="t")
    assert d["status"] == "accepted" and d["verified"] is True, d
    assert await server.read(30) == [(-2500) & 0xFFFF]
    [d] = await act.apply([asset], {"b1": 1.2}, source="t")
    assert await server.read(30) == [1200]
    await act.shutdown()  # releases (0 W) and closes the private connection
    assert await server.read(30) == [0]
    assert act._own_adapters == {}


async def test_start_and_stop_control_lifecycle():
    from types import SimpleNamespace

    from vpp.control import actuator as mod

    try:
        assert mod.start_control(SimpleNamespace(control_enabled=False)) is None
        assert mod.get_setpoint_actuator().enabled is False
        task = mod.start_control(
            SimpleNamespace(control_enabled=True, control_watchdog_interval_s=0.5)
        )
        assert task is not None and mod.get_setpoint_actuator().enabled
        await asyncio.sleep(0)
        await mod.stop_control(task)
        assert task.cancelled() or task.done()
    finally:
        mod.set_setpoint_actuator(None)
