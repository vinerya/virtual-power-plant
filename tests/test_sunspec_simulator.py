"""SunSpec device simulator, and the VPP read + control path end to end against it.

The end-to-end tests run the simulator's pymodbus TCP server on an
ephemeral localhost port and drive it through the project's own code:
SunSpec discovery, the Modbus ingestion loop (``_modbus_device_loop``, which
persists polled values), ``POST /api/v1/optimization/dispatch`` with
``apply=true`` and the real :class:`SetpointActuator` / Modbus setpoint
writer. The simulator is stepped by hand (``tick_s=None``) so the physics
is deterministic.

What this proves is that the VPP implements the SunSpec specification
consistently from register map to dispatch; it says nothing about how a
particular vendor's firmware interprets the same registers.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import uuid
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from vpp.protocols.modbus import (
    INVERTER_MAPS,
    SUNSPEC_INVERTER_BASE,
    SUNSPEC_MODEL_123_LENGTH,
    SUNSPEC_MODEL_124_LENGTH,
    ModbusAdapter,
    discover_sunspec_models,
    find_sunspec_model,
    sunspec_model_123_registers,
    sunspec_model_124_registers,
)
from vpp.simulators.sunspec import (
    ILLEGAL_ADDRESS,
    ILLEGAL_VALUE,
    MODEL_POINTS,
    SimulatorConfig,
    SunSpecDevice,
    SunSpecSimulator,
    model_length,
)

# ---------------------------------------------------------------------------
# The simulator's layout against the SunSpec definitions and the project maps
# ---------------------------------------------------------------------------


@cache
def _spec(model_id: int) -> dict[str, Any]:
    sunspec2 = pytest.importorskip(
        "sunspec2", reason="pysunspec2 not installed (pip install -e '.[dev]')"
    )
    path = Path(sunspec2.__file__).parent / "models" / "json" / f"model_{model_id}.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("model_id", sorted(MODEL_POINTS))
def test_simulator_points_match_the_sunspec_model(model_id: int) -> None:
    spec_points = _spec(model_id)["group"]["points"]
    ours = MODEL_POINTS[model_id]
    assert [p[0] for p in ours] == [p["name"] for p in spec_points]
    for (name, stype, size, sf, rw), point in zip(ours, spec_points, strict=True):
        assert stype == point["type"], name
        assert size == point["size"], name
        assert sf == point.get("sf"), name
        assert rw == (point.get("access") == "RW"), name
    assert model_length(model_id, common_length=66) == sum(p["size"] for p in spec_points) - 2


def test_simulator_layout_matches_the_project_register_maps() -> None:
    """Every SunSpec register the VPP reads or writes sits where the simulator serves it."""
    assert model_length(123) == SUNSPEC_MODEL_123_LENGTH
    assert model_length(124) == SUNSPEC_MODEL_124_LENGTH
    for inv, profile in ((103, "solaredge_se"), (113, "fronius_symo")):
        dev = SunSpecDevice(SimulatorConfig(inverter_model=inv))
        assert dev.model_address(inv) == SUNSPEC_INVERTER_BASE  # common model L = 65
        maps = {
            profile: INVERTER_MAPS[profile].registers,
            "123": sunspec_model_123_registers(dev.model_address(123)),
            "124": sunspec_model_124_registers(dev.model_address(124)),
        }
        for regs in maps.values():
            for key, reg in regs.items():
                ref = reg.sunspec
                assert ref is not None
                assert ref.base == dev.model_address(ref.model), key
                assert reg.address == dev.point_address(ref.model, ref.point), key


# ---------------------------------------------------------------------------
# Device model (no network)
# ---------------------------------------------------------------------------


def _write(dev: SunSpecDevice, model: int, point: str, raw: int) -> int | None:
    return dev.write(dev.point_address(model, point), [raw & 0xFFFF])


def test_register_image_header_strings_and_not_implemented_values() -> None:
    dev = SunSpecDevice(SimulatorConfig(serial="SN-42"))
    assert dev.read(40000, 4) == [0x5375, 0x6E53, 1, 65]
    assert dev.read(dev.end_address, 2) == [0xFFFF, 0]
    assert dev.read(dev.end_address + 2) == ILLEGAL_ADDRESS
    assert dev.read(39999) == ILLEGAL_ADDRESS
    sn = b"".join(r.to_bytes(2, "big") for r in dev.raw(1, "SN")).rstrip(b"\0")
    assert sn == b"SN-42"
    assert dev.raw(103, "TmpTrns") == [0x8000]  # int16 not implemented
    assert dev.raw(103, "StVnd") == [0xFFFF]  # enum16 not implemented
    assert dev.raw(103, "EvtVnd1") == [0xFFFF, 0xFFFF]  # bitfield32 not implemented
    assert dev.raw(123, "VArPct_SF") == [0x8000]
    assert dev.value(103, "Hz") == pytest.approx(50.0)
    assert dev.raw(103, "Hz") == [5000]  # Hz_SF = -2
    f = SunSpecDevice(SimulatorConfig(inverter_model=113))
    assert f.raw(113, "TmpOt") == [0x7FC0, 0]  # float32 NaN
    assert f.value(113, "W") == pytest.approx(6000.0)
    # Full common model (with Pad) moves every following model by one register.
    padded = SunSpecDevice(SimulatorConfig(common_length=66))
    assert padded.model_address(103) == SUNSPEC_INVERTER_BASE + 1
    assert padded.raw(1, "Pad") == [0x8000]


def test_write_validation_is_strict_and_atomic() -> None:
    dev = SunSpecDevice()
    lim = dev.point_address(123, "WMaxLimPct")
    assert _write(dev, 123, "WMaxLimPct_SF", 0) == ILLEGAL_ADDRESS  # read-only
    assert _write(dev, 103, "W", 1) == ILLEGAL_ADDRESS
    assert dev.write(40000, [0]) == ILLEGAL_ADDRESS  # marker
    assert _write(dev, 123, "WMaxLimPct", 1001) == ILLEGAL_VALUE  # 100.1 % at SF -1
    assert _write(dev, 123, "WMaxLim_Ena", 2) == ILLEGAL_VALUE
    assert _write(dev, 124, "StorCtl_Mod", 4) == ILLEGAL_VALUE
    assert _write(dev, 124, "InWRte", 10001) == ILLEGAL_VALUE  # 100.01 % at SF -2
    assert _write(dev, 123, "VArWMaxPct", 0) == ILLEGAL_ADDRESS  # its SF is not implemented
    # WMaxLimPct .. WMaxLim_Ena in one request; the bad Ena rejects all of it.
    assert dev.write(lim, [500, 0, 0, 0, 7]) == ILLEGAL_VALUE
    assert dev.raw(123, "WMaxLimPct") == [1000]
    assert dev.write(lim, [500, 0, 0, 0, 1]) is None
    assert dev.value(123, "WMaxLimPct") == pytest.approx(50.0)


def test_curtailment_and_revert_timeout() -> None:
    dev = SunSpecDevice(SimulatorConfig(wmax_w=10_000, pv_available_w=6_000))
    assert dev.ac_power_w == pytest.approx(6000)
    assert _write(dev, 123, "WMaxLimPct", 400) is None  # 40.0 %
    assert _write(dev, 123, "WMaxLimPct_RvrtTms", 30) is None
    dev.step(1.0)
    assert dev.ac_power_w == pytest.approx(6000)  # not enabled yet
    assert _write(dev, 123, "WMaxLim_Ena", 1) is None
    dev.step(1.0)
    assert dev.ac_power_w == pytest.approx(4000)
    assert dev.raw(103, "St") == [5]  # THROTTLED
    dev.step(28.0)  # 29 s after the last write
    assert dev.ac_power_w == pytest.approx(4000)
    dev.step(1.0)  # revert timer fires
    assert dev.raw(123, "WMaxLim_Ena") == [0]
    assert dev.ac_power_w == pytest.approx(6000)
    assert [what for _, what in dev.reverts] == ["WMaxLim_Ena"]
    # Disconnect: no output.
    assert _write(dev, 123, "Conn", 0) is None
    dev.step(1.0)
    assert dev.ac_power_w == 0 and dev.raw(103, "St") == [8]


def test_battery_forced_charge_discharge_soc_and_limits() -> None:
    cfg = SimulatorConfig(
        pv_available_w=0.0,
        battery_capacity_wh=10_000,
        battery_max_w=5_000,
        soc_pct=50.0,
        min_reserve_pct=10.0,
        battery_efficiency=0.9,
    )
    dev = SunSpecDevice(cfg)
    base = dev.model_address(124)
    dev.step(3600)
    assert dev.soc_pct == pytest.approx(50.0)  # StorCtl_Mod 0: idle
    # Forced charge at 40 %: OutWRte = -40 %, InWRte = +40 % (SF -2).
    assert dev.write(base + 12, [(-4000) & 0xFFFF, 4000]) is None
    assert _write(dev, 124, "StorCtl_Mod", 3) is None
    dev.step(3600)  # 2 kW for an hour, 90 % one-way efficiency
    assert dev.battery_power_w == pytest.approx(-2000)
    assert dev.ac_power_w == pytest.approx(-2000)
    assert dev.soc_pct == pytest.approx(50.0 + 18.0)
    assert dev.value(124, "ChaState") == pytest.approx(68.0)
    assert dev.raw(124, "ChaSt") == [4]  # CHARGING
    # Charging stops at 100 %.
    dev.step(3 * 3600)
    assert dev.soc_pct == pytest.approx(100.0)
    dev.step(60)
    assert dev.battery_power_w == 0 and dev.raw(124, "ChaSt") == [5]  # FULL
    # Forced discharge at 100 % down to the 10 % reserve.
    assert dev.write(base + 12, [10000, (-10000) & 0xFFFF]) is None
    dev.step(60)
    assert dev.battery_power_w == pytest.approx(5000)
    dev.step(10 * 3600)
    assert dev.soc_pct == pytest.approx(10.0)
    dev.step(60)
    assert dev.battery_power_w == 0 and dev.raw(124, "ChaSt") == [2]  # EMPTY
    # PV-only charging (ChaGriSet = 0) is capped by the available PV power.
    assert dev.write(base + 12, [(-10000) & 0xFFFF, 10000]) is None
    assert _write(dev, 124, "ChaGriSet", 0) is None
    dev.pv_available_w = 1500.0
    dev.step(1)
    assert dev.battery_power_w == pytest.approx(-1500)
    assert dev.ac_power_w == pytest.approx(0)


def test_storage_revert_timeout_clears_storctl_mod() -> None:
    dev = SunSpecDevice(SimulatorConfig(pv_available_w=0.0))
    base = dev.model_address(124)
    assert dev.write(base + 12, [2000, (-2000) & 0xFFFF]) is None  # discharge 20 %
    assert _write(dev, 124, "InOutWRte_RvrtTms", 60) is None
    assert _write(dev, 124, "StorCtl_Mod", 3) is None
    dev.step(59)
    assert dev.battery_power_w == pytest.approx(1000)
    dev.step(1)
    assert dev.raw(124, "StorCtl_Mod") == [0]
    assert dev.battery_power_w == 0


# ---------------------------------------------------------------------------
# Over the wire
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _simulator(**config: Any):
    sim = await SunSpecSimulator(SimulatorConfig(**config), tick_s=None).start("127.0.0.1", 0)
    try:
        yield sim
    finally:
        await sim.stop()


async def _adapter(port: int, **config: Any) -> ModbusAdapter:
    adapter = ModbusAdapter()
    adapter.configure(host="127.0.0.1", port=port, poll_interval_s=0, **config)
    await adapter.connect()
    return adapter


def _custom(regs: dict[str, Any], *names: str) -> dict[str, dict[str, Any]]:
    """``custom_registers`` config (JSON-shaped) for the named map registers."""
    out = {}
    for name in names:
        d = dataclasses.asdict(regs[name])
        d["register_type"] = regs[name].register_type.value
        out[name] = d
    return out


@pytest.mark.parametrize(("inverter", "profile"), [(103, "solaredge_se"), (113, "fronius_symo")])
async def test_discovery_and_polling_in_engineering_units(inverter: int, profile: str) -> None:
    async with _simulator(inverter_model=inverter, pv_available_w=4321.0, soc_pct=63.4) as sim:
        base124 = sim.device.model_address(124)
        adapter = await _adapter(
            sim.port,
            device_profile=profile,
            custom_registers=_custom(
                sunspec_model_124_registers(base124),
                "cha_state",
                "cha_state_sf",
                "w_cha_max",
                "w_cha_max_sf",
            ),
        )
        try:
            models = await discover_sunspec_models(adapter)
            assert [m.model_id for m in models] == [1, inverter, 123, 124]
            assert find_sunspec_model(models, inverter).address == SUNSPEC_INVERTER_BASE  # type: ignore[union-attr]
            assert find_sunspec_model(models, 124).address == base124  # type: ignore[union-attr]
            values = await adapter.poll_once()
        finally:
            await adapter.disconnect()
    assert values["ac_power"] == pytest.approx(4321.0, abs=0.5)
    assert values["frequency"] == pytest.approx(50.0)
    assert values["dc_power"] == pytest.approx(4321.0 / 0.97, abs=1.0)
    assert values["ac_energy"] == pytest.approx(1_000_000, abs=1)
    assert values["cha_state"] == pytest.approx(63.4)  # raw 634 at ChaState_SF -1
    assert values["w_cha_max"] == pytest.approx(5000.0)
    if inverter == 103:
        assert values["frequency_scale"] == -2 and values["cha_state_sf"] == -1
        assert values["temperature"] == pytest.approx(35.0 + 25.0 * 0.4321, abs=0.1)
    else:  # model 113: float32, no scale factors
        assert "frequency_scale" not in values


async def test_a_scale_factor_change_between_requests_never_tears_a_reading() -> None:
    """Each value is decoded with the scale factor from the same request.

    The simulated device re-encodes its output after *every* request, flipping
    ``W_SF`` between -1 and 0 (1234.6 W is served as 12346 at -1 or 1235 at
    0), as a device rescaling a changing value would. Reading ``W`` and
    ``W_SF`` in two requests pairs one encoding's value with the other's
    exponent; the adapter reads the whole model block in one request.
    """
    async with _simulator(inverter_model=103) as sim:
        dev = sim.device
        requests: list[tuple[int, int]] = []
        serve = dev.read
        encodings = [-1, 0]

        def set_encoding(sf: int) -> None:
            dev._set_raw(103, "W_SF", [sf & 0xFFFF])
            dev.set_value(103, "W", 1234.6)

        def read(address: int, count: int = 1) -> list[int] | int:
            got = serve(address, count)
            requests.append((address, count))
            encodings.reverse()
            set_encoding(encodings[0])  # the device changes after answering
            return got

        set_encoding(encodings[0])
        dev.read = read  # type: ignore[method-assign]
        w, w_sf = dev.point_address(103, "W"), dev.point_address(103, "W_SF")

        adapter = await _adapter(sim.port, device_profile="solaredge_se")
        try:
            # What separate requests would decode: 10x off.
            [raw_w] = await adapter.read_holding(w, 1)
            [raw_sf] = await adapter.read_holding(w_sf, 1)
            sf = raw_sf - 0x10000 if raw_sf >= 0x8000 else raw_sf
            assert raw_w * 10.0**sf == pytest.approx(12346.0)
            requests.clear()
            readings = [(await adapter.poll_once())["ac_power"] for _ in range(6)]
        finally:
            await adapter.disconnect()
    assert readings == pytest.approx([1234.6, 1235.0] * 3)
    # One request per poll: the model 103 span from W (+14) to Tmp_SF (+37).
    assert requests == [(SUNSPEC_INVERTER_BASE + 14, 24)] * 6


def test_read_plan_groups_models_and_scale_factor_pairs() -> None:
    from vpp.protocols.modbus import (
        RegisterDefinition,
        plan_register_reads,
        split_register_block,
    )

    regs = dict(INVERTER_MAPS["solaredge_se"].registers)
    regs.update(sunspec_model_124_registers(40200))
    regs["plain"] = RegisterDefinition(5, 1, name="plain")
    regs["far"] = RegisterDefinition(1000, 1, scale_factor="far_sf")
    regs["far_sf"] = RegisterDefinition(1400, 1)  # too far apart for one request
    plan = plan_register_reads(regs)
    assert sorted(INVERTER_MAPS["solaredge_se"].registers) == sorted(plan[0])
    assert sorted(sunspec_model_124_registers(40200)) == sorted(plan[1])
    assert plan[2:] == [["plain"], ["far"], ["far_sf"]]
    # A rejected block falls back to value + scale-factor pairs.
    split = split_register_block(plan[0], regs)
    assert ["ac_power", "ac_power_scale"] in split and ["frequency", "frequency_scale"] in split
    assert all(len(p) <= 2 for p in split)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.05)


async def test_dispatch_apply_end_to_end_against_the_simulator(client, auth_headers) -> None:
    """Poll -> dispatch(apply) -> Modbus writes -> device responds -> next poll sees it."""
    from sqlalchemy import select

    from vpp.api.app import _modbus_device_loop
    from vpp.api.routes.protocols import get_registry
    from vpp.control.actuator import SetpointActuator, set_setpoint_actuator
    from vpp.db.engine import get_session_factory
    from vpp.db.models import ResourceModel
    from vpp.portal.telemetry import latest_soc

    async with _simulator(
        inverter_model=103, pv_available_w=6000.0, wmax_w=10_000, soc_pct=50.0
    ) as sim:
        dev = sim.device
        common = {"mode": "tcp", "host": "127.0.0.1", "port": sim.port, "unit_id": 1}
        solar_modbus = {
            **common,
            "device_profile": "solaredge_se",
            "poll_interval_s": 0.2,
            "power_register": "ac_power",
            "control": {
                "enabled": True,
                "profile": "sunspec_123",
                "model_base": "auto",
                "reference_kw": 10.0,  # the inverter's WMax
                "revert_timeout_s": 120,
                "min_interval_s": 0,
            },
        }
        storage_regs = sunspec_model_124_registers(dev.model_address(124))
        battery_modbus = {
            **common,
            "device_profile": "sunspec_storage",  # no built-in map: custom registers only
            "custom_registers": _custom(storage_regs, "cha_state", "cha_state_sf"),
            "poll_interval_s": 0.2,
            "power_register": "battery_w",  # not polled: this resource persists SoC only
            "soc_register": "cha_state",
            "control": {
                "enabled": True,
                "profile": "sunspec_124",
                "model_base": "auto",
                "reference_kw": 5.0,  # WChaMax
                "min_interval_s": 0,
            },
        }

        async def create(rtype: str, rated: float, modbus: dict, **meta: Any) -> str:
            resp = await client.post(
                "/api/v1/resources/",
                json={
                    "name": f"sunspec-sim-{rtype}-{uuid.uuid4().hex[:8]}",
                    "resource_type": rtype,
                    "rated_power": rated,
                    "metadata": {"modbus": modbus, **meta},
                },
                headers=auth_headers,
            )
            assert resp.status_code == 201, resp.text
            return str(resp.json()["id"])

        solar = await create("solar", 10.0, solar_modbus)
        battery = await create("battery", 5.0, battery_modbus, capacity_kwh=10.0)

        async def current_kw(rid: str) -> float:
            async with get_session_factory()() as s:
                row = (
                    await s.execute(select(ResourceModel).where(ResourceModel.id == rid))
                ).scalar_one()
                return float(row.current_power or 0.0)

        async def soc(rid: str) -> float | None:
            async with get_session_factory()() as s:
                return (await latest_soc(s, [rid])).get(rid)

        actuator = SetpointActuator(enabled=True, registry=get_registry())
        set_setpoint_actuator(actuator)
        loops = [
            asyncio.create_task(_modbus_device_loop(solar, solar_modbus)),
            asyncio.create_task(_modbus_device_loop(battery, battery_modbus)),
        ]
        try:
            # 1. The ingestion loop persists the polled output (W -> kW) and SoC.
            async def polled() -> bool:
                return await current_kw(solar) == pytest.approx(6.0) and (
                    await soc(battery)
                ) == pytest.approx(0.5)

            await _wait_for(polled)

            # 2. Curtail the PV inverter to 4 kW through the dispatch API.
            resp = await client.post(
                "/api/v1/optimization/dispatch",
                json={"target_power_kw": 4.0, "resource_ids": [solar], "apply": True},
                headers=auth_headers,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            [delivery] = body["device_deliveries"]
            assert delivery["status"] == "accepted", delivery
            assert delivery["verified"] is True
            assert delivery["setpoint_kw"] == pytest.approx(4.0)
            # Registers as the device sees them: 40.0 % at WMaxLimPct_SF -1.
            assert dev.raw(123, "WMaxLimPct") == [400]
            assert dev.raw(123, "WMaxLimPct_RvrtTms") == [120]
            assert dev.raw(123, "WMaxLim_Ena") == [1]
            # The device follows on its next step; the next poll reflects it.
            dev.step(1.0)
            assert dev.ac_power_w == pytest.approx(4000)

            async def curtailed() -> bool:
                return await current_kw(solar) == pytest.approx(4.0)

            await _wait_for(curtailed)

            # 2b. Ask for full output. The polled 4 kW is the VPP's own cap, not
            # the PV's availability: the optimiser estimates it from the last
            # reading before the cap (6 kW) and the actuator lifts the limit.
            resp = await client.post(
                "/api/v1/optimization/dispatch",
                json={"target_power_kw": 10.0, "resource_ids": [solar], "apply": True},
                headers=auth_headers,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            [alloc] = body["allocations"]
            assert alloc["availability_basis"] == "pre_curtailment_telemetry"
            assert alloc["allocated_power_kw"] == pytest.approx(6.0)
            [delivery] = body["device_deliveries"]
            assert delivery["status"] == "accepted", delivery
            assert delivery["verified"] is True
            assert delivery["uncurtailed"] is True
            assert delivery["setpoint_kw"] == pytest.approx(10.0)
            assert [w["register"] for w in delivery["writes"]] == ["WMaxLim_Ena"]
            assert dev.raw(123, "WMaxLim_Ena") == [0]  # limit lifted
            assert actuator.output_limits() == {}
            dev.step(1.0)
            assert dev.ac_power_w == pytest.approx(6000)

            async def restored() -> bool:
                return await current_kw(solar) == pytest.approx(6.0)

            await _wait_for(restored)

            # 2c. Curtail to 4 kW again (uncapped output is plain telemetry now).
            resp = await client.post(
                "/api/v1/optimization/dispatch",
                json={"target_power_kw": 4.0, "resource_ids": [solar], "apply": True},
                headers=auth_headers,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["allocations"][0]["availability_basis"] == "telemetry"
            assert body["device_deliveries"][0]["status"] == "accepted"
            assert dev.raw(123, "WMaxLim_Ena") == [1]
            assert actuator.output_limits()[solar].limit_kw == pytest.approx(4.0)
            dev.step(1.0)
            await _wait_for(curtailed)

            # 3. Charge the battery at 2 kW (model 124, forced charge).
            resp = await client.post(
                "/api/v1/optimization/dispatch",
                json={"target_power_kw": -2.0, "resource_ids": [battery], "apply": True},
                headers=auth_headers,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            [alloc] = body["allocations"]
            assert alloc["allocated_power_kw"] == pytest.approx(-2.0)
            assert alloc["soc_source"] == "telemetry"  # the SoC polled from ChaState
            [delivery] = body["device_deliveries"]
            assert delivery["status"] == "accepted", delivery
            assert delivery["verified"] is True
            assert dev.raw(124, "OutWRte") == [(-4000) & 0xFFFF]  # -40.00 % at SF -2
            assert dev.raw(124, "InWRte") == [4000]
            assert dev.raw(124, "StorCtl_Mod") == [3]

            dev.step(360.0)  # six minutes at 2 kW, 95 % one-way efficiency
            assert dev.battery_power_w == pytest.approx(-2000)
            expected_soc = 50.0 + 2000 * 0.1 * 0.95 / 10_000 * 100  # 51.9 %
            assert dev.soc_pct == pytest.approx(expected_soc)
            # The PV limit's 120 s revert timer ran out during the step.
            assert [what for _, what in dev.reverts] == ["WMaxLim_Ena"]
            assert dev.ac_power_w == pytest.approx(6000 - 2000)

            async def charged() -> bool:
                return (await soc(battery)) == pytest.approx(expected_soc / 100, abs=1e-3)

            await _wait_for(charged)
            await _wait_for(curtailed)  # 4 kW again, now PV 6 kW minus charging 2 kW

            # 4. Release: control handed back, the battery idles.
            released = await actuator.release(reason="test done")
            assert {d["status"] for d in released} == {"accepted"}
            assert dev.raw(124, "StorCtl_Mod") == [0]
            dev.step(60.0)
            assert dev.battery_power_w == 0
            assert dev.soc_pct == pytest.approx(expected_soc)
        finally:
            for task in loops:
                task.cancel()
            for task in loops:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await actuator.shutdown()
            set_setpoint_actuator(None)
            # Keep later fleet-wide dispatch tests away from these resources.
            # A poll cancelled mid-write rolls back and closes its session
            # before the loop exits, so SQLite's write lock is free here.
            from sqlalchemy import update

            async with get_session_factory()() as s:
                await s.execute(
                    update(ResourceModel)
                    .where(ResourceModel.id.in_([solar, battery]))
                    .values(online=False)
                )
                await s.commit()


def test_cli_parser_builds_a_valid_config() -> None:
    from vpp.simulators.sunspec import build_parser, config_from_args

    args = build_parser().parse_args(
        ["--inverter-model", "113", "--port", "0", "--soc", "80", "--common-length", "66"]
    )
    cfg = config_from_args(args)
    cfg.validate()
    assert (cfg.inverter_model, cfg.soc_pct, cfg.common_length) == (113, 80.0, 66)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--inverter-model", "101"])


def test_vpp_simulate_sunspec_cli_forwards_to_the_simulator() -> None:
    from click.testing import CliRunner

    from vpp.cli.main import cli

    result = CliRunner().invoke(cli, ["simulate", "sunspec", "--help"])
    assert result.exit_code == 0, result.output
    assert "vpp simulate sunspec" in result.output and "--inverter-model" in result.output
