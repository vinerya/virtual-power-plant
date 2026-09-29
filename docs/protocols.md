# Protocols, V2G and demand response

This page covers the grid- and device-facing integrations: the OCPP 1.6-J
Central System, the OpenADR 2.0b VEN, the IEEE 2030.5 client, the V2G fleet
that ties chargers to vehicles, and the demand-response (DR) orchestrator
that turns grid signals into fleet dispatch. MQTT and Modbus telemetry
ingestion are configured in [configuration.md](configuration.md#telemetry-ingestion-mqtt-modbus).

## SIMULATED vs LIVE

Every adapter has a **mode**:

- **LIVE** — the adapter talks to a real peer: charge points connected over
  WebSocket, a real VTN, a real utility server, broker or device.
- **SIMULATED** — the adapter runs its state machine in memory with no real
  endpoint. Useful for demos, tests and development; nothing leaves the
  process.

`GET /api/v1/protocols` lists every registered adapter with `status`,
`mode` and `simulated`. An adapter without a configured endpoint reports
status `simulated` and is **never** shown as `connected`, so a dashboard
cannot mistake a simulation for control of real equipment. The web console's
`/protocols` page shows the same data with LIVE / SIMULATED badges.

All three grid protocols are **off by default**. Each is started at API
startup when its `VPP_<PROTOCOL>_ENABLED` setting is true, registered in the
protocol registry, and supervised: a peer that is down at startup does not
crash the API; the initial connect is retried with exponential backoff.

Operators and admins can connect/disconnect adapters
(`POST /api/v1/protocols/{name}/connect|disconnect`); any operator-side user
can read status, metrics and protocol data.

## OCPP 1.6-J Central System

**Maturity: beta.** Implements the OCPP-J transport (CALL / CALLRESULT /
CALLERROR framing, unique-id correlation, per-call timeouts, one outstanding
CALL per charge point) and these messages:

| Direction | Messages |
|---|---|
| charge point → VPP | BootNotification, Heartbeat, StatusNotification, Authorize, StartTransaction, StopTransaction, MeterValues, DataTransfer; Diagnostics/FirmwareStatusNotification are acknowledged |
| VPP → charge point | RemoteStartTransaction, RemoteStopTransaction, SetChargingProfile, ClearChargingProfile |

Not implemented: OCPP 2.0.1, Security Profiles 2/3 (client certificates on
the charger side), smart-charging messages beyond the four above
(GetCompositeSchedule, TriggerMessage, ChangeConfiguration, Reset, ...).

### Setup

```bash
VPP_OCPP_ENABLED=true
VPP_OCPP_ALLOWED_CHARGE_POINTS='["CP-001","CP-002"]'   # or leave empty = any id
VPP_OCPP_BASIC_AUTH_PASSWORD=<long random>              # Security Profile 1
VPP_OCPP_AUTHORIZED_ID_TAGS='["RFID-1234"]'             # unset = accept all idTags
```

Point each charger at `wss://<your-host>/ocpp/<charge_point_id>` with
subprotocol `ocpp1.6`. With Basic auth the username is the charge point id.
TLS must be terminated by the reverse proxy (see
[deployment.md](deployment.md#reverse-proxy-and-tls)); Basic auth over plain
`ws://` exposes the password.

The route refuses the connection when the Central System is not running
(close `1013`), when the charger does not offer `ocpp1.6` (`1002`) or when
the allow-list / Basic auth check fails (`1008`).

### Data and actions

- `GET /api/v1/protocols/ocpp/charge-points[/{id}]` — per-connector status
  and latest meter readings.
- `GET /api/v1/protocols/ocpp/transactions`
- `POST /api/v1/protocols/ocpp/charge-points/{id}/remote-start|remote-stop`
  (admin, operator).

`MeterValues` publish `RESOURCE_UPDATED` on the `resource_updates` channel.

### V2G and discharge

OCPP 1.6 has no standard way to command discharge. Discharge setpoints are
sent as **negative** `limit` values in charging profiles, a widespread vendor
extension; a charger that does not support it answers `Rejected`, and that
answer is reported, not hidden.

## V2G fleet

**Maturity: beta.** Vehicles are persisted (`v2g_vehicles`,
`v2g_charging_sessions`, `v2g_schedules`; migration `0007`).

- `POST /api/v1/v2g/vehicles` registers a vehicle (duplicate `ev_id` → `409`),
  with an optional `id_tag`.
- **Binding** a vehicle to a charger connector happens automatically when a
  `StartTransaction` carries the vehicle's `id_tag`, or manually with
  `PUT /api/v1/v2g/vehicles/{ev_id}/binding`.
- `StatusNotification` drives plug-in / unplug (`EV_CONNECTED` /
  `EV_DISCONNECTED` on `grid_events`); `MeterValues` update SOC and power.
- `POST /api/v1/v2g/schedule` and `POST /api/v1/v2g/dispatch` compute
  per-vehicle schedules/setpoints and push them as `SetChargingProfile`
  (`TxProfile` during a transaction, else `TxDefaultProfile`; gaps become
  explicit 0 kW periods; periods merged to `VPP_V2G_MAX_PROFILE_PERIODS`).
  Each vehicle's delivery result is reported honestly as one of
  `accepted`, `rejected`, `not_supported`, `simulated`, `not_connected`,
  `not_bound`, `no_ocpp`, `no_slots`, `skipped`, `error`.

## OpenADR 2.0b VEN

**Maturity: beta.** A Virtual End Node using the simple-HTTP **pull** model:
registration (`oadrQueryRegistration`, `oadrCreatePartyRegistration`),
`oadrPoll` on the VTN-requested interval, `oadrDistributeEvent` parsing
(hardened XML parser), `oadrCreatedEvent` opt-in/opt-out, re-registration
when the VTN asks, implicit cancellation, exponential backoff, optional
mutual TLS.

Not implemented: the push transport, XML signatures, VEN report
registration (`oadrRegisterReport`), a VTN server (the VTN role is
simulation-only).

```bash
VPP_OPENADR_ENABLED=true
VPP_OPENADR_VTN_URL=https://vtn.example.com/OpenADR2/Simple/2.0b
VPP_OPENADR_VEN_NAME=my-vpp
VPP_OPENADR_CERT_PATH=/etc/vpp/certs/ven.crt   # if the VTN requires mTLS
VPP_OPENADR_KEY_PATH=/etc/vpp/certs/ven.key
VPP_OPENADR_CA_PATH=/etc/vpp/certs/vtn-ca.pem
```

Data and actions: `GET /api/v1/protocols/openadr/events[/{id}]`;
`POST /api/v1/protocols/openadr/events/{id}/opt` overrides the opt-in/out
decision (admin, operator).

With several API workers the VEN runs on the `protocol-adapters` lease
holder; the OpenADR, IEEE 2030.5 and `/api/v1/dr/status` endpoints are
answered by it whichever worker receives the request (forwarded through
the database), or `503 leader_unavailable` if it does not answer within 3 s.

## IEEE 2030.5 client

**Maturity: beta.** An HTTPS client with mutual TLS (the device
certificate, as the standard requires) that walks DeviceCapability →
EndDeviceList → EndDevice → FunctionSetAssignmentsList → DERProgramList →
DERControlList (+ DefaultDERControl), handles the server `Time` offset and
`s`/`l` paging, identifies its EndDevice by LFDI/SFDI (configured or
derived from the certificate), re-polls on the server `pollRate`, and
exposes the active DER controls ordered by program primacy.

Responses: for server controls with a `replyTo` link the client POSTs
`DERControlResponse` resources, once per status, per the control's
`responseRequired` bitmap — bit 0: `1` *Event Received*; bit 1: `2` *Event
Started* when the control becomes active, `3` *Event Completed* after its
interval (also when the server has dropped it from its list), `6` *Event
Cancelled* / `7` *Event Superseded*. A failed POST is retried on the next
poll. Simulated mode posts nothing.

Not implemented: posting DERStatus / DERCapability, subscription/
notification, DERCurve parsing.

```bash
VPP_IEEE2030_5_ENABLED=true
VPP_IEEE2030_5_SERVER_URL=https://utility.example.com:8443
VPP_IEEE2030_5_CERT_PATH=/etc/vpp/certs/device.crt
VPP_IEEE2030_5_KEY_PATH=/etc/vpp/certs/device.key
VPP_IEEE2030_5_CA_PATH=/etc/vpp/certs/2030-5-ca.pem
VPP_IEEE2030_5_TLS_CIPHERS=ECDHE-ECDSA-AES128-CCM8
```

Data: `GET /api/v1/protocols/ieee2030_5/controls`.

## Demand-response orchestrator

**Maturity: beta; auto-response off by default.** Implemented in
`src/vpp/dr/` and started when OpenADR or IEEE 2030.5 is enabled:

```
VTN event / utility DERControl
  -> translate (vpp.dr.translate: fleet target kW + limits, safety caps)
  -> DB-backed dispatch (same LP + fallback as POST /api/v1/optimization/dispatch,
     over persisted resources + plugged-in V2G vehicles; recorded as an optimization run)
  -> EV setpoints pushed to chargers via OCPP SetChargingProfile
  -> stationary setpoints written by the setpoint actuator (Modbus; see
     "Device control" below; VPP_CONTROL_ENABLED kill switch)
  -> audit row in dr_event_responses + DR_EVENT_RECEIVED / DR_RESPONSE_SENT /
     DISPATCH_EXECUTED / V2G_DISPATCH / DEVICE_SETPOINT events
```

Status and history: `GET /api/v1/dr/status`, `GET /api/v1/dr/responses`.

### Safety rules

- **Off by default** (`VPP_DR_AUTO_RESPONSE_ENABLED=false`). While off,
  signals are still observed, recorded and published, and the VEN answers
  opt-in/out per `VPP_OPENADR_AUTO_OPT_IN`, but nothing is dispatched.
- When on, a new OpenADR event is opted **in** only if its translated
  target is supported and the fleet's capability over the event window
  covers at least `VPP_DR_MIN_OPT_IN_FRACTION` of the request; otherwise
  **out**. Opted-out, test, cancelled and completed events are never
  dispatched.
- During an event the effective target is dispatched and re-planned when
  the target changes or every `VPP_DR_REDISPATCH_INTERVAL_S` (fresh SOC
  each time). When no signal is active any more, EV DR profiles are cleared,
  device setpoints are released and a release is recorded.
- An IEEE 2030.5 program's **DefaultDERControl** applies while no event
  control is active: its limits always clamp, and its target is dispatched
  unless an OpenADR event supplies one (the event wins; the default's limits
  still apply).
- **IEEE 2030.5 limits always clamp** (they are utility grid-safety
  constraints), and the operator caps `VPP_DR_MAX_EXPORT_KW` /
  `VPP_DR_MAX_IMPORT_KW` are applied last.
- Resource limits are never exceeded: each resource is bounded by its rated
  power and the energy available over the interval; EVs only export when
  V2G-capable, plugged in and flexible with respect to their departure
  target. An unreachable target is reported as a shortfall, not forced.

### Translation rules

Sign convention: fleet `target_kw` is **export-positive** (positive =
deliver to the grid / reduce net load).

OpenADR 2.0b, using the current interval's value of the event's first
signal (values are taken as kW; OpenADR signals carry no unit):

| signalName | signalType | target_kw |
|---|---|---|
| `SIMPLE` | level | `VPP_DR_SIMPLE_LEVEL_FRACTIONS[level]` × fleet export capability |
| `LOAD_DISPATCH` | delta | `+value` (shed / deliver `value` kW) |
| `LOAD_DISPATCH` | setpoint / level | `-value` (value = VEN net-load setpoint; negative = export) |
| `LOAD_CONTROL` | x-loadControlSetpoint | `-value` (net-load setpoint) |
| `LOAD_CONTROL` | other | `+value` (offset) |
| `LOAD_PERCENTAGE`, `ELECTRICITY_PRICE` | – | not translated (needs a metered baseline / belongs to price-driven MPC); recorded only |

IEEE 2030.5 active DERControls, highest-priority program first; for each
field the first control that sets it wins:

| Control | Effect |
|---|---|
| `opModConnect=false` / `opModEnergize=false` | target 0 kW |
| `opModTargetW` | target = W / 1000 |
| `opModFixedW` | target = pct / 100 × setMaxW |
| `opModMaxLimW` | export cap = pct / 100 × setMaxW |
| `opModGenLimW` | export cap = W / 1000 |
| `opModLoadLimW` | absorb cap = W / 1000 |

`setMaxW` is `VPP_DR_IEEE2030_5_SET_MAX_W` when configured, else the fleet's
current export capability. When both protocols are active an IEEE 2030.5
target wins over an OpenADR one.

### What "dispatch" reaches

EV setpoints are sent to chargers over OCPP and each charger's answer is
recorded. **Stationary resources** (batteries, PV inverters) get their
allocation through the setpoint actuator described next: only resources that
opted in are written, and only while `VPP_CONTROL_ENABLED=true`. Every other
resource is reported as `not_configured` / `disabled` in the dispatch record
(`device_deliveries`) and receives nothing.

## Device control (setpoint actuator)

**Maturity: beta; off by default.** `src/vpp/control/actuator.py` consumes
dispatch allocations and writes a power setpoint to each opted-in device;
`src/vpp/protocols/modbus_control.py` is the Modbus writer. Users:

- `POST /api/v1/optimization/dispatch` with `"apply": true` (default `false`
  plans only), valid for `interval_minutes`;
- the DR orchestrator (OpenADR events, IEEE 2030.5 event and default
  controls), valid until the next re-dispatch, released when the signal ends.

`GET /api/v1/optimization/setpoints[?resource_id=]` shows the kill switch,
active setpoints and the latest commands.

### Opting a resource in

Add a `control` block to the resource's existing `metadata.modbus` config
(the same block Modbus ingestion reads). Addresses are the 0-based values
sent on the wire; vendor tables often list 1-based register numbers.

```json
{"modbus": {"mode": "tcp", "host": "192.168.1.50", "port": 502, "unit_id": 1,
  "control": {"enabled": true, "profile": "sunspec_123", "model_base": "auto",
              "revert_timeout_s": 900, "max_kw": 8, "deadband_kw": 0.2,
              "min_interval_s": 10}}}
```

| Profile | Writes | Release | Status |
|---|---|---|---|
| `register` (default) | one signed setpoint register: `register` (a map/custom register flagged `"writable": true`) or `address` + `data_type`; `unit` `W` / `kW` / `pct` (of `reference_kw`, default rated power); `scale` (value of one count) or `scale_factor_register` (SunSpec-style int16 exponent); `sign` `export_positive` (default) / `import_positive`; optional `enable_register` (+`enable_value`, `disable_value`) | `disable_value` to `enable_register`, else `release_value` (default 100 for `pct`, 0 otherwise) | generic |
| `sunspec_123` | SunSpec model 123 Immediate Controls at `model_base` (address of the model `ID`, or `"auto"`): `WMaxLimPct` (+5, scaled by `WMaxLimPct_SF` at +23) = setpoint / `reference_kw` (the spec's % of `WMax`, so set `reference_kw` to `WMax`), clamped to 0 %; `WMaxLim_Ena` (+9) = 1; optional `WMaxLimPct_RvrtTms` (+7) = `revert_timeout_s`. A setpoint of 100 % or more is no limit: only `WMaxLim_Ena` = 0 is written (the limit is lifted) | `WMaxLim_Ena` = 0 | offsets verified against the SunSpec model definition; untested on hardware |
| `sunspec_124` | SunSpec model 124 Storage: `OutWRte` (+12) / `InWRte` (+13) as % of `reference_kw` (spec: % of `WDisChaMax` / `WChaMax`; discharge: `OutWRte`=p, `InWRte`=−p; charge the reverse), `StorCtl_Mod` (+5, bitfield: CHARGE + DISCHARGE) = 3 | `StorCtl_Mod` = 0 | offsets verified against the SunSpec model definition; forced charge/discharge **semantics unverified** — vendors interpret them differently; validate on your device |

No vendor-specific absolute control addresses are shipped: SunSpec models
sit at a device-specific address that depends on the models before them.
`"model_base": "auto"` finds the model by SunSpec discovery (the `"SunS"`
marker at 40000, 0 or 50000, then the chain of `ID`/`L` model headers up to
the end model `0xFFFF`); an explicit `model_base` is the 0-based address of
the model's `ID` register. Either way the `ID`/`L` header is read back before
the first write and a mismatch is refused, so a wrong address never gets
written.

**Register maps.** The SunSpec registers shipped in
`src/vpp/protocols/modbus.py` (model 123 and 124 control blocks, the
`fronius_symo` model 113 and `solaredge_se` model 101/103 read maps) are
verified against the official SunSpec model definitions — the
sunspec/models JSON bundled with pysunspec2 1.3.6 — point by point
(offset, size, type, units, access, scale-factor pairing) by
`tests/test_sunspec_models.py`. They have **not** been tested on physical
hardware. The Fronius/SolarEdge read maps assume the inverter model's `ID`
at address 40069 (common model with L = 65, as both vendors document);
polling multiplies values by their `*_SF` scale factor and drops SunSpec
"not implemented" values (0x8000, 0xFFFF, 0x80000000, 0xFFFFFFFF, acc32 0,
float NaN). A value and its scale factor always come from the **same read
request**: the polled registers of one SunSpec model are read as one
contiguous block (e.g. `solaredge_se`: model 103 offsets 14–37, one request
of 24 registers per poll), and any other value is read together with its
scale-factor register as the minimal span covering both (at most 125
registers). Reading them separately let a device that rescales between two
requests produce a value off by a power of ten. If a device rejects a block
read, that poll falls back to one request per value + scale-factor pair;
registers without a scale factor outside SunSpec models (e.g.
`sma_sunnyboy`, custom maps with gaps) are still read one by one. The
`sma_sunnyboy` map uses SMA's proprietary registers, not
SunSpec, and is not covered by that check.
`"simulate": true` runs the whole pipeline without device I/O.

### Safety rules

1. `VPP_CONTROL_ENABLED=false` (default) writes nothing (`disabled`);
   `control.enabled` must also be true.
2. Offline resources are never written (`offline`), including by the
   watchdog.
3. Setpoints are clamped to the resource limits (battery: −charge ..
   +discharge limit; others: 0 .. rated power) and `min_kw` / `max_kw`.
4. Deadband (`deadband_kw`, default 0.1): a near-identical setpoint is not
   rewritten (`unchanged`); rate limit (`min_interval_s`, default 5 s): a
   newer setpoint is held and written by the watchdog (`deferred`).
5. Read-back verification (`verify`, default true): a register that does not
   read back as written is a failure. A write never exceeds the register's
   type (no wrap-around), and only registers flagged writable are written.
6. Watchdog (`VPP_CONTROL_WATCHDOG_INTERVAL_S`): a setpoint expires at the
   end of its dispatch interval plus `VPP_CONTROL_EXPIRY_GRACE_S`; the device
   then goes to `safe_setpoint_kw` if configured, else control is released.
   On API shutdown every active setpoint is released. With
   `revert_timeout_s` the device reverts on its own if the process dies;
   the watchdog refreshes the setpoint every `keepalive_s` (default half the
   revert timeout). A failed release is retried three times.

### Curtailed generation and its availability

A PV (or other generation) resource's available power for dispatch comes
from its polled output. Once the VPP has capped that output, the polled
value is the cap, not what the panels could deliver, and taking it as the
availability would keep the optimiser from ever allocating more — the
limit could only go away when the device's revert timer fired. So:

- The actuator tracks, per resource, the output cap it currently holds: a
  setpoint below the resource's upper limit that the device accepted (read
  back when `verify` is on). Simulated control blocks and batteries are
  not tracked. The cap's start time is kept across rewrites of a
  continuous limit.
- While such a cap is in force and the polled output is at it (within
  max(0.05 kW, 2 % of rated power)), the optimiser does not trust the
  reading. It estimates the availability as the newest telemetry reading
  taken **before** the cap started, if that is at most 15 minutes old
  (`availability_basis` `pre_curtailment_telemetry`; never less than the
  current output), else the nameplate rating (`nameplate_curtailed`, an
  upper bound). A reported `available_kw` in the resource's metadata or
  config still takes precedence, and output clearly below the cap is plain
  `telemetry`. The dispatch response reports the basis per allocation; the
  run's asset summary also records `output_limit_kw`.
- When a generation resource is allocated all of its estimated
  availability, it is not to be curtailed: the actuator writes the
  resource's upper limit instead of the estimate (delivery field
  `"uncurtailed": true`), which for `sunspec_123` lifts the limit
  (`WMaxLim_Ena` = 0). Capping at the estimate would pin the output there
  again, since the next poll could never show more.

Limitations: the estimate is only as good as the last uncurtailed reading
(sun or wind may have changed since, and after 15 minutes of curtailment the
nameplate is used). No SunSpec point the project reads gives the real
availability: model 103/113 report the AC and DC output, which both follow
the cap, and model 123 only the limit itself (model 121's `WMax` and
vendor-specific "available power" points are not read). Forecasts are not
used. The cap state lives in the API process that wrote it: another worker,
or a restarted process, does not know it (on a clean shutdown every cap is
released; after a crash the device's revert timer applies).

Every command is recorded in the `event_log` table (`event_type =
device_setpoint`), stored with the dispatch (`device_deliveries` in the run
metadata / DR response details) and published as a `DEVICE_SETPOINT` event
(WebSocket channel `optimization_events`). Delivery statuses: `accepted`,
`unchanged`, `deferred`, `simulated`, `failed`, `offline`, `disabled`,
`not_configured`.

## Testing without hardware

`vpp simulate sunspec` (same as `python -m vpp.simulators.sunspec`; needs
the `protocols` extra for pymodbus) serves a simulated PV + battery hybrid
inverter over Modbus TCP, so polling, SunSpec discovery and device control
can be exercised end to end without a device.

**What it serves** (defaults, 0-based wire addresses; `--help` lists the
options):

| Address | Model | Contents |
|---|---|---|
| 40000 | `"SunS"` | marker |
| 40002 | 1 common, L = 65 | `Mn` "VPP Simulator", `Md`, `Vr`, `SN` (`--serial`), `DA` |
| 40069 | 103 three-phase inverter, L = 50 (`--inverter-model 113`: float32, L = 60) | `W`, `Hz`, `A`, `PhV`, `WH`, `DCW`, temperatures, `St` … with scale factors (`W_SF` 0, `Hz_SF` −2, `Tmp_SF` −1, …) |
| 40121 | 123 immediate controls, L = 24 | `WMaxLimPct` (`WMaxLimPct_SF` −1), `WMaxLim_Ena`, `WMaxLimPct_RvrtTms`, `Conn`, … |
| 40147 | 124 storage, L = 24 | `WChaMax`, `StorCtl_Mod`, `ChaState` (`ChaState_SF` −1), `OutWRte` / `InWRte` (`InOutWRte_SF` −2), `InOutWRte_RvrtTms`, `ChaGriSet`, … |
| 40173 | end | `0xFFFF`, L = 0 |

(With model 113 the controls start at 40131 and storage at 40157.) The
common model is served with L = 65, as SolarEdge and Fronius devices report
it, so the `solaredge_se` (model 103) and `fronius_symo` (model 113) polling
maps line up; `--common-length 66` serves the spec's full model 1 (with
`Pad`), which moves every later model by one register — use
`"model_base": "auto"` then.

**How it behaves.** PV output is `--pv` watts (default 6000) unless
curtailed: with `WMaxLim_Ena` = 1 the AC output is capped at
`WMaxLimPct` % of `--wmax` (default 10000 W). The battery idles unless
`StorCtl_Mod` activates the limits: its power is 0 clamped into
[−`OutWRte`, +`InWRte`] % of `WChaMax`, so `OutWRte` = p, `InWRte` = −p
discharges at p % and the reverse charges — the interpretation the
`sunspec_124` profile writes. SoC integrates battery power (95 % one-way
efficiency) between the `MinRsvPct` reserve and 100 %. `*_RvrtTms` revert
timers run from the last write to that control (limit disabled,
`StorCtl_Mod` cleared, `Conn` restored). Writes to read-only points answer
ILLEGAL DATA ADDRESS and out-of-range values (`WMaxLimPct` > 100 %,
`StorCtl_Mod` > 3, rates beyond ±100 %) ILLEGAL DATA VALUE (pymodbus 3.10
and later; 3.6–3.9 drop a rejected write silently, which read-back shows). Points it does not model
hold the SunSpec "not implemented" values. `--speed 60` runs one simulated
minute per second, so SoC changes are visible quickly.

**What it does not prove.** The simulator is written from the SunSpec
specification. Passing against it shows that the VPP implements the
specification consistently from register map to dispatch; it does not show
how any vendor's firmware behaves. Real devices differ exactly where it
matters for control: how forced charge/discharge through model 124 is
interpreted (the reason `sunspec_124` is flagged unverified), which points
are implemented or writable, ramp and window times (stored here, not
modelled), set points that do not persist or need a vendor unlock, unit IDs,
and scale factors that change at run time. Validate against the real device
before enabling control on it.

### Automated test

`tests/test_sunspec_simulator.py` starts the simulator in-process on an
ephemeral port and checks, among the device-model unit tests:
discovery finds models 1 / 103 (or 113) / 123 / 124; polled values come out
in engineering units with the served scale factors; and a
`POST /api/v1/optimization/dispatch` with `"apply": true` goes through the
real actuator and Modbus writer, changes the simulator's registers
(`WMaxLimPct` 400 = 40.0 %, `OutWRte` / `InWRte` / `StorCtl_Mod`), and the
next poll by the ingestion loop shows the new output and SoC. After the PV
is curtailed to 40 %, a dispatch asking for full output estimates its
availability from the reading before the cap, lifts the limit
(`WMaxLim_Ena` = 0), and the next poll shows the restored 6 kW. A further
test flips the served `W_SF` after every request and checks that polled
values stay consistent (one block read per poll).

```bash
python -m pytest tests/test_sunspec_simulator.py -q
```

### By hand, against a running API

```bash
# Terminal 1: the device (Ctrl-C to stop)
vpp simulate sunspec --port 5020 --speed 10

# Terminal 2: the API with Modbus polling and device control, one worker
export VPP_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
export VPP_MODBUS_INGESTION_ENABLED=true VPP_CONTROL_ENABLED=true
vpp migrate
vpp users create-admin admin
vpp serve --workers 1
```

Create a PV and a battery resource pointing at the simulator (the battery
polls `ChaState` through `custom_registers` and records it with
`soc_register`, which is where the optimiser reads its state of charge):

```bash
PW='<the admin password chosen above>'
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token -d username=admin -d password="$PW" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
AUTH="Authorization: Bearer $TOKEN"

curl -s -X POST localhost:8000/api/v1/resources/ -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "name": "sim-pv", "resource_type": "solar", "rated_power": 10,
  "metadata": {"modbus": {"host": "127.0.0.1", "port": 5020, "unit_id": 1,
    "device_profile": "solaredge_se", "poll_interval_s": 2,
    "control": {"enabled": true, "profile": "sunspec_123", "model_base": "auto",
                "reference_kw": 10, "revert_timeout_s": 300}}}}'

curl -s -X POST localhost:8000/api/v1/resources/ -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "name": "sim-battery", "resource_type": "battery", "rated_power": 5,
  "metadata": {"capacity_kwh": 10, "modbus": {"host": "127.0.0.1", "port": 5020, "unit_id": 1,
    "device_profile": "sunspec_storage", "poll_interval_s": 2,
    "soc_register": "cha_state",
    "custom_registers": {
      "cha_state": {"address": 40155, "data_type": "uint16", "unit": "%",
                    "scale_factor": "cha_state_sf",
                    "sunspec": {"model": 124, "base": 40147, "point": "ChaState"}},
      "cha_state_sf": {"address": 40169, "data_type": "int16",
                       "sunspec": {"model": 124, "base": 40147, "point": "ChaState_SF"}}},
    "control": {"enabled": true, "profile": "sunspec_124", "model_base": "auto",
                "reference_kw": 5, "revert_timeout_s": 300}}}}'
```

Modbus resources are discovered when the API starts: restart `vpp serve`,
then dispatch (use the ids returned above) and watch the simulator's status
lines:

```bash
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"target_power_kw": 4, "resource_ids": ["<sim-pv id>"], "apply": true}'
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"target_power_kw": -2, "resource_ids": ["<sim-battery id>"], "apply": true}'
curl -s localhost:8000/api/v1/optimization/setpoints -H "$AUTH"
```

The simulator's status lines then show the set points taking effect, e.g.
`ac=4000W pv=6000/6000W battery=-2000W soc=50.3% limit=40.0% stor_ctl=3`
(PV curtailed to a 4 kW export cap while it charges the battery at 2 kW).
`--speed` also speeds up the device's revert timers but not the VPP's
keep-alive refresh, so with `--speed 10` a 300 s `revert_timeout_s` fires
after 30 real seconds; use `--speed 1` to watch keep-alive and revert.

**Docker.** `docker compose --profile sim up -d` also starts `sunspec-sim`
(Modbus TCP on the compose network as `sunspec-sim:5020`, published on
127.0.0.1:5020). Use `"host": "sunspec-sim"` in the resources above and set
`VPP_MODBUS_INGESTION_ENABLED=true VPP_CONTROL_ENABLED=true` for `vpp-api`
(the compose file passes both through; control needs the single default
worker).

