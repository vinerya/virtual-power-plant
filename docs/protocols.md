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

## IEEE 2030.5 client

**Maturity: beta.** An HTTPS client with mutual TLS (the device
certificate, as the standard requires) that walks DeviceCapability →
EndDeviceList → EndDevice → FunctionSetAssignmentsList → DERProgramList →
DERControlList (+ DefaultDERControl), handles the server `Time` offset and
`s`/`l` paging, identifies its EndDevice by LFDI/SFDI (configured or
derived from the certificate), re-polls on the server `pollRate`, and
exposes the active DER controls ordered by program primacy.

Not implemented: posting DERStatus / DERCapability / Response resources
back to the server, subscription/notification, DERCurve parsing.

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
  -> audit row in dr_event_responses + DR_EVENT_RECEIVED / DR_RESPONSE_SENT /
     DISPATCH_EXECUTED / V2G_DISPATCH events
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
  each time). When no signal is active any more, EV DR profiles are cleared
  and a release is recorded.
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

EV setpoints are sent to chargers and each charger's answer is recorded.
**Stationary resources** (batteries, PV, wind) receive their allocation
only as a `DISPATCH_EXECUTED` event on the event bus. The platform does not
ship a driver that writes those setpoints to devices (Modbus is used for
reading telemetry); connect your own device integration to that event
before enabling auto-response for stationary assets.
