# Connecting real data and devices

This guide is for someone who wants to run the platform against a real
fleet: an integrator at a utility or aggregator, or a homeowner with a
battery, rooftop PV (photovoltaics, solar panels) and an EV charger. It
walks through the whole path in order:

1. [Choose a deployment](#1-prerequisites-and-deployment)
2. [Create accounts and an API key](#2-accounts-and-access)
3. [Describe your fleet](#3-model-your-fleet) (sites and resources)
4. [Bring in telemetry](#4-bring-in-telemetry) (HTTP, MQTT, Modbus, history)
5. [Add prices and tariffs](#5-prices-and-tariffs)
6. [Connect grid signals](#6-grid-signals-and-markets) (OpenADR, IEEE 2030.5)
7. [Connect EV chargers](#7-ev-chargers-ocpp-16-j) (OCPP)
8. [Verify before you dispatch](#8-verify-before-you-dispatch)
9. [Turn on device control safely](#9-turn-on-device-control-safely)
10. [Operate it](#10-operate)
11. [Troubleshoot](#11-troubleshooting)

!!! warning "Beta software"

    The project is **beta**. Nothing here has been certified against a
    protocol conformance suite or proven in a field deployment. The SunSpec
    register maps are checked against the SunSpec specification but have
    **not** been tested on physical inverters. Device control is **off by
    default**, and stays off until you turn it on for the platform and for
    each device. Read [step 9](#9-turn-on-device-control-safely) before you
    let the platform write to anything.

Every command below was run against a fresh install. The examples use the
default ports: API on `localhost:8000`, MQTT broker on `localhost:1883`,
SunSpec simulator on `localhost:5020`. Change them to match your setup.

## 1. Prerequisites and deployment

**Install** the package with the extras you need
([Installation](getting-started/installation.md)). For real devices you
want at least `api`, `db`, `cli` and `protocols` (pymodbus for Modbus,
paho-mqtt for MQTT); `solver` gives the optimizer its LP (linear
programming) solver.

**Choose a database.**

| | SQLite (default) | PostgreSQL |
|---|---|---|
| `VPP_DATABASE_URL` | `sqlite+aiosqlite:///./vpp.db` | `postgresql+asyncpg://user:pass@host:5432/vpp` |
| Good for | one home, a lab, a pilot of a few devices | anything larger, several API workers |
| API workers | one | one or more |

[Performance and capacity](performance.md#when-to-move-from-sqlite-to-postgresql)
says where SQLite stops being enough.

**Run one API worker when you control devices.** Device control
(`VPP_CONTROL_ENABLED`) and the OCPP Central System (`VPP_OCPP_ENABLED`)
keep state inside one process. `vpp serve` refuses to start with more than
one worker when either is on. One worker is the default.

**Production basics.** Before the API is reachable from anything but
localhost:

```bash
export VPP_ENV=production
export VPP_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
export VPP_DATABASE_URL=postgresql+asyncpg://vpp:<password>@db-host:5432/vpp
vpp migrate
vpp serve
```

- With `VPP_ENV=production` the API refuses to start unless
  `VPP_SECRET_KEY` is set to at least 32 characters. The key signs login
  tokens; keep it secret and stable (changing it logs everyone out).
- `vpp serve` listens on `127.0.0.1:8000`. Put a reverse proxy (nginx,
  Caddy, Traefik) in front of it that terminates TLS (HTTPS). The
  [deployment guide](deployment.md#reverse-proxy-and-tls) has an nginx
  example, including the WebSocket and OCPP paths.
- Set `VPP_TRUSTED_PROXIES` to the address of that proxy (and of the web
  console server), for example `VPP_TRUSTED_PROXIES='["10.0.0.5"]'`.
  Only then does the rate limiter use the client address from
  `X-Forwarded-For`; otherwise every client behind the proxy shares one
  limit ([details](deployment.md#rate-limiting-behind-the-console)).

Every setting is a `VPP_*` environment variable (or a line in `.env`); the
[configuration reference](configuration.md) lists them all.

## 2. Accounts and access

There is no self-registration. Create the first admin from the command
line (it reads the password from `$VPP_ADMIN_PASSWORD`, `--password-file`,
`--password-stdin` or a prompt):

```bash
export VPP_ADMIN_PASSWORD='pick-a-long-passphrase'
vpp users create-admin admin
```

People sign in to the web console with a username and password. Scripts,
gateways and other integrations should use an **API key** instead. Log in
once as the admin and create one:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -d username=admin -d password="$VPP_ADMIN_PASSWORD" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -s -X POST localhost:8000/api/v1/auth/api-key \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name": "telemetry-bridge", "role": "operator"}'
# {"id": "...", "name": "telemetry-bridge", "key": "vpp_zsxE...", "role": "operator", ...}
```

The `key` is shown **once**; only a hash is stored. Send it on every
request in the `X-API-Key` header (the header name is the
`VPP_API_KEY_HEADER` setting):

```bash
export VPP_KEY='vpp_...'          # the key from the response above
curl -s localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY"
```

The rest of this guide uses `-H "X-API-Key: $VPP_KEY"`. An
`Authorization: Bearer $TOKEN` header works the same way, but tokens expire
after `VPP_JWT_EXPIRE_MINUTES` (60 by default).

**Which role?** A key acts with the lesser of its own role and its
creator's current role.

| Role | Use it for |
|---|---|
| `operator` | anything that writes: telemetry, meter readings, resources, sites, applying dispatch to devices |
| `viewer` | dashboards and reports: reads everything, can run a dispatch dry-run, cannot write telemetry or touch devices |
| `admin` | only for people managing users, tariffs and configuration; avoid giving it to integrations |

Give each integration its own key, so you can revoke one
(`DELETE /api/v1/auth/api-keys/{id}`) without breaking the others. In the
console, users and keys are under **Settings → Users & API keys**. More in
the [API guide](api.md#authentication).

## 3. Model your fleet

A **resource** is one device the optimizer can dispatch. A **site** groups
resources at one location and carries the location and time zone.

!!! note "The console shows the fleet but does not create it"

    The web console lists sites (**Sites**) and resources (**Assets**) with
    live values, but has no forms to create them. Create them through the
    API: with `curl` as below, or interactively at
    `http://localhost:8000/docs`.

### Resource types

`POST /api/v1/resources` accepts three `resource_type` values. Unknown
fields are rejected with `422`, so a typo never disappears silently.

| `resource_type` | Device |
|---|---|
| `battery` | stationary battery (home battery, BESS) |
| `solar` | PV inverter |
| `wind_turbine` (or `wind`) | wind turbine |

There is **no resource type for an EV, an EV charger or a load**:

- EVs are registered as V2G vehicles and reach the optimizer through the
  OCPP charger they are plugged into ([step 7](#7-ev-chargers-ocpp-16-j)).
- Site consumption comes in as interval meter data
  ([meter readings](#meter-readings-and-history)), used for bills and
  tariff studies. It is not a dispatchable resource.

### Fields that matter for dispatch

Power is in **kW**, energy in **kWh**, state of charge (SoC) and
efficiencies are **fractions from 0 to 1**.

| Field | Types | What the optimizer does with it |
|---|---|---|
| `rated_power` (required) | all | upper bound of output (and of charging, for batteries) |
| `capacity_kwh` | battery | energy the battery can deliver over the dispatch interval. Without it the optimizer assumes `rated_power × 4` kWh |
| `max_charge_kw`, `max_discharge_kw` | battery | power limits; default `rated_power`, cannot exceed it |
| `soc_min`, `soc_max` | battery | SoC window kept by the optimizer (default 0.05 – 0.95) |
| `efficiency` | all | one-way efficiency per leg for batteries (default 0.95) |
| `charge_efficiency`, `discharge_efficiency` | battery | override `efficiency` per leg |
| `state_of_charge` (or `current_charge_kwh`) | battery | SoC used **until telemetry arrives**; then the newest telemetry SoC wins |
| `chemistry` | battery | `lfp` or `nmc`: selects the degradation model behind the wear cost |
| `dc_capacity_kw`, `ac_capacity_kw`, `panel_area_m2`, `panel_efficiency` | solar | stored and shown; dispatch uses the measured output |
| `rotor_diameter_m`, `hub_height_m`, `cut_in_speed_ms`, `rated_speed_ms`, `cut_out_speed_ms` | wind | stored and shown; dispatch uses the measured output |
| `metadata` | all | free-form JSON. The `modbus` key inside it configures polling and control ([step 4](#modbus-tcp-and-sunspec)) |

A battery:

```bash
curl -s -X POST localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "name": "home-battery",
  "resource_type": "battery",
  "rated_power": 5,
  "capacity_kwh": 13.5,
  "max_charge_kw": 5,
  "max_discharge_kw": 5,
  "soc_min": 0.1,
  "soc_max": 0.95,
  "efficiency": 0.95,
  "chemistry": "lfp",
  "state_of_charge": 0.5
}'
```

A PV inverter and a wind turbine:

```bash
curl -s -X POST localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "name": "rooftop-pv",
  "resource_type": "solar",
  "rated_power": 10,
  "dc_capacity_kw": 11,
  "ac_capacity_kw": 10
}'

curl -s -X POST localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "name": "farm-wind",
  "resource_type": "wind",
  "rated_power": 50,
  "rotor_diameter_m": 20,
  "hub_height_m": 30,
  "cut_in_speed_ms": 3,
  "rated_speed_ms": 11,
  "cut_out_speed_ms": 25
}'
```

Each response contains the resource's `id` (a UUID). Telemetry, dispatch
and every other call refer to resources by this id; names must be unique
but are only labels. Change a resource with `PUT /api/v1/resources/{id}`
(send only the fields to change; `metadata` is replaced as a whole).

### Generation availability

Dispatch plans a generation resource (solar, wind) to export at most its
**availability**. Each allocation in the dispatch response, and each
resource in the stored run's inputs, says where that number came from in
`availability_basis`. The rules, in order:

| Situation | Availability | `availability_basis` |
|---|---|---|
| `available_kw` set in the resource's `metadata` (or config) | that value | `reported` |
| the VPP is curtailing the resource and its output is at the cap | newest reading taken before the cap (at most 15 min old), else `rated_power` ([details](protocols.md#curtailed-generation-and-its-availability)) | `pre_curtailment_telemetry` / `nameplate_curtailed` |
| fresh telemetry: the newest sample is at most `VPP_GENERATION_TELEMETRY_MAX_AGE_S` (300 s) old | the measured output `current_power`. **0 kW at night means 0 kW available** | `telemetry` |
| no telemetry, or only stale telemetry | depends on `VPP_GENERATION_AVAILABILITY_FALLBACK` (below) | `no_data`, `forecast` or `nameplate_fallback` |

Freshness is judged by the timestamp of the newest stored telemetry sample
(HTTP endpoint or Modbus polling), not by when `current_power` was last
changed. A `current_power` typed in by hand, without telemetry, is not
trusted.

`VPP_GENERATION_AVAILABILITY_FALLBACK` decides what an unobserved
generator is worth:

- `zero` (default): 0 kW (`no_data`). Dispatch never plans export that
  nobody has seen the resource produce.
- `forecast`: the current point of a forecast you store on the resource,
  else 0 kW (`no_data`). The platform does not compute forecasts itself;
  push yours (from a weather service or your own model) into the
  resource's `metadata` as `generation_forecast`, a list of points:

    ```json
    {"generation_forecast": [
      {"timestamp": "2026-10-10T12:00:00Z", "power_kw": 6.5},
      {"timestamp": "2026-10-10T13:00:00Z", "power_kw": 7.1}
    ]}
    ```

    The newest point at or before now is used if it is at most one hour
    old, capped at `rated_power` (`forecast`). Remember that `metadata` is
    replaced as a whole by `PUT /api/v1/resources/{id}`.
- `nameplate`: `rated_power` (`nameplate_fallback`). This was the
  behaviour up to 3.0 (then labelled `nameplate`, and also applied to a
  measured 0 kW); choose it only if you knowingly rely on it.

### Sites

A site needs a name, latitude and longitude. Give it the IANA time zone of
its location: billing periods and time-of-use (TOU) hours are computed in
it. Attach resources by id:

```bash
curl -s -X POST localhost:8000/api/v1/sites -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "name": "Smith household",
  "lat": 37.77,
  "lon": -122.42,
  "region": "CA-north",
  "timezone": "America/Los_Angeles",
  "resource_ids": ["<home-battery id>", "<rooftop-pv id>"]
}'
```

`PATCH /api/v1/sites/{id}` with `resource_ids` replaces the list. A
resource can belong to one site only (`409` otherwise). To let a household
see its site in the customer portal, create a customer account
(`POST /api/v1/customers`, admin) and set the site's `owner_id` to that
account's id.

## 4. Bring in telemetry

The optimizer needs two things from each device: its current **power** and,
for batteries, its **state of charge**. There are three ways in. You can
mix them; each resource can use a different one.

| Path | Best for | Updates live power | Stores SoC |
|---|---|---|---|
| [HTTP](#http-telemetry-endpoint) | your own gateway or SCADA bridge, cloud APIs, backfills | yes | yes |
| [MQTT](#mqtt) | battery gateways that already publish to a broker | yes, with `power` | yes (batteries only) |
| [Modbus TCP / SunSpec](#modbus-tcp-and-sunspec) | inverters and meters on your LAN | yes | yes, with `soc_register` |

Timestamps are stored in UTC. Over HTTP a timestamp without a time zone
is read as UTC; over MQTT it is rejected.

### HTTP telemetry endpoint

`POST /api/v1/resources/{id}/telemetry`, role `operator` or `admin`:

```bash
curl -s -X POST localhost:8000/api/v1/resources/<resource id>/telemetry \
  -H "X-API-Key: $VPP_KEY" -H 'Content-Type: application/json' -d '{
  "source": "scada",
  "samples": [
    {"timestamp": "2026-10-10T12:00:00Z", "power_kw": 28.0},
    {"power_kw": 31.5}
  ]
}'
# {"resource_id": "...", "accepted": 2, "current_power": 31.5}
```

| Field | Meaning |
|---|---|
| `samples` | 1 to 5,000 samples per request |
| `samples[].power_kw` | required, kW. Sign: **positive = consuming or charging, negative = discharging**. Generators report their output as a positive number |
| `samples[].state_of_charge` | optional, battery SoC as a **fraction 0–1** (not percent) |
| `samples[].timestamp` | optional ISO 8601; default: now. More than 5 minutes in the future is rejected (`422`) |
| `source` | optional label stored with the samples (default `api`) |

The newest sample becomes the resource's live `current_power` only if it is
at most 15 minutes old. Older samples are stored as history and leave the
live value alone, so a backfill never overwrites the present.

**Rate limit.** Each client address may make `VPP_RATE_LIMIT_REQUESTS_PER_MINUTE`
requests per minute (120 by default) across the whole API; beyond that the
answer is `429`. Batch samples (up to 5,000 per request) rather than
sending one request per reading, or raise the limit for your gateway.

### MQTT

The API can subscribe to an MQTT broker (Mosquitto, EMQX, HiveMQ...) and
store **battery** telemetry published there. It is off by default:

```bash
export VPP_MQTT_INGESTION_ENABLED=true
export VPP_MQTT_BROKER_HOST=localhost
export VPP_MQTT_BROKER_PORT=1883
export VPP_MQTT_USERNAME=vpp          # if the broker requires login
export VPP_MQTT_PASSWORD='...'
```

**TLS.** To reach the broker over TLS (usually port 8883), turn it on and,
unless the broker's certificate is signed by a CA in the system trust
store, name the CA file. Brokers that require client certificates
(mutual TLS) also get a certificate and key:

```bash
export VPP_MQTT_BROKER_PORT=8883
export VPP_MQTT_TLS_ENABLED=true
export VPP_MQTT_TLS_CA_FILE=/etc/vpp/mqtt-ca.pem        # omit for a public CA
export VPP_MQTT_TLS_CERTFILE=/etc/vpp/mqtt-client.pem   # mutual TLS only
export VPP_MQTT_TLS_KEYFILE=/etc/vpp/mqtt-client.key   # unencrypted PEM
```

The broker's certificate and host name are always checked.
`VPP_MQTT_TLS_INSECURE=true` turns both checks off (the log warns at every
connect); use it only to test against a broker with a throwaway
certificate. A TLS setting without `VPP_MQTT_TLS_ENABLED=true` stops the
API at startup instead of silently connecting in plain text.

**Client id.** Each API process connects as
`vpp-<hostname>-<pid>-<random>`, so several VPP installations can share one
broker. The id is fixed for the life of the process and changes on
restart, which is safe because the API uses a clean MQTT session: the
broker does not keep messages for it while it is away (see below). Set
`VPP_MQTT_CLIENT_ID` only if your broker's access rules match on client
id, and then give each installation its own: two clients with the same id
disconnect each other over and over. Some strict MQTT 3.1.1 brokers
limit ids to 23 characters; the default can be longer, so set an explicit
short id for those.

Restart the API. `GET /api/v1/protocols/` then lists an adapter named
`mqtt` with `"status": "connected"`.

**Topic.** Publish to

```text
vpp/{site_id}/battery/{resource_id}/{metric}
```

- exactly five segments, starting with `vpp`;
- the third segment must be `battery`. Other resource types are ignored;
- `{resource_id}` is the VPP resource id (the UUID from step 3) of a
  `battery` resource. Messages for unknown ids are dropped with a warning
  in the log;
- `{site_id}` and `{metric}` can be any non-empty text; they are not
  checked. Using the site id and a name such as `state` keeps topics
  readable.

The API subscribes to `VPP_MQTT_TOPIC_PREFIX` (default `vpp/#`) with QoS
(quality of service) 1. Narrowing the filter (for example `vpp/site-a/#`)
works; topics outside `vpp/...` are never stored.

**Payload.** A JSON object. `soc` is required; a message without it is
ignored.

```json
{"soc": 72.5, "power": 2.0, "temperature": 31.0, "voltage": 51.2, "current": 39.0, "soh": 98.0,
 "timestamp": "2026-10-10T12:00:00Z"}
```

| Key | Unit |
|---|---|
| `soc` | state of charge in **percent** (0–100). Values of 1 or less are read as a fraction, so send percent and avoid ambiguity below 1 % |
| `soh` | state of health, percent (default 100) |
| `temperature` | °C (default 25) |
| `voltage`, `current` | V, A |
| `power` | kW, same sign as the HTTP endpoint (positive = charging). Stored as history and, from a recent sample, becomes the resource's live `current_power` |
| `timestamp` | optional time the sample was taken: an ISO 8601 string **with** `Z` or an offset (`2026-10-10T12:00:00Z`, `2026-10-10T14:00:00+02:00`), or a JSON number of **seconds** since 1970-01-01 UTC (`1791633600`, fractions allowed) |
| `timestamp_ms` | alternative to `timestamp`: a JSON number of **milliseconds** since 1970-01-01 UTC. Send one or the other, not both |

Example with `mosquitto_pub`:

```bash
mosquitto_pub -h localhost -q 1 \
  -t "vpp/<site id>/battery/<home-battery id>/state" \
  -m '{"soc": 72.5, "power": 2.0, "temperature": 31.0}'
```

**Timestamps and live power** follow the same rules as the
[HTTP endpoint](#http-telemetry-endpoint):

- Without `timestamp`/`timestamp_ms` a sample is stamped with the time the
  API received it.
- A timestamp without a time zone, a number inside a string
  (`"1791633600"`) or anything else unreadable is rejected, and so is a
  sample more than 5 minutes in the future. The message is dropped and
  the log says why (`MQTT telemetry on topic=... dropped: ...`).
  Milliseconds sent in `timestamp` read as a date thousands of years ahead
  and are rejected this way, so the units cannot be mixed up silently.
- A sample at most 15 minutes old is live: if it has `power`, the newest
  such sample per resource becomes the resource's `current_power`, and
  the sample is broadcast to WebSocket clients (`resource_updates`) and to
  alert rules. An older sample is stored as history only and changes
  nothing live, so a gateway that uploads its buffer after an outage
  backfills history without overwriting the present.

What to know before you rely on it:

- Messages are written in batches of up to 500 per database transaction.
  If ingestion falls behind, messages are dropped and the log says
  `MQTT message queue full`.
- The API subscribes with a **clean session**: messages published while
  the API is stopped or disconnected are not queued by the broker and are
  lost. Have gateways buffer and resend with `timestamp`, or use the HTTP
  endpoint for backfills.
- Only one API worker subscribes (the holder of the `mqtt-ingestion`
  lease), so several workers do not store a message twice.
- A broker that refuses the login (wrong username or password, not
  authorized) is logged as `MQTT broker ... refused the connection` and
  retried every 30 seconds.

### Modbus TCP and SunSpec

The API can poll inverters, batteries and meters over Modbus TCP (or
Modbus RTU on a serial line). Turn polling on:

```bash
export VPP_MODBUS_INGESTION_ENABLED=true
```

Then describe the connection in the resource's `metadata.modbus`:

```bash
curl -s -X PUT localhost:8000/api/v1/resources/<rooftop-pv id> -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "metadata": {"modbus": {
    "mode": "tcp", "host": "192.168.1.50", "port": 502, "unit_id": 1,
    "device_profile": "solaredge_se", "poll_interval_s": 5
  }}
}'
```

| Key | Meaning |
|---|---|
| `mode` | `tcp` (default) or `rtu` (then `serial_port`, default `/dev/ttyUSB0`, and `baudrate`, default 9600) |
| `host`, `port` | device address; port 502 is the Modbus default |
| `unit_id` | Modbus unit (slave) id, often 1; some gateways use other ids per device |
| `device_profile` | register map, see below |
| `poll_interval_s` | seconds between polls (default 5) |
| `power_register` | which polled register is the power, in **W** (default `ac_power`). It becomes `current_power` in kW |
| `soc_register` | optional: a polled register holding SoC in **percent**; stored as the battery's SoC |
| `custom_registers` | extra registers, added to the profile's map (see the battery example below) |
| `control` | opts the device in to set points ([step 9](#9-turn-on-device-control-safely)) |

**Profiles shipped** (`src/vpp/protocols/modbus.py`):

| `device_profile` | Device | Notes |
|---|---|---|
| `solaredge_se` | SolarEdge SE inverters, SunSpec model 101/102/103 | values scaled by their `*_SF` registers |
| `fronius_symo` | Fronius Symo in its default "float" SunSpec mode (model 113) | |
| `sma_sunnyboy` | SMA Sunny Boy | SMA's own registers, not SunSpec |
| `generic_meter` | a generic power meter (default when `device_profile` is missing) | power is `power_total`: set `"power_register": "power_total"` |
| any other name, e.g. `custom` | empty map | only your `custom_registers` are polled |

The `solaredge_se` and `fronius_symo` maps assume the inverter model starts
at address 40069 (after a common model of length 65, as both vendors
document). They are checked point by point against the SunSpec model
definitions, but **not on physical hardware**. Compare a few polled values
with the inverter's own display before you trust them. Addresses in
`custom_registers` are the 0-based values sent on the wire; vendor tables
often list 1-based register numbers (40001 = address 40000).

A battery whose SoC is read from SunSpec model 124 (`ChaState`). The
`custom` profile polls only the two registers listed:

```bash
curl -s -X POST localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "name": "garage-battery",
  "resource_type": "battery",
  "rated_power": 5,
  "capacity_kwh": 10,
  "soc_min": 0.1,
  "soc_max": 0.95,
  "metadata": {"modbus": {
    "mode": "tcp", "host": "127.0.0.1", "port": 5020, "unit_id": 1,
    "device_profile": "custom", "poll_interval_s": 5,
    "soc_register": "cha_state",
    "custom_registers": {
      "cha_state": {"address": 40155, "data_type": "uint16", "unit": "%",
                    "scale_factor": "cha_state_sf",
                    "sunspec": {"model": 124, "base": 40147, "point": "ChaState"}},
      "cha_state_sf": {"address": 40169, "data_type": "int16",
                       "sunspec": {"model": 124, "base": 40147, "point": "ChaState_SF"}}
    }
  }}
}'
```

`base` is the address of the model's `ID` register on *your* device (40147
is where the simulator serves model 124). A register definition takes
`address`, `count` (registers, default 1), `register_type` (`holding`
default, `input`, `coil`, `discrete`), `data_type` (`uint16`, `int16`,
`uint32`, `int32`, `acc32`, `uint64`, `float32`), `scale` (multiplier),
`scale_factor` (name of a SunSpec `*_SF` register), `unit` and `sunspec`.

**Polling starts only at API startup.** Restart the API after you add or
change a `metadata.modbus` block. Each device then appears in
`GET /api/v1/protocols/` as `modbus:<resource id>`.

**Test against the simulator first.** `vpp simulate sunspec` serves a PV +
battery hybrid inverter (SunSpec models 1, 103, 123 and 124) over Modbus
TCP, with no hardware:

```bash
vpp simulate sunspec --port 5020 --speed 10
```

Point a resource at `"host": "127.0.0.1", "port": 5020` with
`"device_profile": "solaredge_se"` and, after restarting the API, its
`current_power` shows the simulator's 6.0 kW. The
[tutorial](getting-started/tutorial.md) does exactly that, and
[Testing without hardware](protocols.md#testing-without-hardware) lists
what the simulator serves. The simulator follows the specification; it
does not show how your vendor's firmware behaves.

### Meter readings and history

There is no file import (CSV or otherwise). Bring history in through the
API:

- **Interval meter data for a site** (import/export kWh per interval, used
  for bills): `POST /api/v1/sites/{id}/meter-readings`, up to 20,000
  readings per request. Re-sending an interval overwrites it.

    ```bash
    curl -s -X POST localhost:8000/api/v1/sites/<site id>/meter-readings \
      -H "X-API-Key: $VPP_KEY" -H 'Content-Type: application/json' -d '{
      "interval_minutes": 15,
      "readings": [
        {"timestamp": "2026-10-01T00:00:00-07:00", "import_kwh": 0.42, "export_kwh": 0.0},
        {"timestamp": "2026-10-01T00:15:00-07:00", "import_kwh": 0.38, "export_kwh": 0.0}
      ]
    }'
    # {"site_id": "...", "received": 2, "inserted": 2, "updated": 0}
    ```

    `interval_minutes` is 5, 15, 30 or 60, one value per site. Each
    timestamp is the **start** of its interval and must fall on an interval
    boundary in UTC.

- **Device history** (power and SoC): post timestamped samples to the
  [HTTP telemetry endpoint](#http-telemetry-endpoint), up to 5,000 per
  request. Samples older than 15 minutes are stored without changing the
  live value. For a CSV export, a few lines of Python do it:

    ```python
    import csv, json, os, urllib.request

    RESOURCE = "<resource id>"
    rows = [{"timestamp": r["timestamp"], "power_kw": float(r["power_kw"])}
            for r in csv.DictReader(open("history.csv"))]
    for i in range(0, len(rows), 5000):
        req = urllib.request.Request(
            f"http://localhost:8000/api/v1/resources/{RESOURCE}/telemetry",
            data=json.dumps({"source": "backfill", "samples": rows[i:i + 5000]}).encode(),
            headers={"X-API-Key": os.environ["VPP_KEY"], "Content-Type": "application/json"},
            method="POST")
        print(urllib.request.urlopen(req).read().decode())
    ```

## 5. Prices and tariffs

The platform has **no live price feed in the API**. Price feed adapters
(CAISO LMP, ComEd hourly) exist as library code in `vpp.tariffs.feeds`,
but nothing in the API calls them. Prices reach the optimizer in two ways:

- as a stored **tariff**, by id (`tariff_id`);
- as a list of prices you send with the request (`prices`, one per step,
  currency per kWh).

**Tariffs** are stored as OpenEI URDB (Utility Rate Database) JSON. Four
presets ship with the package: `pge_etouc`, `sce_toudprime` (real tariffs
captured at a stated date) and two labelled illustrative. Store one
(admin):

```bash
curl -s localhost:8000/api/v1/tariffs/presets/pge_etouc -H "Authorization: Bearer $TOKEN" \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); print(json.dumps({"name": p["name"], "utility": p["utility"] or "", "urdb_json": p["urdb_json"]}))' \
  > tariff.json
curl -s -X POST localhost:8000/api/v1/tariffs -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d @tariff.json
```

Check the rates against your utility before relying on a preset. To
import any URDB tariff by its label, set `OPENEI_API_KEY` (an OpenEI API
key) on the API server and call `POST /api/v1/tariffs/import-urdb`.
Exports are credited under a net energy metering (NEM) regime: `none`,
`nem2`, `nem3` or `net_billing`, taken from the tariff or the request
([details](tariffs.md#export-credit-nem)).

Where tariffs are used:

- **Battery schedules**: `POST /api/v1/optimization/schedule` with
  `{"resource_ids": ["<battery id>"], "tariff_id": "<tariff id>", "horizon_hours": 24}`
  plans charge and discharge over the horizon. It computes and records a
  plan; it does not write to devices.
- **Bills**: a customer account carries a `tariff_id`; its bill is computed
  from the site's meter readings (`GET /api/v1/customers/{id}/bill`).
  `POST /api/v1/tariffs/{id}/simulate` runs a bill from a CSV or
  synthetic load ([Tariffs and billing](tariffs.md)).

The single-interval dispatch in [step 8](#8-verify-before-you-dispatch)
does not use prices: it splits a power target you give it.

## 6. Grid signals and markets

| Integration | What is real | Default |
|---|---|---|
| OpenADR 2.0b VEN | talks to a real VTN over HTTPS when `VPP_OPENADR_VTN_URL` is set; simulated otherwise | off |
| IEEE 2030.5 client | talks to a real utility server with mutual TLS when `VPP_IEEE2030_5_SERVER_URL` is set; simulated otherwise | off |
| DR auto-response | turns those signals into fleet dispatch | **off**, even when the protocols are on |
| Trading | order book, risk checks, portfolio | **simulated venue only**: no order ever leaves the process |

**OpenADR 2.0b** (a VEN, virtual end node, polling the utility's VTN,
virtual top node). Your utility or DR program gives you the VTN URL, and
usually a client certificate:

```bash
export VPP_OPENADR_ENABLED=true
export VPP_OPENADR_VTN_URL=https://vtn.example.com/OpenADR2/Simple/2.0b
export VPP_OPENADR_VEN_NAME=my-vpp
export VPP_OPENADR_CERT_PATH=/etc/vpp/certs/ven.crt
export VPP_OPENADR_KEY_PATH=/etc/vpp/certs/ven.key
export VPP_OPENADR_CA_PATH=/etc/vpp/certs/vtn-ca.pem
```

`VPP_OPENADR_VEN_ID` is normally assigned by the VTN at registration; set
it only if the VTN tells you to.

**IEEE 2030.5** (Smart Energy Profile 2.0) needs the device certificate
issued for your installation:

```bash
export VPP_IEEE2030_5_ENABLED=true
export VPP_IEEE2030_5_SERVER_URL=https://utility.example.com:8443
export VPP_IEEE2030_5_CERT_PATH=/etc/vpp/certs/device.crt
export VPP_IEEE2030_5_KEY_PATH=/etc/vpp/certs/device.key
export VPP_IEEE2030_5_CA_PATH=/etc/vpp/certs/2030-5-ca.pem
```

**Watch first.** With auto-response off (`VPP_DR_AUTO_RESPONSE_ENABLED=false`,
the default) events and controls are received, recorded and shown, and the
VEN answers opt-in or opt-out per `VPP_OPENADR_AUTO_OPT_IN`, but nothing is
dispatched. Watch:

```bash
curl -s localhost:8000/api/v1/protocols/ -H "X-API-Key: $VPP_KEY"        # LIVE or SIMULATED per adapter
curl -s localhost:8000/api/v1/protocols/openadr/events -H "X-API-Key: $VPP_KEY"
curl -s localhost:8000/api/v1/protocols/ieee2030_5/controls -H "X-API-Key: $VPP_KEY"
curl -s localhost:8000/api/v1/dr/status -H "X-API-Key: $VPP_KEY"
curl -s localhost:8000/api/v1/dr/responses -H "X-API-Key: $VPP_KEY"
```

Only when the received events look right, consider
`VPP_DR_AUTO_RESPONSE_ENABLED=true`, together with caps
`VPP_DR_MAX_EXPORT_KW` and `VPP_DR_MAX_IMPORT_KW`. Even then, stationary
devices are written only under the rules of
[step 9](#9-turn-on-device-control-safely). How signals become targets is
in [Protocols: demand-response orchestrator](protocols.md#demand-response-orchestrator).

## 7. EV chargers (OCPP 1.6-J)

The API can act as an OCPP 1.6-J Central System: chargers connect to it
over a WebSocket. It is off by default, and needs a single API worker
(charger connections live in one process).

```bash
export VPP_OCPP_ENABLED=true
export VPP_OCPP_ALLOWED_CHARGE_POINTS='["CP-001"]'
export VPP_OCPP_BASIC_AUTH_PASSWORD='<long random password>'
```

In the charger's configuration (its app, web page or installer tool):

| Charger setting | Value |
|---|---|
| Central System / backend URL | `wss://<your host>/ocpp/CP-001` (the last segment is the charge point id) |
| Charge point id | `CP-001`, as in the allow-list |
| OCPP version / subprotocol | OCPP 1.6-J (`ocpp1.6`) |
| Authorization (Security Profile 1) | HTTP Basic: user = charge point id, password = `VPP_OCPP_BASIC_AUTH_PASSWORD` |

Terminate TLS at the reverse proxy and forward `/ocpp/` with WebSocket
upgrades ([nginx example](deployment.md#reverse-proxy-and-tls)). Basic
auth over plain `ws://` exposes the password. A connection with the wrong
id or password is refused (HTTP `403` on the upgrade).

Once connected the charger appears in
`GET /api/v1/protocols/ocpp/charge-points` (and on the console's
**Protocols** page) with its status and meter values.

To dispatch an EV, register the vehicle and bind it to the charger
connector, so the optimizer knows its battery and departure time:

```bash
curl -s -X POST localhost:8000/api/v1/v2g/vehicles -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "name": "family-ev", "capacity_kwh": 60, "current_soc": 0.5, "min_soc": 0.2,
  "target_soc": 0.8, "max_charge_kw": 11, "max_discharge_kw": 0,
  "v2g_capable": false, "id_tag": "RFID-1234"
}'
```

The vehicle is bound automatically when a transaction starts with its
`id_tag`, or by hand with `PUT /api/v1/v2g/vehicles/{ev_id}/binding`
(`{"charge_point_id": "CP-001", "connector_id": 1}`). Set
`v2g_capable: true` and `max_discharge_kw` only for a car and charger that
really support discharging: OCPP 1.6 has no standard discharge command,
and the platform sends discharge as negative charging-profile limits, a
vendor extension many chargers reject. Details in
[Protocols: OCPP](protocols.md#ocpp-16-j-central-system) and
[V2G fleet](protocols.md#v2g-fleet).

## 8. Verify before you dispatch

Check that every resource reports what the device really does.

**Live values.** One line per resource:

```bash
curl -s localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY" \
  | python3 -c 'import json,sys; [print(r["name"], r["online"], r["current_power"], r["state_of_charge"], r["state_of_charge_source"]) for r in json.load(sys.stdin)]'
# home-battery True 2.0 0.725 telemetry      (MQTT)
# garage-battery True 0.0 0.5 telemetry     (SoC from Modbus)
# farm-wind True 31.5 None None             (HTTP)
# rooftop-pv True 6.0 None None             (Modbus)
```

`state_of_charge_source` must say `telemetry` for each battery; `configured`
means no SoC has arrived yet and the optimizer is using the number you
typed in step 3.

**History.** `GET /api/v1/resources/{id}/metrics?window=24h` (windows
`1h`, `6h`, `24h`, `7d`, `30d`) returns time-bucketed power and SoC. The
console's **Assets** page shows the same as charts.

**Integrations.** `GET /api/v1/protocols/` lists each adapter (`mqtt`,
`modbus:<resource id>`, `ocpp`, ...) with `status`, `messages_received`
and `errors`.

**Live stream.** The WebSocket at `/api/v1/ws` pushes every update on the
`resource_updates` channel. Get a 60-second socket token with your key,
then connect with it:

```bash
curl -s -X POST localhost:8000/api/v1/ws/token -H "X-API-Key: $VPP_KEY"
# {"token": "eyJ...", ...}  then connect to
# ws://localhost:8000/api/v1/ws?token=<token>&channels=resource_updates
```

**Staleness.** Every resource response says how old its data is:

```bash
curl -s localhost:8000/api/v1/resources -H "X-API-Key: $VPP_KEY" \
  | python3 -c 'import json,sys; [print(r["name"], r["data_status"], r["telemetry_age_s"], r["telemetry_stale_after_s"]) for r in json.load(sys.stdin)]'
# home-battery fresh 4.2 300.0
# garage-battery fresh 3.1 50.0       (Modbus: 5 s poll x 10)
# farm-wind stale 912.7 300.0         (the HTTP sender stopped)
# rooftop-pv no_data None 50.0        (never reported)
```

| Field | Meaning |
|---|---|
| `telemetry_last_at` | time of the newest sample from any path (HTTP, MQTT, Modbus), UTC |
| `telemetry_age_s` | seconds since then |
| `telemetry_stale_after_s` | this resource's threshold (see below); `null` when the check is off |
| `telemetry_stale` | `true` when the data is older than the threshold |
| `data_status` | `fresh`, `stale`, `no_data` (never reported) or `unchecked` (check off) |
| `link_status` | for a Modbus-polled resource, its connection status (`connected`, `degraded`, `error`, ...) |

The threshold is, in this order: `metadata.telemetry_stale_after_s` on the
resource (`0` turns the check off for it); for a Modbus-polled resource,
its `poll_interval_s` × `VPP_TELEMETRY_STALE_MODBUS_POLL_MULTIPLIER`
(default 10); otherwise `VPP_TELEMETRY_STALE_AFTER_SECONDS` (default 300 s,
`0` turns the check off). Set a longer threshold on a device that reports
rarely, for example `"telemetry_stale_after_s": 1800` in the `metadata` of
a gateway posting every 15 minutes (a `PUT` replaces `metadata` as a
whole: send its other keys, such as `modbus`, too).

Stale data is **not** hidden or zeroed: `current_power` keeps its last
value and `online` changes only when you set it. (Dispatch separately stops
counting on a generator whose newest sample is older than
`VPP_GENERATION_TELEMETRY_MAX_AGE_S`: see
[generation availability](#generation-availability).) What changes is that
you are told:

- the built-in alert rule **Resource telemetry stale** fires once when a
  resource goes stale and resolves itself when data arrives again (console
  **Alerts**, `GET /api/v1/alerts`, and the alert webhook if configured).
  It is checked every `VPP_TELEMETRY_STALE_CHECK_INTERVAL_SECONDS` (30 s).
  Resources that never reported and resources set `"online": false` do not
  alert, and after an API restart each resource gets a full threshold to
  report again. To silence it, disable the rule (`PATCH /api/v1/alerts/rules/{id}`
  with `"enabled": false`) rather than deleting it: a deleted one is
  created again at startup.
- A Modbus device that stops answering is reported `degraded` in
  `GET /api/v1/protocols/` (`modbus:<resource id>`, with
  `consecutive_failed_polls`) and in the resource's `link_status` after
  `VPP_MODBUS_MAX_FAILED_POLLS` (3) failed polls in a row. Polling goes on,
  and the first successful poll reports it `connected` again.

The other built-in rules cover battery SoC below 10 %, temperature above
50 °C and state of health below 80 %. Prometheus users can still alert on
`time() - vpp_resource_last_update_timestamp_seconds`
([operations runbook](operations.md#what-to-monitor)).

**Dry-run dispatch.** Ask the optimizer to split a target across your
resources. Without `"apply": true` nothing is sent to any device:

```bash
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"target_power_kw": 4, "resource_ids": ["<rooftop-pv id>", "<garage-battery id>"]}'
```

`target_power_kw` is **export-positive**: positive asks the fleet to
deliver power, negative asks it to absorb (batteries charge). Leave out
`resource_ids` to use every online resource. The response has
`"applied": false`, a `run_id`, and one allocation per resource with the
inputs it used: `available_power_kw` and `availability_basis` (`telemetry`,
`no_data`, ... see [generation availability](#generation-availability)) for
generators, `state_of_charge` and `soc_source` for
batteries. A `soc_source` of `assumed` means the battery has no SoC at all
and the optimizer used 50 %.

Every run is stored. Read it back, with the full inputs and solution:

```bash
curl -s localhost:8000/api/v1/optimization/runs/<run_id> -H "X-API-Key: $VPP_KEY"
curl -s "localhost:8000/api/v1/optimization/runs?limit=5" -H "X-API-Key: $VPP_KEY"
```

The console shows them under **Trading → Dispatches**.

## 9. Turn on device control safely

Writing set points to a device takes **two switches**, both off by
default:

1. `VPP_CONTROL_ENABLED=true` on the API: the global kill switch. While it
   is false nothing is ever written; deliveries report `disabled`.
2. A `control` block with `"enabled": true` in the resource's
   `metadata.modbus`: the per-device opt-in.

Only Modbus devices can be controlled (EVs are driven through OCPP). The
profiles:

| `control.profile` | Writes | For |
|---|---|---|
| `sunspec_123` | SunSpec model 123 `WMaxLimPct` (output limit, % of `reference_kw`) | PV inverters: curtailment |
| `sunspec_124` | SunSpec model 124 `InWRte` / `OutWRte` / `StorCtl_Mod` | batteries: charge and discharge. **How vendors interpret forced charge/discharge differs; validate on your device** |
| `register` | one setpoint register you describe (`register` or `address`, `data_type`, `unit`, `scale`, `sign`...) | devices without SunSpec controls |

The full list of keys and rules is in
[Protocols: device control](protocols.md#device-control-setpoint-actuator).

**Start with one device and simulate it.** `"simulate": true` runs the
whole safety pipeline and records what would be written, with no device
I/O:

```bash
curl -s -X PUT localhost:8000/api/v1/resources/<rooftop-pv id> -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' -d '{
  "metadata": {"modbus": {
    "mode": "tcp", "host": "192.168.1.50", "port": 502, "unit_id": 1,
    "device_profile": "solaredge_se", "poll_interval_s": 5,
    "control": {"enabled": true, "simulate": true, "profile": "sunspec_123",
                "model_base": "auto", "reference_kw": 10, "revert_timeout_s": 300,
                "max_kw": 10, "deadband_kw": 0.2, "min_interval_s": 10}
  }}
}'
```

- `model_base: "auto"` finds model 123 by SunSpec discovery. The model
  header is read back before the first write, and a mismatch is refused, so
  a wrong address is never written.
- `reference_kw` is what 100 % means: set it to the inverter's `WMax`
  (maximum output, in kW).
- `revert_timeout_s` programs the device's own revert timer: if the VPP
  stops refreshing the limit (crash, network loss), the device returns to
  normal operation by itself after this many seconds. **Always set it**
  where the profile supports it (`sunspec_123`, `sunspec_124`).
- `max_kw` / `min_kw` clamp every set point; `deadband_kw` skips rewrites
  closer than this to the active one; `min_interval_s` limits how often the
  device is written; `safe_setpoint_kw` is written when a set point expires
  (default: control is released instead).

Changes to the `control` block take effect at the next dispatch; no restart
is needed. The kill switch applies to simulated blocks too, so turn it on
now. That needs a restart:

```bash
export VPP_CONTROL_ENABLED=true
vpp serve
```

**Apply one dispatch** to the one device:

```bash
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "X-API-Key: $VPP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"target_power_kw": 4, "resource_ids": ["<rooftop-pv id>"], "apply": true}'
```

Look at `device_deliveries` in the response. With `simulate` it says
`"status": "simulated"`. Remove `"simulate": true` (another `PUT`) and
apply again; a real write reads:

```text
"status": "accepted", "setpoint_kw": 4.0, "verified": true,
"reason": "written and verified",
"writes": [{"register": "WMaxLimPct_RvrtTms", "value": 300, ...},
           {"register": "WMaxLimPct", "value": 40.0, ...},
           {"register": "WMaxLim_Ena", "value": 1, ...}]
```

`verified: true` means every register read back as written. Then confirm
on the device itself (its display, app or the next poll): the PV output
should drop to 4 kW. Applying requires the `operator` or `admin` role.

Each set point lives for the dispatch interval (`interval_minutes`, default
15) plus `VPP_CONTROL_EXPIRY_GRACE_S` (30 s). A watchdog then writes
`safe_setpoint_kw` or releases control. While a set point is active, the
watchdog refreshes it before the device's revert timer runs out. **To keep
control, dispatch again before the interval ends**; nothing re-dispatches
by itself unless DR auto-response is on.

**What was written, and by whom:**

```bash
curl -s localhost:8000/api/v1/optimization/setpoints -H "X-API-Key: $VPP_KEY"
curl -s "localhost:8000/api/v1/audit?action=control." -H "Authorization: Bearer $TOKEN"   # admin
```

`setpoints` shows the kill switch, the active set points and the latest
device commands (each one is also stored in the event log with its
`run_id`). The audit log records who applied each dispatch
(`control.dispatch`), including refused attempts, and who set a resource
offline (`control.resource_offline`) or back online.

**Stopping and rolling back**, fastest first:

| To stop... | Do | Effect |
|---|---|---|
| one device, now | `PUT /api/v1/resources/{id}` with `{"online": false}` | its set point is released at once (or `safe_setpoint_kw` written), the same fallback as an expired dispatch; later dispatches skip it until it is back online |
| one device | `PUT` its `metadata.modbus.control` with `"enabled": false` | the watchdog releases its set point within `VPP_CONTROL_WATCHDOG_INTERVAL_S` (5 s); later dispatches report `disabled` |
| every device | set `VPP_CONTROL_ENABLED=false` and restart the API | on shutdown every active set point is released; after restart nothing is written |
| if the API is down or unreachable | nothing | each device returns to normal on its own after `revert_timeout_s` |

Setting a resource `"online": false` releases what the platform holds on
it, including an output cap left by a previous run of the API; if the
device's revert timer has already lifted that cap, nothing is written. The
release is in the device command log and the change in the audit log
(`control.resource_offline`). If the write fails (device unreachable), the
device keeps its limit until its own revert timer fires. To roll back
completely, delete the `control` blocks; polling and dry-run dispatch keep
working.

Add devices one at a time, and watch each through at least one full
dispatch, expiry and release before adding the next.

## 10. Operate

- **Retention.** Telemetry is kept forever by default. Set
  `VPP_TELEMETRY_RETENTION_DAYS` (30–90 is typical) before the disk fills;
  `vpp prune --dry-run` shows what would be deleted
  ([operations: data retention](operations.md#data-retention)).
- **Backups.** Back up the database (SQLite online backup or `pg_dump`) and
  test a restore ([operations runbook](operations.md#what-to-back-up)).
- **Monitoring.** Scrape `/metrics` with Prometheus; protect it with
  `VPP_METRICS_BEARER_TOKEN` or keep it internal. Stale telemetry raises
  the built-in "Resource telemetry stale" alert; alert on protocol errors
  and failed optimization runs too
  ([observability](observability.md), [what to monitor](operations.md#what-to-monitor)).
- **Upgrades.** Back up, read the release's upgrade notes in the
  [changelog](changelog.md), install the new version, run `vpp migrate`,
  restart ([upgrading](operations.md#upgrading)). A restart releases
  active set points; dispatch again afterwards.
- **Credentials.** Rotate API keys and the secret key as described in
  [operations: credentials](operations.md#credentials).

## 11. Troubleshooting

Logs go to standard output (JSON lines with `VPP_LOG_JSON=true`, the
default in production). Set `VPP_LOG_LEVEL=DEBUG` for more detail.

| Symptom | Likely cause | How to check and fix |
|---|---|---|
| `401` `Invalid API key` | wrong or revoked key, or wrong header | the header is `X-API-Key` (or `VPP_API_KEY_HEADER`); list keys with `GET /api/v1/auth/api-keys` |
| `401` on a token that worked earlier | token expired (60 min), password changed, `VPP_SECRET_KEY` changed | log in again; use an API key for integrations |
| `403` `Role 'viewer' is not authorised for this action` | the key's role is too low | writes need `operator`; a key never exceeds its creator's current role |
| `429` | rate limit per client address | batch requests; behind a proxy set `VPP_TRUSTED_PROXIES`; raise `VPP_RATE_LIMIT_REQUESTS_PER_MINUTE` |
| `422` on resource create | misspelled or wrong-type field | the response names the field; fields of another resource type are rejected too |
| `422` `sample timestamp ... is in the future` | device clock ahead, or local time sent without an offset | send UTC (`...Z`) or include the offset; keep device clocks on NTP |
| telemetry posted but `current_power` unchanged | samples older than 15 minutes (stored as history only), or an MQTT payload without `power` | check timestamps and time zones; read `GET /api/v1/resources/{id}/metrics?window=1h` |
| MQTT: nothing stored | topic not `vpp/<site>/battery/<resource id>/<metric>`, resource is not a `battery`, no `soc` in the payload, payload not JSON, `timestamp` without a time zone or in the future | `GET /api/v1/protocols/` shows `messages_received` for `mqtt`; the log warns about unknown resources; check `VPP_MQTT_TOPIC_PREFIX` |
| MQTT adapter missing from `/api/v1/protocols/` | `VPP_MQTT_INGESTION_ENABLED` not set | set it and restart |
| SoC 100× too small or too large | percent sent where a fraction is expected or the reverse | HTTP `state_of_charge` is 0–1; MQTT `soc` and Modbus `soc_register` are percent |
| power 1000× off | wrong unit | HTTP and MQTT take kW; Modbus `power_register` must be in W |
| Modbus resource never updates | polling enabled after the resource was configured, wrong host/port/unit id, firewall | restart the API (discovery runs at startup); look for `Modbus connect failed` and `Error reading register` in the log and `errors` on `modbus:<id>` in `/api/v1/protocols/` |
| `modbus:<id>` shows `degraded` | the connection is up but the device answers no reads (powered off, wrong unit id, gateway lost it) | check the device and `unit_id`; it turns `connected` by itself on the next good poll. `consecutive_failed_polls` counts the misses |
| Modbus values wrong by a power of ten, or `0` | wrong profile or base address, scale-factor register missing | compare with the device display; for SunSpec devices use `model_base: "auto"` for control and check the base address of `custom_registers` |
| generator planned at 0 kW, `availability_basis: "no_data"` | no telemetry sample in the last `VPP_GENERATION_TELEMETRY_MAX_AGE_S` (300 s) | send telemetry, or choose a fallback ([generation availability](#generation-availability)) |
| device unplugged but the resource still shows power | the last value is kept | `data_status` is `stale` and the "Resource telemetry stale" alert fires after the threshold ([Staleness](#8-verify-before-you-dispatch)); set `"online": false` while it is out |
| "Resource telemetry stale" for a device that reports rarely | its interval is longer than the threshold | set `metadata.telemetry_stale_after_s` on the resource (or `0` to turn the check off for it) |
| dispatch delivery `disabled` | `VPP_CONTROL_ENABLED` is false, or `control.enabled` is false | the `reason` field says which |
| dispatch delivery `not_configured` | no `metadata.modbus.control` block | add one ([step 9](#9-turn-on-device-control-safely)) |
| dispatch delivery `failed` | device refused the write, read-back mismatch, model header mismatch | the `reason` field and the API log name the register; test against `vpp simulate sunspec` |
| `vpp serve` prints `Refusing to start` | several workers with OCPP or device control | use one worker (`VPP_API_WORKERS=1`, the default) |
| `503 leader_unavailable` | several workers and the leader of a background task is restarting | retry after `Retry-After`; check the `lease` lines in the log ([process model](architecture.md#process-model)) |
| OCPP charger cannot connect | `VPP_OCPP_ENABLED` off, id not in the allow-list, wrong password, proxy not forwarding WebSocket upgrades | the upgrade answers `403` for auth failures; test `wss://` through the proxy |
