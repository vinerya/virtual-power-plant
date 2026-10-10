# Tutorial: dispatch a simulated inverter

In about ten minutes, with no hardware: run the bundled SunSpec simulator
(a PV + battery hybrid inverter speaking Modbus TCP), register it as a
resource, let the API poll it, then dispatch a 4 kW export target and watch
the inverter curtail to it.

You need the package with the `api`, `db`, `cli` and `protocols` extras
([Installation](installation.md)), `curl`, and two terminals. Ports 8000
(API) and 5020 (simulator) must be free.

!!! note "Simulated, not validated"

    The simulator is written from the SunSpec specification. It shows that
    the VPP's register maps, polling and control loop are consistent; it
    says nothing about how a particular vendor's firmware behaves. See
    [Testing without hardware](../protocols.md#testing-without-hardware).

## 1. Start the simulated inverter

Terminal 1:

```bash
vpp simulate sunspec --port 5020 --speed 10
```

It serves SunSpec models 1, 103, 123 and 124 on `127.0.0.1:5020` and
prints a status line every few seconds:

```text
INFO t=90s ac=6000W pv=6000/6000W battery=+0W soc=50.0% limit=off stor_ctl=0
```

6 kW of PV, no export limit. `--speed 10` runs ten simulated seconds per
real second. `vpp simulate sunspec --help` lists the other options
(rating, PV power, battery size, initial state of charge).

## 2. Start the API with Modbus polling and device control

Terminal 2, in an empty directory (the SQLite database is created there):

```bash
export VPP_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
export VPP_MODBUS_INGESTION_ENABLED=true VPP_CONTROL_ENABLED=true
export VPP_ADMIN_PASSWORD='pick-a-long-passphrase'
vpp migrate
vpp users create-admin admin
vpp serve
```

Both Modbus polling and device control are off by default; device control
needs a single API worker, which is the default.

## 3. Register the inverter as a resource

Terminal 3:

```bash
export VPP_ADMIN_PASSWORD='pick-a-long-passphrase'   # as in terminal 2
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -d username=admin -d password="$VPP_ADMIN_PASSWORD" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
AUTH="Authorization: Bearer $TOKEN"

PV_ID=$(curl -s -X POST localhost:8000/api/v1/resources -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{
  "name": "sim-pv", "resource_type": "solar", "rated_power": 10,
  "metadata": {"modbus": {"host": "127.0.0.1", "port": 5020, "unit_id": 1,
    "device_profile": "solaredge_se", "poll_interval_s": 2,
    "control": {"enabled": true, "profile": "sunspec_123", "model_base": "auto",
                "reference_kw": 10, "revert_timeout_s": 300}}}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "$PV_ID"
```

`metadata.modbus` tells the ingestion loop where to poll (the
`solaredge_se` register map matches the simulator's model 1 + 103 layout)
and `control` opts this one device into set points through SunSpec model
123 (`WMaxLimPct`), with a 300 s revert timer on the device as a safety
net.

Modbus resources are discovered when the API starts, so **restart
`vpp serve`** in terminal 2 (Ctrl-C, then `vpp serve` again). After a poll
or two the resource reports the simulator's output:

```bash
curl -s localhost:8000/api/v1/resources/$PV_ID -H "$AUTH" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["online"], r["current_power"])'
# True 6.0
```

## 4. Dispatch

First a dry run: the optimizer (Pyomo + HiGHS) computes and records an
allocation but sends nothing to the device:

```bash
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d "{\"target_power_kw\": 4, \"resource_ids\": [\"$PV_ID\"]}"
```

The response has `"applied": false`, the allocation (`4.0` kW of the
`6.0` kW available from telemetry), the solver used and a `run_id`.

Now apply it:

```bash
curl -s -X POST localhost:8000/api/v1/optimization/dispatch -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d "{\"target_power_kw\": 4, \"resource_ids\": [\"$PV_ID\"], \"apply\": true}"
```

This time `"applied": true` and `device_deliveries` shows what was written
and read back:

```text
"status": "accepted", "setpoint_kw": 4.0, "verified": true,
"reason": "written and verified",
"writes": [{"register": "WMaxLimPct_RvrtTms", "value": 300, ...},
           {"register": "WMaxLimPct", "value": 40.0, "raw": [400]},
           {"register": "WMaxLim_Ena", "value": 1, ...}]
```

In terminal 1 the simulator logs the writes, and its next status line
shows the PV curtailed to 40 % of its 10 kW rating:

```text
INFO write 40126..40126: 123.WMaxLimPct=40.0
INFO t=190s ac=4000W pv=4000/6000W battery=+0W soc=50.0% limit=40.0% stor_ctl=0
```

## 5. Look at what happened

```bash
# Active set points, keep-alive watchdog and recent device commands
curl -s localhost:8000/api/v1/optimization/setpoints -H "$AUTH"

# Every optimization run with its inputs and result
curl -s "localhost:8000/api/v1/optimization/runs?limit=5" -H "$AUTH"
```

`GET /api/v1/optimization/runs/{run_id}` returns one run with its stored
inputs and result (`/explain` on the same path compares a *schedule* run
with idling and the rule-based dispatcher; for a dispatch run it answers
`204 No Content`). In the web console ([First run](first-run.md#start-the-web-console)) the same
history is under **Trading → Dispatches**.

## Where to go next

- Connect your own devices and data:
  [Connecting real data and devices](../connecting-real-data.md).
- Add the battery side of the simulator (model 124 storage control) and
  dispatch charge and discharge: the full example is in
  [Protocols: by hand, against a running API](../protocols.md#by-hand-against-a-running-api).
- Run the same thing in Docker: `docker compose --profile sim up -d`
  starts the simulator as `sunspec-sim:5020`.
- Device control safety (clamping, deadband, rate limits, read-back,
  fallback watchdog): [Device control](../protocols.md#device-control-setpoint-actuator).
- Plan a horizon schedule or backtest against a tariff:
  `POST /api/v1/optimization/schedule` and `/backtest` in the
  [REST API reference](../reference/rest-api.md).
