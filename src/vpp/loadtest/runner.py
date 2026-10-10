"""Capacity / soak harness: drive a real ``vpp serve`` process and measure it.

The harness

1. migrates a database (a fresh SQLite file by default, or any
   ``--database-url``, e.g. a scratch PostgreSQL database),
2. starts ``--modbus-devices`` SunSpec simulators (``vpp simulate sunspec``
   subprocesses),
3. boots the API once to bootstrap an admin and register the fleet through
   the HTTP API (``N`` batteries + one solar resource per simulator), then
   restarts it with MQTT / Modbus ingestion enabled (Modbus devices are
   discovered at startup),
4. runs the scenarios concurrently for ``--duration`` seconds:

   (a) telemetry ingestion -- ``N x M`` points/s over MQTT (published to a
       broker the server subscribes to; latency is publish -> WebSocket
       broadcast of the persisted row) or the HTTP telemetry endpoint,
   (b) authenticated API reads of the dashboard endpoints (JWT and API key),
   (c) ``S`` WebSocket subscribers receiving every broadcast,
   (d) a dispatch optimisation over ``K`` resources every ``D`` seconds,
   (e) Modbus polling of the simulators by the server,

5. samples, every ``--sample-interval`` seconds, the server's process-tree
   RSS and CPU, the database size and table row counts, and per-scenario
   throughput / p50 / p95 / p99 / errors, and
6. writes ``report.json`` + ``report.md`` (and the server log) to
   ``--output-dir`` and prints the summary.

Load is open-loop (a fixed request rate, independent of response time) with
a cap on in-flight requests per scenario; when the cap is hit the tick is
counted as ``skipped`` -- the server is not keeping up.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vpp.loadtest import procinfo
from vpp.loadtest.stats import Recorder, linear_slope, thirds_ratio

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx

ADMIN_USERNAME = "loadtest-admin"
_CLI = "from vpp.cli.main import cli; cli(prog_name='vpp')"

#: Dashboard reads issued by scenario (b), round-robin. ``{rid}`` is a random
#: battery that receives telemetry.
READ_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("resources", "/api/v1/resources?limit=50"),
    ("resource", "/api/v1/resources/{rid}"),
    ("metrics_1h", "/api/v1/resources/{rid}/metrics?window=1h"),
    ("metrics_24h", "/api/v1/resources/{rid}/metrics?window=24h"),
    ("sites", "/api/v1/sites"),
    ("alerts", "/api/v1/alerts?limit=50"),
    ("dispatches", "/api/v1/dispatches?limit=20"),
    ("auth_me", "/api/v1/auth/me"),
    ("portfolio", "/api/v1/trading/portfolio"),
    ("protocols", "/api/v1/protocols"),
)

#: Tables whose row counts are sampled.
COUNTED_TABLES = (
    "battery_states",
    "resource_telemetry",
    "optimization_runs",
    "audit_log",
    "alerts",
    "cluster_events",
)


@dataclass
class LoadConfig:
    output_dir: Path
    database_url: str | None = None  # None: a fresh SQLite file in output_dir
    duration_s: float = 60.0
    sample_interval_s: float = 10.0
    resources: int = 20  # N batteries receiving telemetry
    points_per_s: float = 1.0  # M points/s per battery
    ingest: str = "mqtt"  # mqtt | http | none
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_qos: int = 1
    api_rps: float = 20.0
    ws_clients: int = 10
    dispatch_interval_s: float = 15.0
    dispatch_resources: int = 20  # K
    modbus_devices: int = 2
    modbus_poll_s: float = 1.0
    workers: int = 1
    max_inflight: int = 64
    drain_s: float = 10.0
    seed: int = 1
    extra_env: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.ingest not in ("mqtt", "http", "none"):
            raise ValueError("ingest must be mqtt, http or none")
        if self.duration_s <= 0 or self.sample_interval_s <= 0:
            raise ValueError("duration and sample interval must be positive")
        if self.resources < 1:
            raise ValueError("at least one resource is needed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def sqlite_path(database_url: str) -> Path | None:
    m = re.match(r"^sqlite(?:\+\w+)?:///(.+)$", database_url)
    return Path(m.group(1)) if m else None


def db_file_bytes(database_url: str) -> int | None:
    path = sqlite_path(database_url)
    if path is None:
        return None
    return sum(
        p.stat().st_size
        for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal"))
        if p.exists()
    )


async def wait_tcp(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _r, w = await asyncio.open_connection(host, port)
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return True
        except OSError:
            await asyncio.sleep(0.2)
    return False


class ManagedProcess:
    """A subprocess whose output goes to a log file; stopped with SIGINT, then SIGKILL."""

    def __init__(self, name: str, argv: list[str], env: dict[str, str], log_path: Path) -> None:
        self.name = name
        self.argv = argv
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        log = open(self.log_path, "ab")  # noqa: SIM115 - closed with the process
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        log.close()

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tail(self, lines: int = 30) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    async def stop(self, timeout_s: float = 30.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        pids = procinfo.process_tree(self.proc.pid)
        with contextlib.suppress(ProcessLookupError):
            self.proc.send_signal(signal.SIGINT)
        deadline = time.monotonic() + timeout_s
        while self.proc.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if self.proc.poll() is None:
            for pid in pids:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
            await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Database probe (size + row counts)
# ---------------------------------------------------------------------------


class DbProbe:
    def __init__(self, database_url: str) -> None:
        from vpp.db.engine import create_engine_from_settings

        self.url = database_url
        self.engine = create_engine_from_settings(database_url)

    async def sample(self) -> dict[str, Any]:
        from sqlalchemy import func, select, table, text

        out: dict[str, Any] = {"rows": {}}
        try:
            async with self.engine.connect() as conn:
                if self.engine.dialect.name == "postgresql":
                    size = (
                        await conn.execute(text("SELECT pg_database_size(current_database())"))
                    ).scalar()
                    out["db_bytes"] = int(size or 0)
                for name in COUNTED_TABLES:
                    try:
                        query = select(func.count()).select_from(table(name))
                        n = (await conn.execute(query)).scalar()
                        out["rows"][name] = int(n or 0)
                    except Exception:
                        out["rows"][name] = None
                        await conn.rollback()
        except Exception as exc:  # the sample is best effort
            out["error"] = f"{type(exc).__name__}: {exc}"
        if "db_bytes" not in out:
            out["db_bytes"] = db_file_bytes(self.url)
        return out

    async def close(self) -> None:
        await self.engine.dispose()


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


class LoadRun:
    def __init__(self, cfg: LoadConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.out = Path(cfg.output_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.database_url = cfg.database_url or (f"sqlite+aiosqlite:///{self.out / 'loadtest.db'}")
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.password = secrets.token_urlsafe(24)
        self.secret_key = secrets.token_urlsafe(48)
        self.rng = random.Random(cfg.seed)
        self.stop_event = asyncio.Event()  # everything (after the drain)
        self.stop_load = asyncio.Event()  # load generators only
        self.rec: dict[str, Recorder] = {}
        self.api_rec: dict[str, Recorder] = {}
        self.battery_ids: list[str] = []
        self.modbus_ids: list[str] = []
        self.api_key = ""
        self._token = ""
        self._token_at = 0.0
        self.sims: list[ManagedProcess] = []
        self.server: ManagedProcess | None = None
        self.timeline: list[dict[str, Any]] = []
        # MQTT end-to-end matching: (resource_id, soc) -> publish time.
        self.pending: dict[tuple[str, float], float] = {}
        self.published = 0
        self.mqtt_lost = 0
        self.mqtt_unobserved = 0
        self.modbus_last_seen: dict[str, float] = {}
        self.ws_messages = 0
        self.ws_connected = 0
        self.ws_reconnects = 0
        self.dispatch_methods: Counter[str] = Counter()
        self.dispatch_solve_ms: list[float] = []
        self.t0 = 0.0
        self.phase = "load"  # "drain" once the generators have stopped
        self._harness_cpu_last: float | None = None

    def recorder(self, name: str) -> Recorder:
        if name not in self.rec:
            self.rec[name] = Recorder(name)
        return self.rec[name]

    # -- environment --------------------------------------------------------

    def server_env(self, *, ingestion: bool) -> dict[str, str]:
        env = dict(os.environ)
        pw_file = self.out / "admin_password"
        pw_file.write_text(self.password)
        pw_file.chmod(0o600)
        env.update(
            {
                "VPP_ENV": "production",
                "VPP_DATABASE_URL": self.database_url,
                "VPP_SECRET_KEY": self.secret_key,
                "VPP_USE_ALEMBIC": "true",
                "VPP_LOG_LEVEL": "INFO",
                # All load comes from one address; the limiter would reject it.
                "VPP_RATE_LIMIT_ENABLED": "false",
                "VPP_BOOTSTRAP_ADMIN_USERNAME": ADMIN_USERNAME,
                "VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE": str(pw_file),
                "VPP_API_WORKERS": str(self.cfg.workers),
                "VPP_MQTT_INGESTION_ENABLED": str(ingestion and self.cfg.ingest == "mqtt"),
                "VPP_MQTT_BROKER_HOST": self.cfg.mqtt_host,
                "VPP_MQTT_BROKER_PORT": str(self.cfg.mqtt_port),
                "VPP_MQTT_TOPIC_PREFIX": "vpp/#",
                "VPP_MODBUS_INGESTION_ENABLED": str(ingestion and self.cfg.modbus_devices > 0),
            }
        )
        env.update(self.cfg.extra_env)
        return env

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # -- setup --------------------------------------------------------------

    async def migrate(self) -> None:
        from vpp.db.engine import run_alembic_upgrade

        await asyncio.to_thread(run_alembic_upgrade, self.database_url)

    async def start_simulators(self) -> list[int]:
        ports = []
        for i in range(self.cfg.modbus_devices):
            port = free_port()
            sim = ManagedProcess(
                f"sunspec-{i}",
                [
                    sys.executable,
                    "-c",
                    _CLI,
                    "simulate",
                    "sunspec",
                    "--port",
                    str(port),
                    "--serial",
                    f"LT{i:06d}",
                    "--pv",
                    str(4000 + 250 * (i % 8)),
                    "--status-interval",
                    "3600",
                    "--log-level",
                    "WARNING",
                ],
                dict(os.environ),
                self.out / f"sunspec-{i}.log",
            )
            sim.start()
            self.sims.append(sim)
            ports.append(port)
        for sim, port in zip(self.sims, ports, strict=True):
            if not await wait_tcp("127.0.0.1", port, 30):
                raise RuntimeError(f"{sim.name} did not listen on {port}:\n{sim.tail()}")
        return ports

    async def boot(self, *, ingestion: bool, label: str) -> ManagedProcess:
        import httpx

        server = ManagedProcess(
            f"api-{label}",
            [
                sys.executable,
                "-c",
                _CLI,
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                str(self.cfg.workers),
            ],
            self.server_env(ingestion=ingestion),
            self.out / f"server-{label}.log",
        )
        server.start()
        deadline = time.monotonic() + 120
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5) as client:
            while time.monotonic() < deadline:
                if not server.alive():
                    raise RuntimeError(f"API server exited during startup:\n{server.tail()}")
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get("/ready")).status_code == 200:
                        return server
                await asyncio.sleep(0.25)
        await server.stop()
        raise RuntimeError(f"API server not ready after 120 s:\n{server.tail()}")

    async def login(self, client: httpx.AsyncClient) -> str:
        resp = await client.post(
            "/api/v1/auth/token", json={"username": ADMIN_USERNAME, "password": self.password}
        )
        resp.raise_for_status()
        self._token = str(resp.json()["access_token"])
        self._token_at = time.monotonic()
        return self._token

    async def bearer(self, client: httpx.AsyncClient) -> dict[str, str]:
        # Sessions last VPP_JWT_EXPIRE_MINUTES (60 by default): re-login well before.
        if not self._token or time.monotonic() - self._token_at > 20 * 60:
            await self.login(client)
        return {"Authorization": f"Bearer {self._token}"}

    async def register_fleet(self, sim_ports: list[int]) -> None:
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=30) as client:
            headers = await self.bearer(client)
            resp = await client.post(
                "/api/v1/auth/api-key",
                json={"name": "loadtest", "role": "operator"},
                headers=headers,
            )
            resp.raise_for_status()
            self.api_key = str(resp.json()["key"])
            for i in range(self.cfg.resources):
                resp = await client.post(
                    "/api/v1/resources",
                    json={
                        "name": f"lt-battery-{i:05d}",
                        "resource_type": "battery",
                        "rated_power": 5.0,
                        "capacity_kwh": 13.5,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                self.battery_ids.append(str(resp.json()["id"]))
            for i, port in enumerate(sim_ports):
                resp = await client.post(
                    "/api/v1/resources",
                    json={
                        "name": f"lt-pv-modbus-{i:04d}",
                        "resource_type": "solar",
                        "rated_power": 10.0,
                        "metadata": {
                            "modbus": {
                                "mode": "tcp",
                                "host": "127.0.0.1",
                                "port": port,
                                "unit_id": 1,
                                "device_profile": "solaredge_se",
                                "poll_interval_s": self.cfg.modbus_poll_s,
                                "power_register": "ac_power",
                            }
                        },
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                self.modbus_ids.append(str(resp.json()["id"]))

    # -- pacing -------------------------------------------------------------

    async def paced(
        self, rate_hz: float, fn: Callable[[], Awaitable[None]], rec: Recorder
    ) -> None:
        """Call ``fn`` ``rate_hz`` times per second (open loop) until stopped."""
        if rate_hz <= 0:
            return
        loop = asyncio.get_running_loop()
        interval = 1.0 / rate_hz
        next_t = loop.time()
        inflight: set[asyncio.Task[None]] = set()
        while not self.stop_load.is_set():
            now = loop.time()
            if next_t > now:
                await asyncio.sleep(next_t - now)
                continue
            next_t += interval
            if now - next_t > 1.0:  # far behind (harness starved): do not burst
                rec.skip()
                next_t = now
            if len(inflight) >= self.cfg.max_inflight:
                rec.skip()
                continue
            task = asyncio.ensure_future(fn())
            inflight.add(task)
            task.add_done_callback(inflight.discard)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

    async def timed_request(
        self,
        client: httpx.AsyncClient,
        rec: Recorder,
        method: str,
        url: str,
        *,
        also: Recorder | None = None,
        **kwargs: Any,
    ) -> httpx.Response | None:
        t = time.perf_counter()
        try:
            resp = await client.request(method, url, **kwargs)
        except Exception as exc:
            rec.error(type(exc).__name__)
            if also is not None:
                also.error(type(exc).__name__)
            return None
        dt = time.perf_counter() - t
        if resp.status_code >= 400:
            rec.error(f"HTTP {resp.status_code}")
            if also is not None:
                also.error(f"HTTP {resp.status_code}")
            return resp
        rec.ok(dt)
        if also is not None:
            also.ok(dt)
        return resp

    # -- scenario (a): telemetry ingestion -----------------------------------

    def _next_soc(self, seq: int) -> float:
        # Unique per (resource, message) within a run, so the WebSocket
        # broadcast of a persisted row can be matched to its publish time.
        return round(20.0 + (seq % 600_000) * 0.0001, 4)

    async def ingest_mqtt(self) -> None:
        import paho.mqtt.client as mqtt

        rec = self.recorder("ingest_mqtt_publish")
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="vpp-loadtest")
        client.max_queued_messages_set(0)
        client.max_inflight_messages_set(1000)
        client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, 60)
        client.loop_start()
        seq = Counter[str]()
        rr = 0

        async def publish() -> None:
            nonlocal rr
            rid = self.battery_ids[rr % len(self.battery_ids)]
            rr += 1
            seq[rid] += 1
            soc = self._next_soc(seq[rid])
            payload = json.dumps({"soc": soc, "power": round(self.rng.uniform(-5, 5), 3)})
            self.pending[(rid, soc)] = time.time()
            info = client.publish(
                f"vpp/site-lt/battery/{rid}/telemetry", payload, self.cfg.mqtt_qos
            )
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self.pending.pop((rid, soc), None)
                rec.error(f"publish rc={info.rc}")
                return
            self.published += 1
            rec.ok()

        try:
            await self.paced(self.cfg.resources * self.cfg.points_per_s, publish, rec)
        finally:
            await asyncio.sleep(0.5)
            client.loop_stop()
            client.disconnect()

    async def ingest_http(self, client: httpx.AsyncClient) -> None:
        rec = self.recorder("ingest_http")
        rr = 0

        async def post() -> None:
            nonlocal rr
            rid = self.battery_ids[rr % len(self.battery_ids)]
            rr += 1
            body = {
                "samples": [
                    {
                        "power_kw": round(self.rng.uniform(-5, 5), 3),
                        "state_of_charge": round(self.rng.uniform(0.2, 0.8), 4),
                    }
                ],
                "source": "loadtest",
            }
            headers = {self._api_key_header: self.api_key}
            await self.timed_request(
                client,
                rec,
                "POST",
                f"/api/v1/resources/{rid}/telemetry",
                json=body,
                headers=headers,
            )

        await self.paced(self.cfg.resources * self.cfg.points_per_s, post, rec)

    _api_key_header = "X-API-Key"

    # -- scenario (b): API reads ---------------------------------------------

    async def api_reads(self, client: httpx.AsyncClient) -> None:
        agg = self.recorder("api_read")
        i = 0

        async def read() -> None:
            nonlocal i
            name, path = READ_ENDPOINTS[i % len(READ_ENDPOINTS)]
            use_key = (i // len(READ_ENDPOINTS)) % 2 == 1
            i += 1
            rid = self.rng.choice(self.battery_ids)
            if use_key:
                headers = {self._api_key_header: self.api_key}
            else:
                headers = await self.bearer(client)
            rec = self.api_rec.setdefault(name, Recorder(name))
            await self.timed_request(
                client, rec, "GET", path.format(rid=rid), headers=headers, also=agg
            )

        await self.paced(self.cfg.api_rps, read, agg)

    # -- scenario (c): WebSocket subscribers ---------------------------------

    def _on_ws_message(self, raw: str | bytes, probe: bool, rec: Recorder) -> None:
        now = time.time()
        self.ws_messages += 1
        try:
            msg = json.loads(raw)
        except ValueError:
            rec.error("bad json")
            return
        data = msg.get("data") if isinstance(msg, dict) else None
        if not isinstance(data, dict):
            return
        ts = data.get("timestamp")
        if isinstance(ts, (int, float)):
            rec.ok(max(0.0, now - float(ts)))
        else:
            rec.ok()
        if not probe or data.get("event_type") != "resource_updated":
            return
        payload = data.get("data") or {}
        rid = payload.get("resource_id")
        source = data.get("source")
        if source == "mqtt.telemetry_ingestion":
            key: Any = (rid, payload.get("soc"))
            sent = self.pending.pop(key, None)
            if sent is not None:
                self.recorder("ingest_mqtt_e2e").ok(max(0.0, now - sent))
        elif source == "modbus.telemetry_ingestion" and rid in self.modbus_last_seen:
            last = self.modbus_last_seen[rid]
            if last:
                self.recorder("modbus_update_interval").ok(now - last)
            self.modbus_last_seen[rid] = now

    async def ws_subscriber(self, idx: int, client: httpx.AsyncClient) -> None:
        import websockets

        probe = idx == 0
        rec = self.recorder("ws_broadcast")
        url_base = self.base_url.replace("http://", "ws://") + "/api/v1/ws"
        channels = "resource_updates,optimization_events,market_data,alerts"
        first = True
        while not self.stop_event.is_set():
            resp = await self.timed_request(
                client,
                self.recorder("ws_token"),
                "POST",
                "/api/v1/ws/token",
                headers={self._api_key_header: self.api_key},
            )
            if resp is None or resp.status_code != 200:
                await asyncio.sleep(1.0)
                continue
            token = resp.json()["token"]
            try:
                async with websockets.connect(
                    f"{url_base}?token={token}&channels={channels}",
                    max_size=None,
                    open_timeout=30,
                    ping_interval=None,
                ) as ws:
                    self.ws_connected += 1
                    if not first:
                        self.ws_reconnects += 1
                    first = False
                    try:
                        while not self.stop_event.is_set():
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            self._on_ws_message(raw, probe, rec)
                    finally:
                        self.ws_connected -= 1
                    code = getattr(ws, "close_code", None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.stop_event.is_set():
                    return
                code = _close_code(exc)
                if code == 4001:  # credential expired / revoked: reconnect
                    continue
                rec.error(f"ws {type(exc).__name__} code={code}")
                await asyncio.sleep(1.0)
                continue
            if self.stop_event.is_set():
                return
            if code != 4001:
                rec.error(f"ws closed code={code}")

    # -- scenario (d): dispatch ----------------------------------------------

    async def dispatch_loop(self, client: httpx.AsyncClient) -> None:
        rec = self.recorder("dispatch")
        ids = (self.battery_ids + self.modbus_ids)[: max(1, self.cfg.dispatch_resources)]

        async def dispatch() -> None:
            target = round(self.rng.uniform(-0.5, 0.5) * 5.0 * len(ids), 2)
            resp = await self.timed_request(
                client,
                rec,
                "POST",
                "/api/v1/optimization/dispatch",
                json={"target_power_kw": target, "resource_ids": ids},
                headers=await self.bearer(client),
            )
            if resp is not None and resp.status_code == 200:
                body = resp.json()
                self.dispatch_methods[str(body.get("method"))] += 1
                self.dispatch_solve_ms.append(float(body.get("solve_time_ms") or 0.0))

        await self.paced(1.0 / self.cfg.dispatch_interval_s, dispatch, rec)

    # -- sampling -------------------------------------------------------------

    async def sample_loop(self, probe: DbProbe) -> None:
        loop = asyncio.get_running_loop()
        last = loop.time()
        assert self.server is not None
        cpu_last = procinfo.cpu_seconds(procinfo.process_tree(self.server.pid)) or 0.0
        while not self.stop_event.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.stop_event.wait(), self.cfg.sample_interval_s)
            now = loop.time()
            elapsed, last = now - last, now
            self.timeline.append(await self._sample(probe, elapsed, cpu_last))
            cpu_last = self.timeline[-1].get("_cpu_total") or cpu_last
            self._progress(self.timeline[-1])

    async def _sample(self, probe: DbProbe, elapsed: float, cpu_last: float) -> dict[str, Any]:
        assert self.server is not None
        pids = procinfo.process_tree(self.server.pid)
        cpu = procinfo.cpu_seconds(pids)
        wall = time.time()
        # Devices silent for 5 poll intervals (after the first sample) are stale.
        if self.modbus_ids and self.timeline:
            limit = max(5.0, 5 * self.cfg.modbus_poll_s)
            for seen in self.modbus_last_seen.values():
                if not seen or wall - seen > limit:
                    self.recorder("modbus_update_interval").error("stale device")
        db = await probe.sample()
        row: dict[str, Any] = {
            "t": round(time.monotonic() - self.t0, 1),
            "phase": self.phase,
            "server_rss_mb": _mb(procinfo.rss_bytes(pids)),
            "harness_rss_mb": _mb(procinfo.rss_bytes([os.getpid()])),
            "server_cpu_pct": round(100 * (cpu - cpu_last) / elapsed, 1)
            if cpu is not None and elapsed > 0
            else None,
            "_cpu_total": cpu,
            # The harness shares the machine: near 100 % it is the bottleneck.
            "harness_cpu_pct": self._harness_cpu(elapsed),
            "db_mb": _mb(db.get("db_bytes")),
            "rows": db.get("rows"),
            "ws_connected": self.ws_connected,
            "mqtt_pending": len(self.pending),
            "scenarios": {name: r.window(elapsed) for name, r in self.rec.items()},
            "endpoints": {name: r.window(elapsed) for name, r in self.api_rec.items()},
        }
        if "error" in db:
            row["db_error"] = db["error"]
        return row

    def _harness_cpu(self, elapsed: float) -> float | None:
        now = procinfo.cpu_seconds([os.getpid()])
        last, self._harness_cpu_last = self._harness_cpu_last, now
        if now is None or last is None or elapsed <= 0:
            return None
        return round(100 * (now - last) / elapsed, 1)

    def _progress(self, row: dict[str, Any]) -> None:
        sc = row["scenarios"]
        parts = [
            f"t={row['t']:>6.0f}s",
            f"rss={row['server_rss_mb']}MB",
            f"cpu={row['server_cpu_pct']}%",
            f"harness_cpu={row['harness_cpu_pct']}%",
            f"db={row['db_mb']}MB",
        ]
        for name in ("ingest_mqtt_e2e", "ingest_http", "api_read", "ws_broadcast", "dispatch"):
            if name in sc:
                w = sc[name]
                parts.append(f"{name}={w['rate']}/s p95={w['p95_ms']}ms err={w['errors']}")
        self.log("  ".join(parts))

    # -- orchestration --------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        import httpx

        cfg = self.cfg
        self.log(f"[loadtest] output: {self.out}")
        self.log(f"[loadtest] database: {_redact(self.database_url)}")
        await self.migrate()
        if cfg.ingest == "mqtt" and not await wait_tcp(cfg.mqtt_host, cfg.mqtt_port, 3):
            raise RuntimeError(
                f"No MQTT broker at {cfg.mqtt_host}:{cfg.mqtt_port}. Start one "
                "(e.g. `mosquitto -p 1883` or `docker run -p 1883:1883 eclipse-mosquitto:2 "
                "mosquitto -c /mosquitto-no-auth.conf`) or use --ingest http."
            )
        probe = DbProbe(self.database_url)
        try:
            ports = await self.start_simulators()
            setup = await self.boot(ingestion=False, label="setup")
            try:
                await self.register_fleet(ports)
            finally:
                await setup.stop()
            self.modbus_last_seen = dict.fromkeys(self.modbus_ids, 0.0)
            self.log(
                f"[loadtest] fleet: {len(self.battery_ids)} batteries, "
                f"{len(self.modbus_ids)} Modbus PV; starting measured server"
            )
            self.server = await self.boot(ingestion=True, label="run")
            self.t0 = time.monotonic()
            baseline = await self._sample(probe, 1.0, 0.0)
            baseline["t"] = 0.0
            baseline["scenarios"] = {}
            # No interval yet: CPU since process start is not a rate.
            baseline["server_cpu_pct"] = baseline["harness_cpu_pct"] = None
            self.timeline.append(baseline)
            limits = httpx.Limits(max_connections=cfg.max_inflight * 4 + cfg.ws_clients + 8)
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=30, limits=limits
            ) as client:
                await self.login(client)
                tasks = [
                    asyncio.create_task(self.ws_subscriber(i, client))
                    for i in range(cfg.ws_clients if cfg.ws_clients > 0 else 1)
                ]
                for _ in range(100):  # let the sockets connect before load starts
                    if self.ws_connected >= len(tasks):
                        break
                    await asyncio.sleep(0.1)
                if cfg.ingest == "mqtt":
                    tasks.append(asyncio.create_task(self.ingest_mqtt()))
                elif cfg.ingest == "http":
                    tasks.append(asyncio.create_task(self.ingest_http(client)))
                tasks.append(asyncio.create_task(self.api_reads(client)))
                if cfg.dispatch_interval_s > 0 and cfg.dispatch_resources > 0:
                    tasks.append(asyncio.create_task(self.dispatch_loop(client)))
                sampler = asyncio.create_task(self.sample_loop(probe))
                load_started = time.monotonic()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.stop_load.wait(), cfg.duration_s)
                duration = time.monotonic() - load_started
                # Stop issuing load; give queued MQTT messages time to land.
                n_ws = cfg.ws_clients if cfg.ws_clients > 0 else 1
                self.stop_load.set()
                self.phase = "drain"
                await asyncio.gather(*tasks[n_ws:], return_exceptions=True)
                drain_deadline = time.monotonic() + cfg.drain_s
                while self.pending and time.monotonic() < drain_deadline:
                    await asyncio.sleep(0.25)
                self.stop_event.set()
                await asyncio.gather(*tasks, sampler, return_exceptions=True)
                metrics_text = await self._scrape_metrics(client)
            final_db = await probe.sample()
            self._count_lost_mqtt(final_db)
        finally:
            if self.server is not None:
                await self.server.stop()
            for sim in self.sims:
                await sim.stop(5)
            await probe.close()
        return self._report(duration, final_db, metrics_text)

    def _count_lost_mqtt(self, final_db: dict[str, Any]) -> None:
        """Messages never persisted are errors; persisted ones the probe missed are not.

        The probe socket can miss broadcasts while it reconnects (its session
        expires after VPP_JWT_EXPIRE_MINUTES), so the database decides.
        """
        if self.cfg.ingest != "mqtt" or not self.published:
            return
        start = (
            (self.timeline[0].get("rows") or {}).get("battery_states") if self.timeline else None
        )
        end = (final_db.get("rows") or {}).get("battery_states")
        unmatched = len(self.pending)
        if start is None or end is None:
            lost = unmatched
        else:
            lost = max(0, self.published - (end - start))
        self.mqtt_lost = lost
        self.mqtt_unobserved = max(0, unmatched - lost)
        if lost:
            rec = self.recorder("ingest_mqtt_e2e")
            rec.errors += lost
            rec.error_kinds["not persisted"] += lost

    async def _scrape_metrics(self, client: httpx.AsyncClient) -> str:
        with contextlib.suppress(Exception):
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                return resp.text
        return ""

    # -- report ---------------------------------------------------------------

    def _report(
        self, duration: float, final_db: dict[str, Any], metrics_text: str
    ) -> dict[str, Any]:
        cfg = self.cfg
        timeline = [{k: v for k, v in row.items() if k != "_cpu_total"} for row in self.timeline]
        log_stats = _scan_log(self.out / "server-run.log")
        scenarios = {name: r.summary(duration) for name, r in self.rec.items()}
        endpoints = {name: r.summary(duration) for name, r in self.api_rec.items()}
        trends = _trends(timeline, duration)
        start_rows = (timeline[0].get("rows") or {}) if timeline else {}
        end_rows = final_db.get("rows") or {}
        row_growth = {
            t: (end_rows.get(t) or 0) - (start_rows.get(t) or 0)
            for t in COUNTED_TABLES
            if end_rows.get(t) is not None
        }
        errors = sum(r.errors for r in self.rec.values())
        report: dict[str, Any] = {
            "config": {
                **{k: v for k, v in asdict(cfg).items() if k not in ("output_dir", "extra_env")},
                "output_dir": str(self.out),
                "database_url": _redact(self.database_url),
                "dialect": "sqlite" if sqlite_path(self.database_url) else "postgresql",
                "extra_env": cfg.extra_env,
            },
            "machine": procinfo.machine_info(),
            "duration_s": round(duration, 1),
            "fleet": {"batteries": len(self.battery_ids), "modbus_devices": len(self.modbus_ids)},
            "scenarios": scenarios,
            "endpoints": endpoints,
            "mqtt": {
                "published": self.published,
                "not_persisted": self.mqtt_lost,
                # Persisted, but the probe socket missed the broadcast (reconnecting).
                "persisted_not_observed": self.mqtt_unobserved,
            },
            "websocket": {
                "clients": cfg.ws_clients,
                "messages": self.ws_messages,
                "reconnects": self.ws_reconnects,
            },
            "dispatch": {
                "methods": dict(self.dispatch_methods),
                "solve_ms_p50": _pct(self.dispatch_solve_ms, 50),
                "solve_ms_p95": _pct(self.dispatch_solve_ms, 95),
            },
            "db": {
                "start_mb": timeline[0].get("db_mb") if timeline else None,
                "end_mb": _mb(final_db.get("db_bytes")),
                "row_growth": row_growth,
            },
            "server_log": log_stats,
            "trends": trends,
            "total_errors": errors + log_stats["errors"],
            "timeline": timeline,
            "metrics_excerpt": _metrics_excerpt(metrics_text),
        }
        (self.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        from vpp.loadtest.report import render_markdown

        (self.out / "report.md").write_text(render_markdown(report))
        return report


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _mb(v: int | float | None) -> float | None:
    return None if v is None else round(v / 1024 / 1024, 2)


def _pct(values: list[float], q: float) -> float | None:
    from vpp.loadtest.stats import percentile

    p = percentile(sorted(values), q)
    return None if p is None else round(p, 2)


def _redact(url: str) -> str:
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


def _close_code(exc: BaseException) -> int | None:
    rcvd = getattr(exc, "rcvd", None)
    if rcvd is not None:
        return getattr(rcvd, "code", None)
    return getattr(exc, "code", None)


_ERROR_RE = re.compile(r'"level":\s*"(error|critical)"|\b(ERROR|CRITICAL)\b|^Traceback', re.M)
_WARN_RE = re.compile(r'"level":\s*"warning"|\bWARNING\b')
_EVENT_RE = re.compile(r'"event":\s*"([^"]{0,160})')


def _scan_log(path: Path) -> dict[str, Any]:
    errors = warnings = 0
    kinds: Counter[str] = Counter()
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        is_err = bool(_ERROR_RE.search(line))
        is_warn = not is_err and bool(_WARN_RE.search(line))
        if not (is_err or is_warn):
            continue
        if is_err:
            errors += 1
        else:
            warnings += 1
        m = _EVENT_RE.search(line)
        msg = m.group(1) if m else line.strip()[:160]
        kinds[("ERROR " if is_err else "WARNING ") + re.sub(r"\d+", "N", msg)[:120]] += 1
    return {"errors": errors, "warnings": warnings, "top": dict(kinds.most_common(10))}


def _trends(timeline: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """RSS slope and latency creep, ignoring the first 20% of the run (warm-up)."""
    rows = [
        r for r in timeline if r.get("phase") == "load" and r.get("t") and r["t"] >= 0.2 * duration
    ]
    xs = [r["t"] / 3600.0 for r in rows]
    out: dict[str, Any] = {}
    slope = linear_slope(xs, [r.get("server_rss_mb") for r in rows])
    out["server_rss_mb_per_hour"] = None if slope is None else round(slope, 2)
    rss: list[float] = [r["server_rss_mb"] for r in timeline if r.get("server_rss_mb") is not None]
    if rss:
        out["server_rss_mb_start"] = rss[0]
        out["server_rss_mb_peak"] = max(rss)
        out["server_rss_mb_end"] = rss[-1]
    for key in ("server_cpu_pct", "harness_cpu_pct"):
        vals = [r[key] for r in timeline if r.get("phase") == "load" and r.get(key) is not None]
        out[f"{key}_mean"] = round(sum(vals) / len(vals), 1) if vals else None
        out[f"{key}_peak"] = max(vals) if vals else None
    db_slope = linear_slope(xs, [r.get("db_mb") for r in rows])
    out["db_mb_per_hour"] = None if db_slope is None else round(db_slope, 2)
    creep: dict[str, Any] = {}
    names = {n for r in rows for n in (r.get("scenarios") or {})}
    for name in sorted(names):
        p95s = [(r.get("scenarios") or {}).get(name, {}).get("p95_ms") for r in rows]
        creep[name] = thirds_ratio(p95s)
    endpoint_names = {n for r in rows for n in (r.get("endpoints") or {})}
    for name in sorted(endpoint_names):
        p95s = [(r.get("endpoints") or {}).get(name, {}).get("p95_ms") for r in rows]
        creep[f"GET {name}"] = thirds_ratio(p95s)
    out["p95_last_third_over_first_third"] = creep
    return out


_METRIC_PREFIXES = (
    "vpp_api_requests_in_progress",
    "vpp_telemetry_events_total",
    "vpp_optimization_runs_total",
    "process_resident_memory_bytes",
)


def _metrics_excerpt(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line and not line.startswith("#") and line.startswith(_METRIC_PREFIXES)
    ][:40]


def run_load(cfg: LoadConfig) -> dict[str, Any]:
    """Run the harness synchronously and return the report dict."""
    return asyncio.run(LoadRun(cfg).run())
