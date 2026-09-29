"""VPP command-line interface built with Click."""

from __future__ import annotations

import json
import sys

import click


@click.group()
@click.version_option(package_name="virtual-power-plant")
def cli() -> None:
    """Virtual Power Plant — production-grade energy management platform."""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--host", default=None, help="Bind host (default: VPP_API_HOST)")
@click.option("--port", default=None, type=int, help="Bind port (default: VPP_API_PORT)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option(
    "--workers",
    default=None,
    type=click.IntRange(min=1),
    help="Number of worker processes (default: VPP_API_WORKERS)",
)
def serve(host: str | None, port: int | None, reload: bool, workers: int | None) -> None:
    """Start the VPP API server.

    With more than one worker, singleton background work runs on the holder
    of a database lease and trading commands are forwarded to it; see
    docs/architecture.md#process-model.
    """
    import os

    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required: pip install virtual-power-plant[api]", err=True)
        sys.exit(1)

    from vpp.cluster.topology import TopologyError, validate_topology
    from vpp.settings import get_settings

    settings = get_settings()
    effective = workers if workers is not None else settings.api_workers
    if reload and effective > 1:
        click.echo("--reload runs a single process; ignoring the worker count", err=True)
        effective = 1
    try:
        validate_topology(settings.model_copy(update={"api_workers": effective}))
    except TopologyError as exc:
        click.echo(f"Refusing to start: {exc}", err=True)
        sys.exit(2)
    # Worker processes read the effective count (topology checks + startup log).
    os.environ["VPP_API_WORKERS"] = str(effective)
    get_settings.cache_clear()

    uvicorn.run(
        "vpp.api.app:create_app",
        host=host if host is not None else settings.api_host,
        port=port if port is not None else settings.api_port,
        reload=reload,
        workers=effective,
        factory=True,
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@cli.command()
def init() -> None:
    """Initialise the database and create default configuration."""
    import asyncio

    from vpp.db.engine import init_db
    from vpp.settings import get_settings

    async def _init() -> None:
        settings = get_settings()
        await init_db(settings.database_url)
        click.echo(f"Database initialised ({settings.database_url})")

    asyncio.run(_init())


@cli.command()
@click.option("--revision", default="head", show_default=True, help="Target revision.")
def migrate(revision: str) -> None:
    """Run pending database migrations (alembic upgrade)."""
    from vpp.db.engine import run_alembic_upgrade
    from vpp.settings import get_settings

    settings = get_settings()
    run_alembic_upgrade(settings.database_url, revision)
    click.echo(f"Database migrated to {revision} ({settings.database_url})")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Only count the rows that would be deleted.")
def prune(dry_run: bool) -> None:
    """Delete rows older than their VPP_*_RETENTION_DAYS setting, once.

    Runs the same pass the API runs periodically on the data-retention lease
    holder (see docs/security.md#data-retention). A retention of 0 keeps a
    table forever.
    """
    import asyncio

    from vpp.retention import prune as run_prune
    from vpp.settings import get_settings

    settings = get_settings()
    results = asyncio.run(_with_factory(lambda f: run_prune(f, settings, dry_run=dry_run)))
    verb = "would delete" if dry_run else "deleted"
    for r in results:
        if r.enabled:
            click.echo(
                f"  {r.table:<20} {verb} {r.rows:>8} row(s) older than {r.retention_days} days"
            )
        else:
            click.echo(f"  {r.table:<20} kept forever ({r.env_var}=0)")
    total = sum(r.rows for r in results)
    click.echo(f"{'Dry run: ' if dry_run else ''}{total} row(s) {verb} in total")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

_PASSWORD_ENV = "VPP_ADMIN_PASSWORD"


def _read_new_password(password_stdin: bool, password_file: str | None) -> str:
    """Password from stdin, a file, $VPP_ADMIN_PASSWORD, or an interactive prompt."""
    import os

    from vpp.auth.bootstrap import read_password_file

    if password_stdin:
        return sys.stdin.readline().removesuffix("\n").removesuffix("\r")
    if password_file:
        return read_password_file(password_file)
    env = os.environ.get(_PASSWORD_ENV)
    if env:
        return env
    if not sys.stdin.isatty():
        raise click.UsageError(
            f"No password given: use --password-stdin, --password-file or ${_PASSWORD_ENV}"
        )
    return str(click.prompt("Password", hide_input=True, confirmation_prompt=True))


async def _with_factory(fn):
    """Run ``fn(session_factory)`` against the configured database.

    Uses a private engine (not the process-global one) and prepares the
    schema the same way the API does: ``alembic upgrade head`` when
    ``VPP_USE_ALEMBIC`` is set, else ``create_all``.
    """
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from vpp.db import models as _models  # noqa: F401  (register tables)
    from vpp.db.base import Base
    from vpp.db.engine import create_engine_from_settings, run_alembic_upgrade
    from vpp.settings import get_settings

    settings = get_settings()
    engine = create_engine_from_settings(settings.database_url)
    try:
        if settings.use_alembic:
            await asyncio.to_thread(run_alembic_upgrade, settings.database_url)
        else:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        return await fn(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        await engine.dispose()


async def _with_session(fn):
    """Run ``fn(session)`` against the configured database, then commit."""

    async def _run(factory):
        async with factory() as session:
            result = await fn(session)
            await session.commit()
            return result

    return await _with_factory(_run)


_password_options = [
    click.option("--password-stdin", is_flag=True, help="Read the password from stdin."),
    click.option(
        "--password-file",
        type=click.Path(exists=True, dir_okay=False),
        help="Read the password from a file (e.g. a mounted secret).",
    ),
]


def _with_password_options(fn):
    for option in reversed(_password_options):
        fn = option(fn)
    return fn


@cli.group("users")
def users_group() -> None:
    """Manage user accounts (works without a running API)."""


@users_group.command("create-admin")
@click.argument("username")
@_with_password_options
def users_create_admin(username: str, password_stdin: bool, password_file: str | None) -> None:
    """Create an admin account USERNAME.

    The password is read from --password-stdin, --password-file,
    $VPP_ADMIN_PASSWORD or an interactive prompt, in that order, and must
    satisfy the password policy. The password is never echoed or logged.
    """
    import asyncio

    from vpp.auth.bootstrap import BootstrapError, create_admin

    password = _read_new_password(password_stdin, password_file)
    try:
        asyncio.run(_with_session(lambda s: create_admin(s, username, password)))
    except BootstrapError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Created admin user {username!r}")


@users_group.command("set-password")
@click.argument("username")
@_with_password_options
def users_set_password(username: str, password_stdin: bool, password_file: str | None) -> None:
    """Set USERNAME's password (e.g. a locked-out admin) and revoke their sessions.

    Also re-activates the account.
    """
    import asyncio

    from vpp.auth.passwords import password_problem
    from vpp.auth.security import get_password_hash, revoke_user_sessions
    from vpp.db.repositories import UserRepository

    password = _read_new_password(password_stdin, password_file)
    problem = password_problem(password, username=username)
    if problem:
        raise click.ClickException(problem)

    async def _set(session) -> bool:
        user = await UserRepository.get_by_username(session, username)
        if user is None:
            return False
        user.hashed_password = get_password_hash(password)
        user.is_active = True
        revoke_user_sessions(user)
        return True

    if not asyncio.run(_with_session(_set)):
        raise click.ClickException(f"No user named {username!r}")
    click.echo(f"Password of {username!r} updated; existing sessions revoked")


@users_group.command("list")
def users_list() -> None:
    """List user accounts."""
    import asyncio

    from sqlalchemy import select

    from vpp.db.models import UserModel

    async def _list(session):
        return list(
            (await session.execute(select(UserModel).order_by(UserModel.username))).scalars()
        )

    users = asyncio.run(_with_session(_list))
    if not users:
        click.echo("No users. Create one with: vpp users create-admin <username>")
        return
    for u in users:
        click.echo(f"  {u.username:<32} {u.role:<11} {'active' if u.is_active else 'INACTIVE'}")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@cli.group("resource")
def resource_group() -> None:
    """Manage energy resources."""


@resource_group.command("list")
def resource_list() -> None:
    """List all registered resources."""
    import asyncio

    from vpp.db.engine import get_db, init_db
    from vpp.db.repositories import ResourceRepository
    from vpp.settings import get_settings

    async def _list() -> None:
        settings = get_settings()
        await init_db(settings.database_url)
        async for session in get_db():
            items = await ResourceRepository.list_all(session)
            if not items:
                click.echo("No resources registered.")
                return
            for r in items:
                click.echo(
                    f"  [{r.resource_type:>12}] {r.name:<30} {r.rated_power:>8.1f} kW  {'ONLINE' if r.online else 'OFFLINE'}"
                )

    asyncio.run(_list())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("target_power", type=float)
def dispatch(target_power: float) -> None:
    """Run dispatch optimisation for TARGET_POWER kW."""
    from vpp.config import VPPConfig
    from vpp.core import VirtualPowerPlant

    vpp = VirtualPowerPlant(config=VPPConfig())
    success = vpp.optimize_dispatch(target_power)
    status = "SUCCESS" if success else "FAILED"
    total = vpp.get_total_power()
    click.echo(f"Dispatch {status}: target={target_power:.1f} kW  actual={total:.1f} kW")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Show VPP platform status."""
    from vpp.settings import get_settings

    settings = get_settings()
    click.echo("=== Virtual Power Plant Platform ===")
    click.echo(f"  Environment : {settings.env}")
    click.echo(
        f"  Database    : {'PostgreSQL' if 'postgresql' in settings.database_url else 'SQLite'}"
    )
    click.echo(f"  API         : http://{settings.api_host}:{settings.api_port}")
    click.echo(f"  Metrics     : {'enabled' if settings.metrics_enabled else 'disabled'}")
    click.echo(f"  Log level   : {settings.log_level}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@cli.group("config")
def config_group() -> None:
    """Configuration management."""


@config_group.command("show")
def config_show() -> None:
    """Display current platform configuration."""
    from vpp.settings import get_settings

    settings = get_settings()
    click.echo(json.dumps(settings.model_dump(exclude={"secret_key"}), indent=2, default=str))


@config_group.command("validate")
@click.argument("path", type=click.Path(exists=True))
def config_validate(path: str) -> None:
    """Validate a YAML/JSON configuration file."""
    import yaml

    with open(path) as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    click.echo(f"Loaded configuration from {path}")
    click.echo(f"  Keys: {list(data.keys())}")
    click.echo("  Validation: OK (basic structure check passed)")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@cli.group("benchmark")
def benchmark_group() -> None:
    """Run and manage VPP benchmarks."""


@benchmark_group.command("list")
def benchmark_list() -> None:
    """List available benchmark scenarios and datasets."""
    from vpp.benchmarks.datasets import DatasetRegistry
    from vpp.benchmarks.scenarios import ScenarioRegistry

    click.echo("=== Datasets ===")
    for name in DatasetRegistry.list_all():
        ds = DatasetRegistry.get(name)
        s = ds.spec()
        click.echo(
            f"  {name:<25} {s.duration_hours:>4}h @ {s.resolution_minutes}min  ({s.n_steps} steps)"
        )

    click.echo("\n=== Scenarios ===")
    for name in ScenarioRegistry.list_all():
        sc = ScenarioRegistry.get(name)
        click.echo(f"  {name:<35} [{sc.category.value}]")
        click.echo(f"    {sc.description[:80]}")


@benchmark_group.command("run")
@click.argument("scenario_name")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
def benchmark_run(scenario_name: str, seed: int) -> None:
    """Run a benchmark scenario with all built-in methods."""
    from vpp.benchmarks.runner import (
        BenchmarkMethod,
        BenchmarkRunner,
        NoOpMethod,
        RuleBasedPeakShaving,
        SimpleV2GScheduler,
    )
    from vpp.benchmarks.scenarios import ScenarioRegistry

    scenario = ScenarioRegistry.get(scenario_name)
    click.echo(f"Running scenario: {scenario_name}")
    click.echo(f"  {scenario.description}\n")

    runner = BenchmarkRunner()
    methods: list[BenchmarkMethod] = [NoOpMethod(), RuleBasedPeakShaving(), SimpleV2GScheduler()]

    for method in methods:
        try:
            result = runner.run(scenario_name, method, seed=seed)
            click.echo(f"  [{method.name}] solve={result.solve_time_s * 1000:.1f}ms")
            for k, v in sorted(result.metrics.values.items()):
                click.echo(f"    {k:<35} {v:>12.4f}")
        except Exception as e:
            click.echo(f"  [{method.name}] ERROR: {e}")

    click.echo("\n" + runner.generate_report(f"Benchmark: {scenario_name}"))


@benchmark_group.command("report")
@click.option("--scenario", default=None, help="Run specific scenario (default: all)")
@click.option("--seeds", default="42", help="Comma-separated seeds")
def benchmark_report(scenario: str | None, seeds: str) -> None:
    """Generate a full benchmark comparison report."""
    from vpp.benchmarks.runner import (
        BenchmarkMethod,
        BenchmarkRunner,
        NoOpMethod,
        RuleBasedPeakShaving,
        SimpleV2GScheduler,
    )
    from vpp.benchmarks.scenarios import ScenarioRegistry

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    methods: list[BenchmarkMethod] = [NoOpMethod(), RuleBasedPeakShaving(), SimpleV2GScheduler()]

    runner = BenchmarkRunner()
    scenario_names = [scenario] if scenario else ScenarioRegistry.list_all()

    for name in scenario_names:
        click.echo(f"Running {name}...")
        for method in methods:
            for seed in seed_list:
                try:
                    runner.run(name, method, seed=seed)
                except Exception as e:
                    click.echo(f"  [{method.name}] seed={seed} ERROR: {e}")

    report = runner.generate_report()
    click.echo("\n" + report)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--horizon", default=24, type=int, help="MPC look-ahead steps")
@click.option("--ticks", default=24, type=int, help="Number of MPC ticks to run")
@click.option("--interval", default=60, type=int, help="Tick interval in minutes")
@click.option("--no-warm-start", is_flag=True, help="Disable warm-starting the solver")
def mpc(horizon: int, ticks: int, interval: int, no_warm_start: bool) -> None:
    """Run a quick MPC demo over a synthetic CAISO-style price profile."""
    import math
    from datetime import datetime, timedelta

    from vpp.optimization.backtest import BacktestConfig, run_backtest
    from vpp.optimization.mpc import MPCConfig, MPCController

    battery = {
        "battery_capacity_kwh": 100.0,
        "max_charge_kw": 50.0,
        "max_discharge_kw": 50.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
    }
    prices = [
        50.0 + 30.0 * math.sin(2 * math.pi * ((k % 24) - 4) / 24.0) for k in range(ticks + horizon)
    ]
    cfg = MPCConfig(
        horizon_steps=horizon,
        interval_minutes=interval,
        warm_start=not no_warm_start,
    )
    ctrl = MPCController(cfg, battery)
    start = datetime(2025, 1, 1)
    bt_cfg = BacktestConfig(
        start=start,
        end=start + timedelta(minutes=interval * ticks),
        interval_minutes=interval,
    )

    def perfect(now: datetime, H: int):
        k = int((now - start).total_seconds() // (interval * 60))
        sl = prices[k : k + H]
        if len(sl) < H:
            sl = sl + [sl[-1]] * (H - len(sl))
        return {"prices": sl}

    res = run_backtest(ctrl, bt_cfg, prices[:ticks], [0.0] * ticks, [0.0] * ticks, perfect)
    click.echo(
        json.dumps(
            {
                "ticks": ticks,
                "horizon": horizon,
                "interval_minutes": interval,
                "warm_start": not no_warm_start,
                "realized_cost": round(res.realized_cost, 4),
                "wall_time_s": round(res.wall_time_s, 3),
                "avg_solve_ms": round(res.cumulative_solve_time_ms / max(ticks, 1), 2),
                "fallback_count": res.fallback_count,
                "final_soc_kwh": round(res.soc_trajectory[-1], 3) if res.soc_trajectory else None,
            },
            indent=2,
        )
    )


@cli.command()
@click.argument("demo_name", required=False, default=None)
def demo(demo_name: str | None) -> None:
    """Run a demo application. Without arguments, lists available demos."""
    demos_available = {
        "residential": "Residential VPP — 10 homes with solar + battery",
        "ev_fleet": "EV Fleet V2G — 50-vehicle parking garage",
        "microgrid": "Microgrid Islanding — grid fault and island transition",
        "trading": "Trading Bot — automated multi-market arbitrage",
        "protocols": "Multi-Protocol — OpenADR + OCPP + MQTT + Modbus",
        "dashboard": "Interactive Dashboard — live terminal UI",
    }

    if demo_name is None:
        click.echo("Available demos:")
        for name, desc in demos_available.items():
            click.echo(f"  {name:<15} {desc}")
        click.echo("\nRun with: vpp demo <name>")
        return

    if demo_name not in demos_available:
        click.echo(f"Unknown demo: {demo_name}. Available: {', '.join(demos_available)}", err=True)
        sys.exit(1)

    import importlib

    module = importlib.import_module(f"vpp.demos.{demo_name}_demo")
    module.run()


# ---------------------------------------------------------------------------
# Device simulators
# ---------------------------------------------------------------------------


@cli.group("simulate")
def simulate_group() -> None:
    """Simulated devices for testing without hardware."""


@simulate_group.command(
    "sunspec",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def simulate_sunspec(args: tuple[str, ...]) -> None:
    """Serve a SunSpec PV + battery inverter over Modbus TCP (--help for options)."""
    try:
        import pymodbus  # noqa: F401
    except ImportError:
        click.echo("pymodbus is required: pip install virtual-power-plant[protocols]", err=True)
        sys.exit(1)
    from vpp.simulators.sunspec import main

    sys.exit(main(list(args), prog="vpp simulate sunspec"))
