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
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
def serve(host: str, port: int, reload: bool, workers: int) -> None:
    """Start the VPP API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required: pip install virtual-power-plant[api]", err=True)
        sys.exit(1)

    uvicorn.run(
        "vpp.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
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
def migrate() -> None:
    """Run pending database migrations (via Alembic)."""
    click.echo("Migrations will be available after alembic is configured.")


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
    from vpp.db.engine import init_db, get_db
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
                click.echo(f"  [{r.resource_type:>12}] {r.name:<30} {r.rated_power:>8.1f} kW  {'ONLINE' if r.online else 'OFFLINE'}")

    asyncio.run(_list())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("target_power", type=float)
def dispatch(target_power: float) -> None:
    """Run dispatch optimisation for TARGET_POWER kW."""
    from vpp.core import VirtualPowerPlant
    from vpp.config import VPPConfig

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
    click.echo(f"  Database    : {'PostgreSQL' if 'postgresql' in settings.database_url else 'SQLite'}")
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
    from benchmarks.datasets import DatasetRegistry
    from benchmarks.scenarios import ScenarioRegistry

    click.echo("=== Datasets ===")
    for name in DatasetRegistry.list_all():
        ds = DatasetRegistry.get(name)
        s = ds.spec()
        click.echo(f"  {name:<25} {s.duration_hours:>4}h @ {s.resolution_minutes}min  ({s.n_steps} steps)")

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
    from benchmarks.runner import BenchmarkRunner, NoOpMethod, RuleBasedPeakShaving, SimpleV2GScheduler
    from benchmarks.scenarios import ScenarioRegistry

    scenario = ScenarioRegistry.get(scenario_name)
    click.echo(f"Running scenario: {scenario_name}")
    click.echo(f"  {scenario.description}\n")

    runner = BenchmarkRunner()
    methods = [NoOpMethod(), RuleBasedPeakShaving(), SimpleV2GScheduler()]

    for method in methods:
        try:
            result = runner.run(scenario_name, method, seed=seed)
            click.echo(f"  [{method.name}] solve={result.solve_time_s*1000:.1f}ms")
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
    from benchmarks.runner import BenchmarkRunner, NoOpMethod, RuleBasedPeakShaving, SimpleV2GScheduler
    from benchmarks.scenarios import ScenarioRegistry

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    methods = [NoOpMethod(), RuleBasedPeakShaving(), SimpleV2GScheduler()]

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

    # Import and run demo
    try:
        module = __import__(f"demos.{demo_name}_demo", fromlist=["run"])
        module.run()
    except ImportError as e:
        click.echo(f"Could not load demo '{demo_name}': {e}", err=True)
        click.echo("Make sure you are running from the project root.", err=True)
        sys.exit(1)
