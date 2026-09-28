# Library features

The `vpp` package can be used as a Python library, without the API server.
This page describes the building blocks that exist today, with snippets
that run against the current code, and is explicit about what is only a
configuration field without behaviour behind it.

For the operational platform (API, console, protocols) start with the
[README](README.md) and [docs/](docs/). Maturity labels (production-grade /
beta / simulated / research) are defined in the README.

Install with the extras you need, e.g.
`pip install -e ".[solver,degradation]"` (Pyomo + HiGHS, rainflow).

## Contents

- [Platform configuration document](#platform-configuration-document)
- [Battery models](#battery-models)
- [Optimization framework](#optimization-framework)
- [Battery degradation](#battery-degradation)
- [Tariff engine](#tariff-engine)
- [Trading package](#trading-package)
- [V2G, grid and research modules](#v2g-grid-and-research-modules)
- [Not implemented](#not-implemented)

## Platform configuration document

`vpp.config.VPPConfig` is a typed, validated description of a VPP:
optimization objectives and constraints, heuristic and rule-engine
settings, monitoring, simulation, security and resources. It loads from
YAML or JSON and validates with detailed errors. The API serves the same
structure as a strict JSON Schema (`GET /api/v1/config/schema`) and applies
documents with `PUT /api/v1/config` (versioned in the database).

```python
from vpp.config import VPPConfig

config = VPPConfig.load_from_file("configs/advanced_vpp_config.yaml")
assert config.validate_and_log()
print(config.name, [r.name for r in config.resources])
```

`configs/advanced_vpp_config.yaml` is a complete example. Note that several
sections are **descriptive only** today — see [Not implemented](#not-implemented).

## Battery models

`vpp.models.battery` provides two cell-level simulation models behind one
factory. **Maturity: research** (used by examples, not by the API's
optimizer, which uses its own linear battery model and the degradation
package).

- `SimpleEquivalentCircuitModel` — SOC, voltage, internal resistance,
  efficiency, simple thermal and ageing terms.
- `AdvancedElectrochemicalModel` — adds lithium concentration dynamics,
  Butler-Volmer kinetics and more detailed ageing; parameters are
  illustrative, not fitted to a specific cell.

```python
from vpp.models.battery import BatteryParameters, create_battery_model

params = BatteryParameters(
    nominal_capacity=200.0,    # Ah
    nominal_voltage=400.0,     # V
    max_voltage=450.0,         # V
    min_voltage=320.0,         # V
    max_current=100.0,         # A
    internal_resistance=0.01,  # Ohm
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
)
battery = create_battery_model("simple", params, config.resources[0])  # or "advanced"
for hour in range(3):
    state = battery.update(power_setpoint=10.0, dt=3600.0)  # kW, seconds
    print(f"hour {hour}: SOC={state.soc:.3f} T={state.temperature:.1f}C")
```

## Optimization framework

`vpp.optimization` combines exact solvers with rule-based fallbacks behind
a plugin interface (`OptimizationPlugin`, `OptimizationEngine`,
`solve_with_fallback`). If a plugin is unavailable, times out or returns an
invalid solution, the engine falls back to rules and says so in the result
status (`FALLBACK_USED`).

| Component | What it is | Maturity |
|---|---|---|
| `planning.build_allocation_problem` + `solvers/allocation_plugin.py` | single-interval fleet allocation LP (Pyomo + HiGHS) used by `POST /api/v1/optimization/dispatch` and the DR orchestrator | beta |
| `mpc.MPCController`, `MultiResourceMPCController` | rolling-horizon MPC (Pyomo + HiGHS) with warm start, wear cost, tariff hooks, terminal SOC | beta |
| `backtest.run_backtest` | closed-loop MPC replay with perfect / persistence / noisy forecasts and baselines | beta |
| `solvers/stochastic_plugin.py` | scenario-based stochastic dispatch with CVaR | research |
| `realtime` | fast rule-based dispatch and an MPC plugin for grid-service signals | research |
| `distributed` | ADMM coordination across sites plus consensus rules | research |

Try MPC from the command line (synthetic CAISO-like prices):

```bash
vpp mpc --horizon 24 --ticks 24
```

Solvers: Pyomo with HiGHS (`highspy`) is the supported backend (`solver`
extra). PuLP is a core dependency used by older code paths. Without the
`solver` extra the API falls back to rule-based allocation.

## Battery degradation

`vpp.degradation` (**beta**): throughput, calendar and rainflow
(cycle-counting) degradation models with LFP / NMC presets, a
`DegradationUpdater` that turns stored SOC history into state-of-health
samples (`battery_soh_samples`), and wear-cost hooks that keep the
optimizer's cost of cycling consistent with the SOH model. The API runs
the updater periodically (`VPP_DEGRADATION_UPDATER_*`).

## Tariff engine

`vpp.tariffs` (**beta**) parses URDB JSON into components (TOU energy with
per-period tiers and sell rates, tiered energy, TOU/flat demand, fixed,
minimum, adders, taxes) and bills interval data, including billing cycles
and NEM export credit. Full rules: [docs/tariffs.md](docs/tariffs.md).

```python
from datetime import datetime, timedelta, timezone

from vpp.tariffs import MeterTrace, load_urdb_json
from vpp.tariffs.simulation import simulate_bill

tariff = load_urdb_json("src/vpp/tariffs/presets/pge_etouc.json")
start = datetime(2026, 7, 1, tzinfo=timezone.utc)
hours = 24 * 31
trace = MeterTrace(
    timestamps=[start + timedelta(hours=h) for h in range(hours)],
    import_kwh=[0.8 if 16 <= h % 24 < 21 else 0.4 for h in range(hours)],
    export_kwh=[1.5 if 11 <= h % 24 < 14 else 0.0 for h in range(hours)],
)
result = simulate_bill(tariff, trace, start, start + timedelta(hours=hours), nem_regime="nem2")
print(f"total ${result.total:.2f}, export credit ${result.export_credit:.2f}")
for item in result.line_items:
    print(f"  {item.label}: {item.amount:.2f}")
```

Price feeds in `vpp.tariffs.feeds` (CAISO LMP, ComEd hourly pricing,
OpenADR price signals, synthetic) plug into tariff-to-optimizer conversion;
they are library features and are not polled by the API.

## Trading package

`vpp.trading` (**simulated**): markets (day-ahead, real-time, ancillary,
bilateral), order books with market / limit / stop / stop-limit / iceberg /
IOC / FOK orders, a trading engine, portfolio accounting (realized and
unrealized P&L, fees, drawdown), a risk manager (position, daily loss,
drawdown, VaR, concentration limits) and strategies (arbitrage, momentum,
mean reversion, a machine-learning placeholder, multi-market) with
backtests. The API exposes it on a simulated venue with synthetic
liquidity; there is no connection to a real exchange or ISO market.

## V2G, grid and research modules

- `vpp.v2g` (**beta** through the API): EV and fleet models, a scheduler
  (TOU-aware, departure-SOC constrained), an aggregator for flexibility
  windows and bids, and the persistent store / OCPP bridge used by the API.
- `vpp.grid` (**simulated**): grid-forming inverter models (droop control,
  virtual synchronous machine, virtual inertia) and a microgrid controller
  (fault detection, islanding, reconnection, load shedding) used by the
  microgrid demo.
- `vpp.research` (**research**): baseline forecasters (persistence,
  linear, exponential smoothing, ensemble), anomaly detectors (Z-score,
  IQR, moving average) and a seeded experiment runner. Nothing in the
  operational path depends on it. There is no Gaussian-process,
  reinforcement-learning, federated-learning or digital-twin code, despite
  what older release notes said.
- `vpp.benchmarks` (**research**): synthetic datasets (residential, CAISO-like,
  EU multi-zone, EV fleet), scenarios, metrics and a runner
  (`vpp benchmark run PEAK_SHAVING`).

## Not implemented

These appear in the configuration schema or older docs but have no
behaviour behind them yet:

- **Rule engine execution.** `rules` (inference method, conflict
  resolution, rule conditions/actions) is validated and stored, but no
  engine evaluates those rules. Alerting rules are a separate, working
  feature (`/api/v1/alerts/rules`).
- **Heuristic solvers** such as genetic algorithms or particle swarm
  optimization: only configuration fields exist.
- **`enable_hot_reload`, `security.encryption_enabled`,
  `backup_config`**: flags only. Configuration changes are applied through
  `PUT /api/v1/config`, not by watching files; `VPPConfig.backup_to_file()`
  exists but is not called automatically.
- **Multi-objective optimization**: `optimization.objectives` and
  `constraints` are validated and stored, but no optimizer reads them; the
  dispatch LP and MPC have fixed cost functions (energy cost, wear cost,
  penalties).
