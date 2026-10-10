# Optimization

Receding-horizon model predictive control and its closed-loop backtest
harness (the code behind `POST /api/v1/optimization/schedule` and
`/backtest`), and the plugin framework with rule-based fallbacks that the
stochastic, real-time and distributed optimizers build on.

```python
from vpp.optimization.mpc import MPCConfig, MPCController
from vpp.optimization.backtest import BacktestConfig, run_backtest
from vpp.optimization import OptimizationEngine, OptimizationProblem
```

::: vpp.optimization.mpc

::: vpp.optimization.backtest

::: vpp.optimization.base
