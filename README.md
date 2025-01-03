<div align="center">

# 🔋 Virtual Power Plant Library

A comprehensive Python library for virtual power plant management, simulation, and research.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)]()

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Contributing](#-contributing)

</div>

---

## ✨ Features

### 🏭 Core Components
- **🔋 Resource Management**
  - Battery storage with charge/discharge cycles and degradation modeling
  - ☀️ Solar PV with weather-dependent generation and panel efficiency
  - 🌪️ Wind turbines with detailed physical modeling and power curves
  - 🔌 Extensible resource framework for custom implementations

- **🎯 Advanced Optimization**
  - 📋 Rule-based optimization with customizable rules
  - 📊 Linear/Mixed Integer Programming (LP/MILP)
  - 🤖 Reinforcement Learning framework (future)
  - ⚖️ Constraint handling and multi-objective optimization

### 🔬 Research Capabilities
- **🧪 Simulation Framework**
  - 🌤️ Weather pattern generation
  - 📈 Market price simulation
  - 🔄 Resource behavior modeling
  - 📉 Time series analysis

- **📐 Analysis Tools**
  - 📊 Performance metrics calculation
  - 🔍 Strategy comparison
  - 📋 Resource utilization analysis
  - ⚠️ Constraint violation tracking

- **📊 Visualization**
  - 📈 Power output plots
  - 📊 Performance dashboards
  - 📉 Strategy comparison charts
  - 📊 Time series analysis

### 🏢 Production Features
- **🛡️ Reliability**
  - ✅ Comprehensive validation
  - 🚫 Error handling
  - 📝 State tracking
  - 📊 Performance monitoring

## 🚀 Installation

```bash
# ⚡ Basic installation
pip install virtual-power-plant

# 🔬 With research tools
pip install virtual-power-plant[research]

# 🛠️ Development installation
pip install virtual-power-plant[dev]
```

<details>
<summary>📦 Dependencies</summary>

- Core:
  - `numpy>=1.20.0`
  - `pulp>=2.7.0`

- Research:
  - `matplotlib>=3.4.0`
  - `pandas>=1.3.0`
  - `scipy>=1.7.0`
  - `torch>=1.9.0`

- Development:
  - `pytest>=6.0.0`
  - `black>=21.5b2`
  - `mypy>=0.900`
</details>

## 📖 Usage Examples

### 🏭 Production Use

```python
from vpp import VirtualPowerPlant
from vpp.resources import Battery, Solar, WindTurbine
from vpp.optimization import LinearProgrammingStrategy
from vpp.config import VPPConfig

# Initialize VPP with LP strategy
vpp = VirtualPowerPlant(
    config=VPPConfig(name="Production VPP"),
    strategy=LinearProgrammingStrategy()
)

# Add resources
battery = Battery(
    capacity=2000,      # 2 MWh
    current_charge=1500, # 75% charged
    max_power=500,      # 500 kW
    nominal_voltage=400  # 400V
)

solar = Solar(
    peak_power=1000,    # 1 MW
    panel_area=5000,    # 5000 m²
    efficiency=0.22     # 22% efficiency
)

wind = WindTurbine(
    rated_power=2000,   # 2 MW
    rotor_diameter=70,  # 70m rotor
    hub_height=80,      # 80m height
    cut_in_speed=3.0,   # Min wind speed
    cut_out_speed=25.0, # Max wind speed
    rated_speed=12.0    # Rated wind speed
)

# Add resources and optimize
vpp.add_resource(battery)
vpp.add_resource(solar)
vpp.add_resource(wind)

result = vpp.optimize_dispatch(target_power=3000)  # Target 3 MW
print(f"Optimization success: {result.success}")
print(f"Power allocation: {result.resource_allocation}")
```

### 🔬 Research and Analysis

```python
from vpp import VirtualPowerPlant
from vpp.simulation import Simulator, SimulationConfig
from vpp.analysis import PerformanceAnalyzer
from vpp.visualization import PowerPlotter, StrategyPlotter
from datetime import datetime, timedelta

# Configure simulation
config = SimulationConfig(
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(days=7),
    time_step=timedelta(minutes=15),
    include_weather=True,
    include_market=True
)

# Create simulator
simulator = Simulator(config, vpp.resources, vpp.strategy)

# Run simulation with varying target power
def target_power_func(timestamp: datetime) -> float:
    hour = timestamp.hour
    if 9 <= hour <= 20:  # Peak hours
        return 3000
    return 1500

metrics = simulator.run(target_power_func)

# Analyze results
analyzer = PerformanceAnalyzer()
system_perf = analyzer.analyze_system(
    simulator.states,
    simulator.optimization_results
)

# Visualize results
plotter = PowerPlotter()
power_fig = plotter.plot_power_output(simulator.states)
power_fig.savefig('power_output.png')

strategy_plotter = StrategyPlotter()
comparison_fig = strategy_plotter.plot_strategy_comparison({
    'LP': simulator.optimization_results
})
comparison_fig.savefig('strategy_comparison.png')
```

### 🔧 Custom Optimization Strategy

```python
from vpp.optimization import OptimizationStrategy, OptimizationResult
from typing import List

class CustomStrategy(OptimizationStrategy):
    def optimize(
        self,
        resources: List[EnergyResource],
        target_power: float
    ) -> OptimizationResult:
        # Your custom optimization logic here
        allocations = {}
        for resource in resources:
            # Example: Simple proportional allocation
            power = (resource.rated_power / sum(r.rated_power for r in resources)) * target_power
            allocations[resource.__class__.__name__] = min(power, resource.rated_power)
        
        return OptimizationResult(
            success=True,
            target_power=target_power,
            actual_power=sum(allocations.values()),
            resource_allocation=allocations
        )

# Use custom strategy
vpp = VirtualPowerPlant(
    config=VPPConfig(name="Custom VPP"),
    strategy=CustomStrategy()
)
```

## 📚 Documentation

Work in Progress ...

## 🤝 Contributing

We welcome contributions! Areas of interest:
- 🔋 New resource types
- 🎯 Optimization strategies
- 📊 Analysis tools
- 📖 Documentation
- 💡 Examples and tutorials

See [Contributing Guide](https://github.com/vinerya/virtual-power-plant/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made with Love](https://img.shields.io/badge/Made%20with-Love-red.svg)](https://github.com/vinerya)

Made with ❤️ by Moudather Chelbi & Mariem Khemir

</div>
