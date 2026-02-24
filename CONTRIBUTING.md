# Contributing to Virtual Power Plant

We love your input! We want to make contributing to the Virtual Power Plant platform as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Development Process
We use GitHub Flow, so all code changes happen through pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes (`pytest tests/ -v` — all 226 tests must pass)
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Run the test suite:
```bash
pytest tests/ -v
```

## Code Style
- We use [Black](https://github.com/psf/black) for code formatting
- Type hints are required for all functions and classes
- Docstrings should follow Google style
- Maximum line length is 88 characters (Black default)

## Architecture Principles

- **Rule-based logic for production** — AI/ML goes in `src/vpp/research/` only, never in production code paths
- **Plugin architecture** — new optimization methods, protocols, and strategies should be plugins
- **Edge-first, cloud-optional** — the platform must run standalone without external services

## Testing
- All new code must include tests
- Tests are written using pytest
- Run tests with: `pytest tests/ -v`
- Aim for 90%+ test coverage
- Test files go in `tests/` with the naming convention `test_<module>.py`

## Project Structure

Key directories for contributors:

| Directory | What it contains |
|-----------|-----------------|
| `src/vpp/optimization/` | Stochastic, real-time, distributed optimizers, plugin system |
| `src/vpp/trading/` | Markets, orders, strategies, portfolio, risk management |
| `src/vpp/protocols/` | OpenADR, OCPP, MQTT, Modbus, IEEE 2030.5 adapters |
| `src/vpp/v2g/` | EV models, smart scheduling, fleet aggregation |
| `src/vpp/grid/` | Grid-forming inverters, microgrid controller |
| `src/vpp/research/` | ML/AI models (non-production) |
| `src/vpp/api/` | FastAPI REST + WebSocket endpoints |
| `benchmarks/` | Datasets, scenarios, metrics, benchmark runner |
| `demos/` | Interactive demo applications |
| `tests/` | Test suite (226 tests) |

## Priority Contribution Areas

- **New resource types**: fuel cells, pumped hydro, thermal storage
- **Protocol adapters**: SunSpec, DNP3, IEC 61850
- **Real grid data**: ISO API integrations, utility data feeds
- **Optimization algorithms**: new solver plugins
- **Documentation**: tutorials, case studies, and deployment guides

## Pull Request Process
1. Update the README.md with details of changes if needed
2. Update the documentation with any new features
3. The PR will be merged once you have the sign-off of two maintainers

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/vinerya/virtual-power-plant/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/vinerya/virtual-power-plant/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).
