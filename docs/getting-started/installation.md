# Installation

Two ways to run the platform. Docker Compose gives you the API, PostgreSQL
and the web console in one command. The Python package is what you want
for development, for the tutorial, or to use the optimization and tariff
code as a library.

## Docker Compose

Requires Docker with the Compose plugin.

```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
export VPP_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
export POSTGRES_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
docker compose up -d --build
```

Keep both values: the database password is set on the volume at first
start, and changing the secret key signs everybody out. The API container
runs the migrations before it starts. Continue with
[creating the first admin](first-run.md#with-docker-compose).

Optional services, all in the same compose project:

| Command | Adds |
|---|---|
| `docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d` | Prometheus (:9090) and Grafana (:3001) |
| `docker compose --profile sim up -d` | the SunSpec inverter simulator as `sunspec-sim:5020` |
| `VPP_TRACING_ENABLED=true docker compose --profile tracing up -d` | Jaeger (:16686) with OpenTelemetry tracing |

The [deployment guide](../deployment.md) covers every compose variable,
TLS, and running behind a reverse proxy.

## Python package

Python 3.10, 3.11 or 3.12. The base install is the library: optimization
(Pyomo + HiGHS), tariffs and the in-process resource models. The server,
database, CLI and protocols come as extras:

| Extra | For |
|---|---|
| `api` | the FastAPI server (`vpp serve`) |
| `db` | SQLAlchemy, alembic and the SQLite/PostgreSQL async drivers |
| `cli` | the `vpp` command |
| `protocols` | MQTT, Modbus (and `vpp simulate sunspec`), OpenADR / IEEE 2030.5 clients |
| `migrations` | the sync PostgreSQL driver alembic uses for `vpp migrate` against PostgreSQL |
| `monitoring` | Prometheus `/metrics` |
| `tracing` | OpenTelemetry export |
| `degradation` | rainflow cycle counting for battery state of health |
| `research` | pandas, scipy, scikit-learn, cvxpy for the research modules |
| `dev` | tests and linters |
| `docs` | this documentation site |

From a clone (recommended until a release is published on PyPI):

```bash
git clone https://github.com/vinerya/virtual-power-plant.git
cd virtual-power-plant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[api,db,cli,protocols,monitoring]"
vpp --version
```

From PyPI, once a release is published there:

```bash
pip install "virtual-power-plant[api,db,cli,protocols,monitoring]"
```

Next: [First run](first-run.md).
