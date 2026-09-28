# ============================================================================
# Virtual Power Plant API (FastAPI) - multi-stage build
#
#   docker build -t vpp-api .
#   docker run -p 8000:8000 -e VPP_SECRET_KEY=... vpp-api
#
# The package is installed from its wheel; the alembic migrations ship inside
# it (vpp/migrations), so `vpp migrate` works from any directory. Run
# migrations with `vpp migrate` (docker-compose.yml does this before
# starting uvicorn).
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Project metadata first (cache-friendly layer), then the sources.
COPY pyproject.toml README.md ./
COPY src/ src/

# Wheels for the package and every runtime extra the API uses:
#   api         FastAPI, uvicorn, JWT, httpx
#   db          SQLAlchemy async, alembic, aiosqlite, asyncpg
#   protocols   MQTT, Modbus, lxml, httpx (OpenADR / IEEE 2030.5 clients)
#   solver      Pyomo + HiGHS (dispatch / MPC); falls back to rules without it
#   degradation rainflow cycle counting for battery SOH
#   monitoring  prometheus-client (/metrics)
#   cli         click + rich (`vpp migrate`, `vpp serve`, ...)
# psycopg2-binary: alembic migrations run on a *sync* driver
# (postgresql+asyncpg URLs are rewritten to postgresql+psycopg2).
RUN pip wheel --wheel-dir /wheels \
    ".[api,db,protocols,solver,degradation,monitoring,cli]" \
    "psycopg2-binary>=2.9"

# ============================================================================
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="vpp-api" \
      org.opencontainers.image.description="Open-source Virtual Power Plant platform - API" \
      org.opencontainers.image.source="https://github.com/vinerya/virtual-power-plant" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies from the builder stage.
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels

# Sample configuration files (library examples).
COPY configs/ configs/

# Non-root user. /app/data is a writable place for a SQLite database
# (e.g. VPP_DATABASE_URL=sqlite+aiosqlite:///./data/vpp.db); /app itself is
# writable too so the default ./vpp.db works for a quick `docker run`.
RUN adduser --disabled-password --gecos "" vpp \
    && mkdir -p /app/data \
    && chown vpp:vpp /app /app/data
USER vpp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Behind a reverse proxy, set FORWARDED_ALLOW_IPS to the proxy's address so
# uvicorn trusts its X-Forwarded-For / X-Forwarded-Proto headers.
CMD ["uvicorn", "vpp.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
