# ============================================================================
# Virtual Power Plant Platform — Multi-stage Docker build
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project metadata first (cache-friendly layer)
COPY pyproject.toml setup.py ./
COPY src/ src/

# Build wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# Install runtime dependencies
RUN pip wheel --wheel-dir /wheels \
    "virtual-power-plant[api,db,cli,monitoring]"

# ============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Moudather Chelbi <moudather.chelbi@gmail.com>"
LABEL description="Virtual Power Plant Platform — production-ready VPP management"

WORKDIR /app

# Install wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy config examples
COPY configs/ configs/
COPY .env.example .env.example

# Non-root user for security
RUN adduser --disabled-password --gecos "" vpp
USER vpp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

CMD ["uvicorn", "vpp.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
