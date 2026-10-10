"""Optional OpenTelemetry tracing.

Tracing is off unless ``VPP_TRACING_ENABLED=true`` *and* the ``tracing``
extra (``pip install "virtual-power-plant[tracing]"``) is installed. This
module never imports OpenTelemetry at import time, so the core package works
without it; with tracing requested but the extra missing, :func:`setup_tracing`
logs a warning and the platform keeps running untraced.

Call sites use :func:`span`, which returns a shared no-op context manager
while tracing is off (one global lookup, no allocation)::

    from vpp import tracing

    with tracing.span("vpp.actuator.setpoint", resource_id=rid) as sp:
        ...
        sp.set(status="accepted", verified=True)

Keyword attributes are recorded as ``vpp.<name>``; ``None`` values are
dropped and values that are not ``str``/``bool``/``int``/``float`` (or a list
of them) are stringified.

Exporter, sampler and service name come from the standard OpenTelemetry
environment variables -- ``OTEL_EXPORTER_OTLP_ENDPOINT``,
``OTEL_EXPORTER_OTLP_PROTOCOL`` (``http/protobuf`` default, or ``grpc``),
``OTEL_TRACES_EXPORTER`` (``otlp`` default, ``console`` or ``none``),
``OTEL_TRACES_SAMPLER`` / ``OTEL_TRACES_SAMPLER_ARG``, ``OTEL_SERVICE_NAME``
(default ``vpp``) and ``OTEL_RESOURCE_ATTRIBUTES``.

Trace context crosses process boundaries as a W3C ``traceparent`` carrier:
:func:`inject` returns it (``{}`` when off) and :func:`continue_trace`
re-attaches it on the receiving side (used for forwarded cluster calls, see
:mod:`vpp.cluster.rpc`).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

logger = logging.getLogger(__name__)

#: OpenTelemetry ``Tracer`` while tracing is on, else ``None``.
_tracer: Any = None
#: The ``TracerProvider`` spans are recorded with (needed by instrumentors).
_provider: Any = None
_instrumented_httpx = False

DEFAULT_SERVICE_NAME = "vpp"
_TRACER_NAME = "vpp"
_PRIMITIVES = (str, bool, int, float)
_KINDS = ("internal", "server", "client", "producer", "consumer")


def is_enabled() -> bool:
    """``True`` while spans are being recorded."""
    return _tracer is not None


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


class _NoopSpan:
    """What :func:`span` returns while tracing is off: does nothing, cheaply."""

    __slots__ = ()

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def set(self, **attrs: Any) -> None:
        """Add attributes to the span (no-op)."""

    def error(self, description: str = "") -> None:
        """Mark the span as failed (no-op)."""

    @property
    def recording(self) -> bool:
        return False


_NOOP = _NoopSpan()


def _value(v: Any) -> Any:
    if isinstance(v, _PRIMITIVES):
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, _PRIMITIVES) for x in v):
        return list(v)
    return str(v)


def _attributes(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {f"vpp.{k}": _value(v) for k, v in attrs.items() if v is not None}


class _Span:
    """A live span; :meth:`set` / :meth:`error` annotate it."""

    __slots__ = ("_attrs", "_cm", "_kind", "_name", "_span")

    def __init__(self, name: str, kind: str, attrs: dict[str, Any]) -> None:
        self._name = name
        self._kind = kind
        self._attrs = attrs
        self._cm: Any = None
        self._span: Any = None

    def __enter__(self) -> _Span:
        from opentelemetry.trace import SpanKind

        self._cm = _tracer.start_as_current_span(
            self._name,
            kind=getattr(SpanKind, self._kind.upper(), SpanKind.INTERNAL),
            attributes=_attributes(self._attrs),
        )
        self._span = self._cm.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._cm.__exit__(exc_type, exc, tb)

    def set(self, **attrs: Any) -> None:
        if self._span is not None:
            self._span.set_attributes(_attributes(attrs))

    def error(self, description: str = "") -> None:
        if self._span is not None:
            from opentelemetry.trace import Status, StatusCode

            self._span.set_status(Status(StatusCode.ERROR, description or None))

    @property
    def recording(self) -> bool:
        return bool(self._span is not None and self._span.is_recording())


def span(name: str, *, kind: str = "internal", **attrs: Any) -> _NoopSpan | _Span:
    """Context manager for one span named *name* (a no-op while tracing is off).

    *kind* is ``internal`` (default), ``server``, ``client``, ``producer`` or
    ``consumer``. An exception escaping the block is recorded on the span and
    marks it failed.
    """
    if _tracer is None:
        return _NOOP
    return _Span(name, kind if kind in _KINDS else "internal", attrs)


# ---------------------------------------------------------------------------
# Context propagation (W3C traceparent)
# ---------------------------------------------------------------------------


def inject() -> dict[str, str]:
    """Return the current trace context as a W3C carrier (``{}`` when off)."""
    if _tracer is None:
        return {}
    from opentelemetry import propagate

    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextlib.contextmanager
def continue_trace(carrier: Mapping[str, str] | None) -> Iterator[None]:
    """Make spans opened inside the block children of the span in *carrier*."""
    if _tracer is None or not carrier:
        yield
        return
    from opentelemetry import context, propagate

    token = context.attach(propagate.extract(dict(carrier)))
    try:
        yield
    finally:
        context.detach(token)


def current_ids() -> tuple[str, str] | None:
    """``(trace_id, span_id)`` of the current span as hex, or ``None``."""
    if _tracer is None:
        return None
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def add_trace_ids(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor: add ``trace_id`` / ``span_id`` inside a span."""
    if _tracer is not None and "trace_id" not in event_dict:
        ids = current_ids()
        if ids is not None:
            event_dict["trace_id"], event_dict["span_id"] = ids
    return event_dict


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def service_name() -> str:
    return os.environ.get("OTEL_SERVICE_NAME") or DEFAULT_SERVICE_NAME


def _exporter() -> Any:
    """Span exporter chosen by ``OTEL_TRACES_EXPORTER`` / ``OTEL_EXPORTER_OTLP_PROTOCOL``."""
    kind = os.environ.get("OTEL_TRACES_EXPORTER", "otlp").split(",")[0].strip().lower()
    if kind == "none":
        return None
    if kind == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()
    protocol = (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "http/protobuf"
    ).lower()
    if protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcExporter,
            )
        except ImportError:
            # The tracing extra ships only the HTTP exporter (grpcio is large).
            logger.warning(
                "OTEL_EXPORTER_OTLP_PROTOCOL=grpc needs "
                "opentelemetry-exporter-otlp-proto-grpc; exporting over http/protobuf"
            )
        else:
            return GrpcExporter()
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    # Endpoint, headers, timeout and TLS come from OTEL_EXPORTER_OTLP_* env vars.
    return OTLPSpanExporter()


def enable(provider: Any) -> None:
    """Record spans with *provider* (an SDK ``TracerProvider``).

    :func:`setup_tracing` calls this; tests call it with a provider wired to
    an in-memory exporter. The global OpenTelemetry provider is not touched.
    """
    global _tracer, _provider
    _provider = provider
    _tracer = provider.get_tracer(_TRACER_NAME)


def disable() -> None:
    """Stop recording spans (does not shut the provider down)."""
    global _tracer, _provider, _instrumented_httpx
    _tracer = None
    _provider = None
    if _instrumented_httpx:
        with contextlib.suppress(Exception):
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().uninstrument()
        _instrumented_httpx = False
    with contextlib.suppress(Exception):
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        instrumentor = SQLAlchemyInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def setup_tracing(settings: Any) -> bool:
    """Configure tracing from *settings* + the ``OTEL_*`` environment.

    Idempotent. Returns whether tracing is on. With
    ``settings.tracing_enabled`` set but the ``tracing`` extra missing, logs
    a warning and returns ``False``.
    """
    if not getattr(settings, "tracing_enabled", False):
        return False
    if _tracer is not None:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning(
            "VPP_TRACING_ENABLED is set but OpenTelemetry is not installed (%s); "
            "continuing without tracing. Install it with: "
            'pip install "virtual-power-plant[tracing]"',
            exc,
        )
        return False

    from vpp import __version__

    resource = Resource.create({"service.name": service_name(), "service.version": __version__})
    # No sampler argument: the SDK reads OTEL_TRACES_SAMPLER(_ARG) itself
    # (default parentbased_always_on).
    provider = TracerProvider(resource=resource)
    try:
        exporter = _exporter()
    except ImportError as exc:
        logger.warning("OTLP exporter unavailable (%s); spans are not exported", exc)
        exporter = None
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    # Global as well, so third-party instrumentation finds it. OpenTelemetry
    # refuses to replace an already-set global provider (logs a warning).
    trace.set_tracer_provider(provider)
    enable(provider)
    instrument_httpx()
    logger.info(
        "OpenTelemetry tracing enabled (service=%s, exporter=%s)",
        service_name(),
        type(exporter).__name__ if exporter is not None else "none",
    )
    return True


def shutdown_tracing() -> None:
    """Flush pending spans (the provider shuts down at interpreter exit)."""
    if _provider is not None:
        with contextlib.suppress(Exception):
            _provider.force_flush(5000)


# ---------------------------------------------------------------------------
# Library instrumentation
# ---------------------------------------------------------------------------


def instrument_fastapi(app: Any) -> bool:
    """Trace every HTTP request / WebSocket of *app* (no-op when off)."""
    if _tracer is None:
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi missing; requests not traced")
        return False
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=_provider,
        # Scrapes and probes would drown out the interesting traces.
        excluded_urls="/metrics,/health,/healthz,/ready",
    )
    return True


def instrument_sqlalchemy(engine: Any) -> bool:
    """Trace the statements of *engine* (an ``AsyncEngine`` or ``Engine``)."""
    if _tracer is None:
        return False
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    except ImportError:
        logger.warning("opentelemetry-instrumentation-sqlalchemy missing; SQL not traced")
        return False
    instrumentor = SQLAlchemyInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        # The instrumentor is a singleton that instruments once; re-arm it for
        # the new engine (init_db replaces the previous, disposed engine).
        instrumentor.uninstrument()
    instrumentor.instrument(
        engine=getattr(engine, "sync_engine", engine),
        tracer_provider=_provider,
        enable_commenter=False,
        # The instrumentation declares sqlalchemy<2.1 but only uses the stable
        # engine event API; without this it silently does nothing on 2.1.
        skip_dep_check=True,
    )
    return True


def instrument_httpx() -> bool:
    """Trace outbound httpx calls (OpenADR, IEEE 2030.5, tariff feeds, webhooks)."""
    global _instrumented_httpx
    if _tracer is None or _instrumented_httpx:
        return _instrumented_httpx
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        return False
    HTTPXClientInstrumentor().instrument(tracer_provider=_provider)
    _instrumented_httpx = True
    return True
