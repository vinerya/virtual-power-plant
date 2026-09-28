"""Structured logging configuration using ``structlog``.

Provides JSON output in production and coloured human-readable output
during development.  Automatically attaches ``request_id`` when available.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_VPP_HANDLER_ATTR = "_vpp_structlog_handler"


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the VPP platform.

    Parameters
    ----------
    level:
        Root log level (DEBUG, INFO, WARNING, ERROR).
    json_output:
        ``True`` for JSON (production), ``False`` for coloured dev output.
    log_file:
        Optional path to a log file (in addition to stderr).
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ``foreign_pre_chain`` runs the shared processors (notably
    # ``merge_contextvars``) on records emitted through plain stdlib
    # ``logging`` too, so ``request_id`` bound by
    # :class:`vpp.api.middleware.RequestIdMiddleware` shows up on every line,
    # not only on lines logged through structlog.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Root handler (stderr).  Only handlers previously installed by this
    # function are replaced, so calling it repeatedly (e.g. once per
    # ``create_app()``) is idempotent and handlers installed by others --
    # pytest's log capture, an embedding application -- are left alone.
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        if getattr(handler, _VPP_HANDLER_ATTR, False):
            root.removeHandler(handler)
            handler.close()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _VPP_HANDLER_ATTR, True)
    root.addHandler(stream_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _VPP_HANDLER_ATTR, True)
        root.addHandler(file_handler)

    # Quiet noisy libraries
    for name in ("uvicorn.access", "sqlalchemy.engine", "httpcore", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structured logger instance."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
