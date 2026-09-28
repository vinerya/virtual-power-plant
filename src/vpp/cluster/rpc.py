"""Forward a call to the process that leads a lease, through the database.

Some state lives in exactly one process: the simulated trading venue's order
books, the alert evaluator's per-rule history. When a request for it lands
on another API worker, that worker writes the call to ``cluster_calls``
(status ``pending``) and waits; the leader's executor loop claims it
(``UPDATE ... SET status='running' WHERE id = :id AND status = 'pending'``,
so exactly one process runs it), runs the registered handler in its own DB
session and writes the JSON result (or the HTTP error) back.

Outcomes seen by the caller:

* ``done`` / ``failed`` -- the handler's result, or its ``HTTPException``
  re-raised with the same status and detail.
* Nobody claimed the call before the timeout (no leader, or the leader is
  stuck): the caller flips it to ``cancelled`` -- it will never run -- and
  gets :class:`LeaderUnavailableError` (HTTP 503, ``leader_unavailable``).
* The call was claimed but did not finish in time:
  :class:`CallOutcomeUnknownError` (HTTP 504, ``leader_timeout``, with the
  call id); it may still complete.

Fire-and-forget submissions (``wait=False``) are dropped (``expired``) if no
leader claims them before their deadline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, select, update

from vpp.cluster.node import node_id, utcnow
from vpp.db.models import ClusterCallModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any], "AsyncSession"], Awaitable[Any]]

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_POLL_S = 0.25
RETENTION = timedelta(hours=1)
FINAL_STATUSES = ("done", "failed", "cancelled", "expired")

_handlers: dict[tuple[str, str], Handler] = {}


def register_handler(target: str, method: str, handler: Handler) -> None:
    """Register *handler* for ``(target, method)`` (last registration wins)."""
    _handlers[(target, method)] = handler


def handles(target: str) -> bool:
    return any(t == target for t, _ in _handlers)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ClusterCallError(Exception):
    """A forwarded call failed; carries the HTTP status/detail to answer with."""

    def __init__(
        self, status_code: int, detail: Any, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def to_http(self) -> Exception:
        from fastapi import HTTPException

        return HTTPException(self.status_code, detail=self.detail, headers=self.headers)


class LeaderUnavailableError(ClusterCallError):
    def __init__(self, target: str, timeout_s: float) -> None:
        super().__init__(
            503,
            {
                "code": "leader_unavailable",
                "message": (
                    f"No API process holding the '{target}' lease answered within "
                    f"{timeout_s:g}s; the request was not executed. Retry shortly."
                ),
                "target": target,
            },
            {"Retry-After": "5"},
        )


class CallOutcomeUnknownError(ClusterCallError):
    def __init__(self, target: str, call_id: str) -> None:
        super().__init__(
            504,
            {
                "code": "leader_timeout",
                "message": (
                    f"The '{target}' leader accepted the request but did not finish in "
                    "time; it may still complete."
                ),
                "target": target,
                "call_id": call_id,
            },
        )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)  # numpy arrays
    if callable(tolist):
        return tolist()
    model_dump = getattr(value, "model_dump", None)  # pydantic models
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


# ---------------------------------------------------------------------------
# Caller side
# ---------------------------------------------------------------------------


def _factory(
    session_factory: async_sessionmaker[AsyncSession] | None,
) -> async_sessionmaker[AsyncSession]:
    if session_factory is not None:
        return session_factory
    from vpp.db.engine import get_session_factory

    return get_session_factory()


async def submit(
    target: str,
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> str:
    """Persist a pending call for the *target* leader. Returns the call id."""
    now = utcnow()
    call_id = str(uuid.uuid4())
    async with _factory(session_factory)() as session:
        session.add(
            ClusterCallModel(
                id=call_id,
                target=target,
                method=method,
                payload_json=dumps(payload or {}),
                status="pending",
                caller=node_id(),
                created_at=now,
                deadline_at=now + timedelta(seconds=timeout_s),
            )
        )
        await session.commit()
    return call_id


async def call(
    target: str,
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> Any:
    """Run ``method`` on the *target* leader and return its JSON result.

    Raises :class:`ClusterCallError` (see the module docstring).
    """
    factory = _factory(session_factory)
    call_id = await submit(target, method, payload, timeout_s=timeout_s, session_factory=factory)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    delay = min(0.02, poll_s)
    claimed = False
    while True:
        row = await _load(factory, call_id)
        if row is not None and row.status in ("done", "failed"):
            return _outcome(row)
        claimed = claimed or (row is not None and row.status == "running")
        now = loop.time()
        if now >= deadline:
            break
        await asyncio.sleep(min(delay, deadline - now))
        delay = min(delay * 2, poll_s)

    if not claimed and await _cancel_if_pending(factory, call_id):
        raise LeaderUnavailableError(target, timeout_s)
    # Claimed (running): give it a short grace period before giving up.
    grace_deadline = loop.time() + min(timeout_s, 5.0)
    while loop.time() < grace_deadline:
        row = await _load(factory, call_id)
        if row is not None and row.status in ("done", "failed"):
            return _outcome(row)
        await asyncio.sleep(poll_s)
    raise CallOutcomeUnknownError(target, call_id)


async def _load(
    factory: async_sessionmaker[AsyncSession], call_id: str
) -> ClusterCallModel | None:
    async with factory() as session:
        return await session.get(ClusterCallModel, call_id)


def _outcome(row: ClusterCallModel) -> Any:
    if row.status == "done":
        return json.loads(row.result_json) if row.result_json else None
    error = json.loads(row.error_json or "{}")
    raise ClusterCallError(
        int(error.get("status_code", 500)),
        error.get("detail", "forwarded call failed"),
        error.get("headers"),
    )


async def _cancel_if_pending(factory: async_sessionmaker[AsyncSession], call_id: str) -> bool:
    async with factory() as session:
        result = await session.execute(
            update(ClusterCallModel)
            .where(ClusterCallModel.id == call_id, ClusterCallModel.status == "pending")
            .values(status="cancelled", completed_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        return int(getattr(result, "rowcount", 0) or 0) == 1


# ---------------------------------------------------------------------------
# Executor side
# ---------------------------------------------------------------------------


async def execute_pending(
    session_factory: async_sessionmaker[AsyncSession],
    targets: Iterable[str],
    *,
    limit: int = 20,
) -> int:
    """Claim and run pending calls for *targets* (one pass). Returns how many ran."""
    targets = [t for t in targets if handles(t)]
    if not targets:
        return 0
    now = utcnow()
    async with session_factory() as session:
        # Calls nobody claimed in time: their callers have given up.
        await session.execute(
            update(ClusterCallModel)
            .where(
                ClusterCallModel.target.in_(targets),
                ClusterCallModel.status == "pending",
                ClusterCallModel.deadline_at < now,
            )
            .values(status="expired", completed_at=now)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        ids = list(
            (
                await session.execute(
                    select(ClusterCallModel.id)
                    .where(
                        ClusterCallModel.target.in_(targets),
                        ClusterCallModel.status == "pending",
                    )
                    .order_by(ClusterCallModel.created_at)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    ran = 0
    for call_id in ids:
        if await _run_one(session_factory, call_id):
            ran += 1
    return ran


async def _run_one(session_factory: async_sessionmaker[AsyncSession], call_id: str) -> bool:
    async with session_factory() as session:
        claimed = await session.execute(
            update(ClusterCallModel)
            .where(
                and_(
                    ClusterCallModel.id == call_id,
                    ClusterCallModel.status == "pending",
                    ClusterCallModel.deadline_at >= utcnow(),
                )
            )
            .values(status="running", executor=node_id())
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        if int(getattr(claimed, "rowcount", 0) or 0) != 1:
            return False
        row = await session.get(ClusterCallModel, call_id)
    if row is None:
        return False

    handler = _handlers.get((row.target, row.method))
    status, result_json, error_json = "done", None, None
    if handler is None:
        status = "failed"
        error_json = dumps(
            {"status_code": 501, "detail": f"no handler for {row.target}.{row.method}"}
        )
    else:
        try:
            payload = json.loads(row.payload_json or "{}")
            async with session_factory() as work_session:
                try:
                    result = await handler(payload, work_session)
                    await work_session.commit()
                except Exception:
                    await work_session.rollback()
                    raise
            result_json = dumps(result)
        except ClusterCallError as exc:
            status = "failed"
            error_json = dumps(
                {"status_code": exc.status_code, "detail": exc.detail, "headers": exc.headers}
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and hasattr(exc, "detail"):  # HTTPException
                status = "failed"
                error_json = dumps(
                    {
                        "status_code": status_code,
                        "detail": exc.detail,
                        "headers": getattr(exc, "headers", None),
                    }
                )
            else:
                logger.exception("Forwarded call %s.%s failed", row.target, row.method)
                status = "failed"
                error_json = dumps({"status_code": 500, "detail": "Internal Server Error"})

    async with session_factory() as session:
        await session.execute(
            update(ClusterCallModel)
            .where(ClusterCallModel.id == call_id)
            .values(
                status=status,
                result_json=result_json,
                error_json=error_json,
                completed_at=utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    return True


async def purge_finished(
    session_factory: async_sessionmaker[AsyncSession], *, older_than: timedelta = RETENTION
) -> int:
    cutoff = utcnow() - older_than
    async with session_factory() as session:
        result = await session.execute(
            delete(ClusterCallModel)
            .where(
                ClusterCallModel.status.in_(FINAL_STATUSES),
                ClusterCallModel.created_at < cutoff,
            )
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        return int(getattr(result, "rowcount", 0) or 0)


async def run_executor(
    session_factory: async_sessionmaker[AsyncSession],
    served_targets: Callable[[], Iterable[str]],
    *,
    poll_s: float = DEFAULT_POLL_S,
    purge_every_s: float = 300.0,
) -> None:
    """Forever: run pending calls for the targets this process currently leads."""
    loop = asyncio.get_running_loop()
    next_purge = loop.time() + purge_every_s
    while True:
        try:
            targets = [t for t in served_targets() if handles(t)]
            ran = await execute_pending(session_factory, targets) if targets else 0
            if targets and loop.time() >= next_purge:
                next_purge = loop.time() + purge_every_s
                await purge_finished(session_factory)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cluster call executor pass failed")
            ran = 0
        if not ran:
            await asyncio.sleep(poll_s)


async def on_leader(
    target: str,
    method: str,
    payload: dict[str, Any],
    local: Callable[[], Awaitable[Any]],
    *,
    timeout_s: float | None = None,
) -> Any:
    """Run *local* here when this process leads *target*, else forward ``method``.

    For API routes: a forwarding failure is raised as ``HTTPException``.
    *timeout_s* caps ``VPP_CLUSTER_CALL_TIMEOUT_SECONDS`` for this call (reads
    that should fail fast when the leader is gone).
    """
    from vpp.cluster.lease import is_local

    if is_local(target):
        return await local()
    from vpp.settings import get_settings

    settings = get_settings()
    wait_s = float(settings.cluster_call_timeout_seconds)
    if timeout_s is not None:
        wait_s = min(wait_s, timeout_s)
    try:
        return await call(
            target,
            method,
            payload,
            timeout_s=wait_s,
            poll_s=settings.cluster_poll_interval_seconds,
        )
    except ClusterCallError as exc:
        raise exc.to_http() from exc
