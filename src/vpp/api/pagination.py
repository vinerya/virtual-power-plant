"""Offset pagination shared by the list endpoints.

List endpoints keep returning a bare JSON array (the web console and
existing clients depend on that shape) and report the total number of
matching rows -- before ``limit`` / ``offset`` are applied -- in the
``X-Total-Count`` response header. ``limit`` has a per-endpoint default and
a hard maximum (a larger value is a ``422``); ``offset`` defaults to 0.
Endpoints that already accepted ``skip`` keep it as an alias of ``offset``
(``offset`` wins when both are sent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import Query
from sqlalchemy import func, select

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Response
    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession

TOTAL_COUNT_HEADER = "X-Total-Count"


@dataclass(frozen=True)
class Page:
    """Resolved ``limit`` / ``offset`` of a list request."""

    limit: int
    offset: int


def page_params(
    *, default_limit: int = 50, max_limit: int = 200, legacy_skip: bool = False
) -> Callable[..., Page]:
    """FastAPI dependency factory returning a :class:`Page`.

    ``legacy_skip`` also accepts ``skip`` (older name of ``offset``).
    """
    limit_q: Any = Query(
        default_limit, ge=1, le=max_limit, description=f"Page size (max {max_limit})"
    )
    offset_q: Any = Query(None, ge=0, description="Rows to skip (default 0)")

    if legacy_skip:
        skip_q: Any = Query(0, ge=0, description="Alias of offset (deprecated)")

        def _dep_skip(
            limit: int = limit_q, offset: int | None = offset_q, skip: int = skip_q
        ) -> Page:
            return Page(limit=limit, offset=offset if offset is not None else skip)

        return _dep_skip

    def _dep(limit: int = limit_q, offset: int | None = offset_q) -> Page:
        return Page(limit=limit, offset=offset or 0)

    return _dep


async def count_rows(session: AsyncSession, stmt: Select) -> int:
    """``COUNT(*)`` of ``stmt`` with its ordering and paging removed."""
    sub = stmt.order_by(None).limit(None).offset(None).subquery()
    return int((await session.execute(select(func.count()).select_from(sub))).scalar_one())


def set_total(response: Response, total: int) -> None:
    """Report ``total`` matching rows in the ``X-Total-Count`` header."""
    response.headers[TOTAL_COUNT_HEADER] = str(total)


async def paginate(
    session: AsyncSession, response: Response, stmt: Select, page: Page
) -> list[Any]:
    """Run ``stmt`` for one page, set ``X-Total-Count`` and return the rows.

    ``stmt`` must be ordered deterministically. Returns ``Row`` objects when
    it selects several entities, scalars otherwise.
    """
    set_total(response, await count_rows(session, stmt))
    result = await session.execute(stmt.offset(page.offset).limit(page.limit))
    if len(stmt.column_descriptions) == 1:
        return list(result.scalars().all())
    return list(result.all())
