"""Ownership-scoped lookups shared by the sites, customer and metrics routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from vpp.db.models import ResourceModel, SiteModel, UserModel
from vpp.schemas.auth import UserRole

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def is_customer(user: UserModel) -> bool:
    return user.role == UserRole.CUSTOMER.value


async def owned_sites(session: AsyncSession, user_id: str) -> list[SiteModel]:
    """Sites owned by ``user_id``, oldest first (the first is the primary premise)."""
    result = await session.execute(
        select(SiteModel)
        .where(SiteModel.owner_id == user_id)
        .order_by(SiteModel.created_at, SiteModel.id)
    )
    return list(result.scalars().all())


async def owned_resources(session: AsyncSession, user_id: str) -> list[ResourceModel]:
    """Resources belonging to any site owned by ``user_id``."""
    result = await session.execute(
        select(ResourceModel)
        .join(SiteModel, ResourceModel.site_id == SiteModel.id)
        .where(SiteModel.owner_id == user_id)
        .order_by(ResourceModel.name)
    )
    return list(result.scalars().all())


async def visible_site(
    session: AsyncSession, user: UserModel, site_id: str
) -> SiteModel | None:
    """Return the site if ``user`` may see it, else ``None``."""
    site = await session.get(SiteModel, site_id)
    if site is None:
        return None
    if is_customer(user) and site.owner_id != user.id:
        return None
    return site


async def visible_resource(
    session: AsyncSession, user: UserModel, resource_id: str
) -> ResourceModel | None:
    """Return the resource if ``user`` may see it, else ``None``."""
    resource = await session.get(ResourceModel, resource_id)
    if resource is None:
        return None
    if not is_customer(user):
        return resource
    if resource.site_id is None:
        return None
    site = await session.get(SiteModel, resource.site_id)
    if site is None or site.owner_id != user.id:
        return None
    return resource
