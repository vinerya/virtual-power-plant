"""Creating admin accounts outside the API: CLI and first-boot bootstrap.

* ``vpp users create-admin`` (see :mod:`vpp.cli.main`) calls
  :func:`create_admin`.
* On startup the API calls :func:`bootstrap_admin_from_settings`: when
  ``VPP_BOOTSTRAP_ADMIN_USERNAME`` and ``VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE``
  are set **and the users table is empty**, it creates that admin. The
  password is read from a file (e.g. a Docker/Kubernetes secret) so it never
  sits in the environment, and it is never logged. Once any user exists the
  settings are ignored (the file is not even read), so it is safe to leave
  them set or to delete the secret afterwards.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from vpp.auth.passwords import validate_password
from vpp.auth.security import get_password_hash
from vpp.db.models import UserModel
from vpp.db.repositories import UserRepository
from vpp.schemas.auth import UserRole
from vpp.settings import Settings

logger = logging.getLogger(__name__)

#: Same rule as ``UserCreate.username``.
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")


class BootstrapError(RuntimeError):
    """Invalid admin bootstrap input (bad username, weak password, ...)."""


def validate_username(username: str) -> None:
    if not _USERNAME_RE.fullmatch(username):
        raise BootstrapError("Username must be 3-64 characters of letters, digits, '_' or '-'")


def read_password_file(path: str | Path) -> str:
    """Read a password from ``path``, dropping one trailing newline."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapError(f"Cannot read password file {path}: {exc.strerror}") from exc
    return text.removesuffix("\n").removesuffix("\r")


async def count_users(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count(UserModel.id))) or 0)


async def create_admin(session: AsyncSession, username: str, password: str) -> UserModel:
    """Create an active admin after validating username and password.

    Raises :class:`BootstrapError` (with a message safe to print: it never
    contains the password) on invalid input or an existing username.
    """
    validate_username(username)
    try:
        validate_password(password, username=username)
    except ValueError as exc:
        raise BootstrapError(str(exc)) from exc
    if await UserRepository.get_by_username(session, username) is not None:
        raise BootstrapError(f"User {username!r} already exists")
    return await UserRepository.create_user(
        session,
        username=username,
        hashed_password=get_password_hash(password),
        role=UserRole.ADMIN.value,
    )


async def bootstrap_admin_from_settings(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> bool:
    """Create the first admin from settings if no user exists. Returns True if created.

    Raises :class:`BootstrapError` when the bootstrap is configured but
    unusable (missing file, weak password), so a broken deployment fails
    loudly instead of starting without any way to log in.
    """
    username = settings.bootstrap_admin_username
    if not username:
        return False
    if not settings.bootstrap_admin_password_file:
        raise BootstrapError(
            "VPP_BOOTSTRAP_ADMIN_USERNAME is set but VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE is not"
        )
    async with session_factory() as session:
        if await count_users(session) > 0:
            logger.debug("Admin bootstrap skipped: users already exist")
            return False
        password = read_password_file(settings.bootstrap_admin_password_file)
        await create_admin(session, username, password)
        try:
            await session.commit()
        except IntegrityError:
            # Another worker/replica bootstrapped concurrently.
            await session.rollback()
            logger.info("Admin bootstrap: %r was created by another process", username)
            return False
    logger.warning("Bootstrapped initial admin account %r (users table was empty)", username)
    return True
