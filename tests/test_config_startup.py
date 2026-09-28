"""The newest stored config document is re-applied when the API starts."""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest
import yaml
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vpp.api import app as app_module
from vpp.api.deps import get_live_config, reset_live_config
from vpp.api.routes.config import apply_stored_config
from vpp.config import VPPConfig
from vpp.db import engine as db_engine
from vpp.db.base import Base
from vpp.db.models import ConfigDocumentModel
from vpp.settings import Settings


def _doc(**top) -> str:
    data = VPPConfig().to_dict()
    data.update(top)
    return yaml.safe_dump(data, sort_keys=False)


@pytest.fixture(autouse=True)
def _fresh_config():
    reset_live_config()
    yield
    reset_live_config()


@contextlib.asynccontextmanager
async def _db(create_tables: bool = True):
    path = f"test_config_startup_{uuid.uuid4().hex[:8]}.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///./{path}")
    try:
        if create_tables:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        yield path, async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
        for suffix in ("", "-wal", "-shm", "-journal"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(path + suffix)


async def _store(factory, *docs: str) -> None:
    async with factory() as session:
        for i, text in enumerate(docs, start=1):
            session.add(ConfigDocumentModel(version=i, yaml=text, hash=f"h{i}"))
        await session.commit()


@pytest.mark.asyncio
async def test_missing_table_keeps_defaults():
    async with _db(create_tables=False) as (_path, factory):
        assert await apply_stored_config(factory) is None
    assert get_live_config().name == VPPConfig().name


@pytest.mark.asyncio
async def test_empty_table_keeps_defaults():
    async with _db() as (_path, factory):
        assert await apply_stored_config(factory) is None
    assert get_live_config().name == VPPConfig().name


@pytest.mark.asyncio
async def test_newest_document_applied():
    async with _db() as (_path, factory):
        await _store(factory, _doc(name="Old VPP"), _doc(name="Boot VPP"))
        assert await apply_stored_config(factory) == 2
    assert get_live_config().name == "Boot VPP"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["name: [unclosed", "bogus_key: 1\n"])
async def test_invalid_stored_document_is_skipped(bad):
    async with _db() as (_path, factory):
        await _store(factory, bad)
        assert await apply_stored_config(factory) is None
    assert get_live_config().name == VPPConfig().name


@pytest.fixture
def preserve_db_globals():
    saved = db_engine._engine, db_engine._session_factory
    yield
    db_engine._engine, db_engine._session_factory = saved


@pytest.mark.asyncio
async def test_lifespan_applies_stored_config(monkeypatch, preserve_db_globals):
    async with _db() as (path, factory):
        await _store(factory, _doc(name="Lifespan VPP"))
        settings = Settings(database_url=f"sqlite+aiosqlite:///./{path}")
        monkeypatch.setattr(app_module, "get_settings", lambda: settings)
        fastapi_app = app_module.create_app()
        async with fastapi_app.router.lifespan_context(fastapi_app):
            assert get_live_config().name == "Lifespan VPP"
