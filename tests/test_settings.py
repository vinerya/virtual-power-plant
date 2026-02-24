"""Tests for application settings."""

import os

import pytest

from vpp.settings import Settings


def test_defaults():
    s = Settings(env="development", secret_key="test")
    assert s.api_port == 8000
    assert s.log_level == "INFO"
    assert s.is_development


def test_production_flag():
    s = Settings(env="production", secret_key="x")
    assert s.is_production
    assert not s.is_development


def test_invalid_log_level():
    with pytest.raises(Exception):
        Settings(log_level="VERBOSE", secret_key="x")


def test_database_is_sqlite():
    s = Settings(database_url="sqlite+aiosqlite:///./test.db", secret_key="x")
    assert s.database_is_sqlite


def test_database_is_postgres():
    s = Settings(database_url="postgresql+asyncpg://u:p@localhost/db", secret_key="x")
    assert not s.database_is_sqlite
