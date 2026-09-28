"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from vpp.settings import Settings


def test_defaults():
    s = Settings(env="development", secret_key="test")
    assert s.api_port == 8000
    assert s.log_level == "INFO"
    assert s.is_development


def test_production_flag():
    s = Settings(env="production", secret_key="x" * 32)
    assert s.is_production
    assert not s.is_development


def test_invalid_log_level():
    with pytest.raises(ValidationError, match="log_level must be one of"):
        Settings(log_level="VERBOSE", secret_key="x")


def test_database_is_sqlite():
    s = Settings(database_url="sqlite+aiosqlite:///./test.db", secret_key="x")
    assert s.database_is_sqlite


def test_database_is_postgres():
    s = Settings(database_url="postgresql+asyncpg://u:p@localhost/db", secret_key="x")
    assert not s.database_is_sqlite


def test_removed_redis_url_is_ignored(monkeypatch):
    """VPP_REDIS_URL was never consumed and has been removed; deployments that
    still set it must keep starting (extra settings are ignored)."""
    monkeypatch.setenv("VPP_REDIS_URL", "redis://localhost:6379/0")
    s = Settings(secret_key="x")
    assert not hasattr(s, "redis_url")


# Settings that are intentionally not read from src/vpp. Keep this empty
# unless there is a documented reason; an unread VPP_* variable misleads
# operators into thinking it does something.
# Settings intentionally not referenced by name in src/ (name -> reason).
_UNREFERENCED_SETTINGS_ALLOWLIST: dict[str, str] = {}


def test_every_setting_is_read_somewhere():
    """Guard against dead settings: each field must be referenced in src/vpp.

    A reference is the field name as a word in any module other than
    settings.py, or ``self.<field>`` inside settings.py (validators and
    properties such as ``config_file_path``).
    """
    import re
    from pathlib import Path

    import vpp

    pkg = Path(vpp.__file__).parent
    settings_src = (pkg / "settings.py").read_text(encoding="utf-8")
    other_src = "\n".join(
        p.read_text(encoding="utf-8") for p in pkg.rglob("*.py") if p.name != "settings.py"
    )
    unread = [
        name
        for name in Settings.model_fields
        if name not in _UNREFERENCED_SETTINGS_ALLOWLIST
        and not re.search(rf"\b{name}\b", other_src)
        and not re.search(rf"\bself\.{name}\b", settings_src)
    ]
    assert not unread, f"Settings never read in src/vpp (remove them or use them): {unread}"
