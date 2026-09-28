"""Single source of truth for the package version.

The version is declared once, in ``pyproject.toml``.  At runtime it is read
from the installed distribution's metadata; when the package is imported
from a source checkout that was never installed (e.g. ``PYTHONPATH=src``),
it falls back to parsing ``pyproject.toml`` directly.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

DISTRIBUTION_NAME = "virtual-power-plant"
UNKNOWN_VERSION = "0.0.0+unknown"

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _project_table_fallback(text: str) -> dict[str, Any]:
    """Read ``name``/``version`` from ``[project]`` without a TOML library.

    Only used on Python 3.10 when ``tomli`` isn't installed; both keys are
    plain strings in this repository's pyproject.
    """
    table = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if table is None:
        return {}
    return dict(re.findall(r'^(name|version)\s*=\s*"([^"]*)"', table.group(1), re.M))


def _version_from_pyproject(path: Path | None = None) -> str | None:
    try:
        raw = (path or _PYPROJECT).read_bytes()
    except OSError:
        return None
    if tomllib is not None:
        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            return None
        project = data.get("project", {})
    else:
        project = _project_table_fallback(raw.decode("utf-8", errors="replace"))
    if project.get("name") != DISTRIBUTION_NAME:
        return None
    value = project.get("version")
    return value if isinstance(value, str) else None


def get_version() -> str:
    """Return the installed package version, or the pyproject value as fallback."""
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _version_from_pyproject() or UNKNOWN_VERSION


__version__ = get_version()
