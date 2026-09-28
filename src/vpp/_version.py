"""Single source of truth for the package version.

The version is declared once, in ``pyproject.toml``.  At runtime it is read
from the installed distribution's metadata; when the package is imported
from a source checkout that was never installed (e.g. ``PYTHONPATH=src``),
it falls back to parsing ``pyproject.toml`` directly.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None  # type: ignore[assignment]

DISTRIBUTION_NAME = "virtual-power-plant"
UNKNOWN_VERSION = "0.0.0+unknown"

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _version_from_pyproject(path: Path | None = None) -> str | None:
    if tomllib is None:
        return None
    try:
        with (path or _PYPROJECT).open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project", {})
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
