"""Bundled URDB-shaped tariff presets (``vpp/tariffs/presets/*.json``).

Presets are starting points for creating a tariff without an OpenEI key.
Each file documents its provenance in ``_comment`` / ``_source_date``;
``illustrative_*`` presets are generic structures, not any utility's schedule.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PRESET_DIR = Path(__file__).resolve().parent / "presets"


@lru_cache(maxsize=1)
def _load_all() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        with open(path) as f:
            out[path.stem] = json.load(f)
    return out


def list_presets() -> list[tuple[str, dict[str, Any]]]:
    return list(_load_all().items())


def get_preset(preset_id: str) -> dict[str, Any] | None:
    data = _load_all().get(preset_id)
    return json.loads(json.dumps(data)) if data is not None else None  # defensive copy
