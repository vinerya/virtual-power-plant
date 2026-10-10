"""Process-tree RSS / CPU and machine description (read from Linux ``/proc``)."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def _children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name (field 2) may contain spaces; fields after it are fixed.
        fields = stat[stat.rfind(")") + 2 :].split()
        children.setdefault(int(fields[1]), []).append(int(entry.name))
    return children


def process_tree(pid: int) -> list[int]:
    """``pid`` and all its descendants (uvicorn workers are children)."""
    if not Path("/proc").is_dir():
        return [pid]
    tree = _children_map()
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(tree.get(p, []))
    return out


def rss_bytes(pids: list[int]) -> int | None:
    """Sum of resident set sizes, or None when it cannot be read."""
    total = 0
    found = False
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    found = True
                    break
        except OSError:
            continue
    return total if found else None


def cpu_seconds(pids: list[int]) -> float | None:
    """User + system CPU seconds consumed so far by ``pids``."""
    ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    total = 0.0
    found = False
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        total += (int(fields[11]) + int(fields[12])) / ticks
        found = True
    return total if found else None


def machine_info() -> dict[str, Any]:
    cpu_model = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    import sqlite3

    return {
        "cpu": cpu_model,
        "logical_cpus": os.cpu_count(),
        "memory_gib": mem_total,
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
    }
