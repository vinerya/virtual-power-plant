"""Run every script in ``examples/`` so they cannot silently rot.

Each example runs in a subprocess (fresh interpreter, like a user would run
it) from an empty temporary working directory, so an example that writes
into the current directory shows up as a failure here.  All current
examples are self-contained: none needs a server or external service.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# Extra CLI arguments per example (keyed by file name).
EXAMPLE_ARGS: dict[str, list[str]] = {}

# Examples that need a running server / external service: name -> skip reason.
NEEDS_EXTERNAL_SERVICE: dict[str, str] = {}

EXAMPLES = sorted(p.name for p in EXAMPLES_DIR.glob("*.py"))


def test_examples_are_discovered():
    assert EXAMPLES, f"no examples found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name: str, tmp_path: Path) -> None:
    if name in NEEDS_EXTERNAL_SERVICE:
        pytest.skip(NEEDS_EXTERNAL_SERVICE[name])

    env = os.environ.copy()
    # Import the checkout under test even when another copy is installed.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH")) if p
    )
    env["PYTHONIOENCODING"] = "utf-8"  # the examples print emoji / symbols

    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name), *EXAMPLE_ARGS.get(name, [])],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert result.returncode == 0, (
        f"{name} exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
    )
    assert "Traceback" not in result.stderr, result.stderr[-4000:]
    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert not leftovers, f"{name} wrote files into the working directory: {leftovers}"
