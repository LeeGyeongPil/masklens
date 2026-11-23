"""The CI workflow stays minimal and its pinned install list matches uv.lock (stdlib only)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_workflow_is_minimal() -> None:
    assert len(re.findall(r"^  [A-Za-z_-]+:\s*$", WORKFLOW.split("jobs:")[1], re.M)) == 1
    assert "runs-on: ubuntu-latest" in WORKFLOW
    assert "timeout-minutes:" in WORKFLOW
    assert "permissions:\n  contents: read" in WORKFLOW
    assert sorted(re.findall(r"uses:\s*(\S+)", WORKFLOW)) == [
        "actions/checkout@v6",
        "actions/setup-python@v6",
    ]
    assert re.findall(r'python-version:\s*"([^"]+)"', WORKFLOW) == ["3.13"]
    assert "python -m pip install -r requirements-ci.txt" in WORKFLOW
    assert "python -m pytest -q" in WORKFLOW
    for banned in ("matrix", "schedule", "cron", "if:", "uv ", "uname", "-rA"):
        assert banned not in WORKFLOW, banned


def test_ci_requirements_match_lock() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for pkg in lock["package"]:
        if "version" in pkg:
            locked.setdefault(pkg["name"], set()).add(pkg["version"])
    pinned = 0
    for line in (ROOT / "requirements-ci.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not m:
            continue
        name, version = m.group(1).lower().replace("_", "-"), m.group(2)
        assert version in locked.get(name, set()), f"{name}=={version} not in uv.lock"
        pinned += 1
    assert pinned > 10
