"""Isolate all tests under a temporary FLEET_HOME."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def fleet_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "fleet_home"
    home.mkdir()
    monkeypatch.setenv("FLEET_HOME", str(home))
    monkeypatch.chdir(ROOT)
    return home


@pytest.fixture()
def root() -> Path:
    return ROOT
