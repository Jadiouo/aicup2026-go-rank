"""Shared fixtures, markers and the require_pkg helper."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL_CMD = "python3 -m venv .venv && .venv/bin/python -m pip install pyarrow lightgbm sgfmill"


def require_pkg(name: str):
    """Import a required package or FAIL (never skip) with the install command (AC-ENV-1)."""
    try:
        return importlib.import_module(name)
    except ImportError:
        pytest.fail(f"required package '{name}' is missing. Install: {INSTALL_CMD}", pytrace=False)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def training_dir() -> Path:
    d = ROOT / "data" / "training"
    if not (d / "train_A.csv").exists():
        pytest.skip(f"real data not found: {d}")
    return d
