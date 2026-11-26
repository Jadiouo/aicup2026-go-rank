from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .errors import ConfigError

REQUIRED_KEYS = ("seed",)


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    p = Path(path)
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {p}: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"config {p} is not a mapping")
    if overrides:
        cfg.update(overrides)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"config {p} missing required keys: {missing}")
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
