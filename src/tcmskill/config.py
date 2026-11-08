"""YAML config with attribute access, method presets and command-line overrides.

Usage:
    cfg = load_config("configs/default.yaml", method="ours", overrides=["skills.verify=false"])
    cfg.skills.verify  -> False
"""
from __future__ import annotations

import copy
import os
from typing import Any, Iterable

import yaml


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover - debugging aid
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def to_cfg(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: to_cfg(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_cfg(v) for v in obj]
    return obj


def to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def apply_override(cfg: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must look like a.b.c=value, got: {override}")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw) if raw != "" else ""
    cur = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(
    path: str = "configs/default.yaml",
    method: str | None = None,
    overrides: Iterable[str] = (),
    methods_path: str | None = None,
) -> Cfg:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if method:
        methods_path = methods_path or os.path.join(os.path.dirname(path), "methods.yaml")
        with open(methods_path, encoding="utf-8") as f:
            presets = yaml.safe_load(f) or {}
        if method not in presets:
            raise KeyError(f"unknown method '{method}'. Available: {sorted(presets)}")
        preset = copy.deepcopy(presets[method])
        cfg.setdefault("method", {})
        cfg["method"]["name"] = method
        cfg["method"]["type"] = preset.pop("type")
        cfg["method"]["description"] = preset.pop("description", "")
        deep_update(cfg, preset)
    for o in overrides or ():
        apply_override(cfg, o)
    return to_cfg(cfg)
