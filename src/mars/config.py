"""Model-agnostic YAML loading: ``_base_`` merge, corpus files, and ``--set``."""

from __future__ import annotations

import copy
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto ``base``.

    Nested dicts are merged; other values replace.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


_SCIENTIFIC = re.compile(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+")


def _scalar(value: str) -> Any:
    """Parse one override value as YAML, reading 1e-5 style numbers as floats.

    YAML 1.1 needs a decimal point in exponent notation (1.0e-5), so 1e-5 would
    otherwise arrive as a string.
    """
    return float(value) if _SCIENTIFIC.fullmatch(value.strip()) else yaml.safe_load(value)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse key.sub=value pairs into a nested dict. Values are YAML scalars."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            next_node = node.get(part)
            if not isinstance(next_node, dict):
                next_node = {}
                node[part] = next_node
            node = next_node
        node[leaf] = _scalar(value)
    return out


def _corpus_yaml_path(name: str) -> Path:
    """Resolve ``configs/datasets/<name>.yaml`` from cwd or the repo root."""
    candidates = [
        Path("configs/datasets") / f"{name}.yaml",
        Path(__file__).resolve().parents[2] / "configs" / "datasets" / f"{name}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no corpus YAML for {name!r}; expected configs/datasets/{name}.yaml")


def load_corpus_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/datasets/<name>.yaml`` without folding it into ``data:``."""
    path = _corpus_yaml_path(name)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"dataset config {path} must be a mapping")
    raw.pop("_base_", None)
    return raw


def load_yaml(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML mapping, following its _base_ chain of sibling files."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping")
    base_ref = raw.pop("_base_", None)
    if base_ref:
        raw = merge(load_yaml(path.parent / base_ref), raw)
    if overrides:
        raw = merge(raw, overrides)
    return raw


def build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a nested mapping, rejecting unknown keys.

    Nested dataclass fields are built recursively, except where the field type
    defines ``from_dict``, which is then used instead.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        if isinstance(ftype, type) and is_dataclass(ftype) and isinstance(value, dict):
            from_dict = getattr(ftype, "from_dict", None)
            kwargs[name] = from_dict(value) if from_dict else build_dataclass(ftype, value)
            continue
        kwargs[name] = value
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts."""
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj
