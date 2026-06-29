"""Loads and resolves benchmark definitions from configs/benchmarks.yaml.

The harness is intentionally a *zero-dependency control plane*: if PyYAML is
available (e.g. inside the project's uv env) it is used, otherwise a tiny
stdlib parser handles the simple ``name: {key: value}`` schema this file uses.
That keeps the harness runnable with a bare ``python3`` — no install, no
project deps, nothing to pollute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import BENCHMARKS_YAML, BenchmarkSpec

try:
    import yaml  # PyYAML if present (optional)
except ImportError:  # pragma: no cover - fallback path is exercised in tests
    yaml = None


def _coerce_scalar(value: str):
    if value == "":
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_parse(text: str) -> dict:
    """Parse the restricted schema used by benchmarks.yaml without PyYAML.

    Supports: top-level mapping keys (no indent) and 2-space-indented
    ``key: value`` scalar pairs. Full-line ``#`` comments and blanks are
    ignored. This is NOT a general YAML parser — only our own file shape.
    """
    data: dict = {}
    current = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            raise ValueError(f"benchmarks.yaml line {lineno}: expected 'key: value'")
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip()
        if indent == 0:
            current = key
            data[current] = {}
        else:
            if current is None:
                raise ValueError(
                    f"benchmarks.yaml line {lineno}: indented entry before any name")
            data[current][key] = _coerce_scalar(value)
    return data


def _load_raw(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"benchmark registry not found: {path}")
    text = path.read_text()
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _minimal_yaml_parse(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping of name -> benchmark spec")
    return data


def load_benchmarks(path: Path = BENCHMARKS_YAML) -> Dict[str, BenchmarkSpec]:
    raw = _load_raw(path)
    benches: Dict[str, BenchmarkSpec] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"benchmark {name!r} must be a mapping")
        if "command" not in entry:
            raise ValueError(f"benchmark {name!r} is missing required 'command'")
        benches[name] = BenchmarkSpec(
            name=name,
            command=str(entry["command"]),
            gpu=str(entry.get("gpu", "none")),
            ngpus=int(entry.get("ngpus", 0)),
            timeout_minutes=int(entry.get("timeout_minutes", 30)),
            description=str(entry.get("description", "")),
        )
    return benches


def get_benchmark(name: str, path: Path = BENCHMARKS_YAML) -> BenchmarkSpec:
    benches = load_benchmarks(path)
    if name not in benches:
        available = ", ".join(sorted(benches)) or "(none)"
        raise KeyError(f"unknown benchmark {name!r}. Available: {available}")
    return benches[name]
