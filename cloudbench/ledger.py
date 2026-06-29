"""Persistent ledger of cloud resources the harness has created.

This is the backstop that survives a hard kill (SIGKILL, power loss, closed
laptop) where in-process ``finally`` cleanup never runs. A record is written
to disk the moment we *attempt* to create a resource, and removed only after
it is confirmed destroyed. Anything left in the ledger is a possible leak —
``python -m cloudbench.cli reap`` (a.k.a. ``make reap``) destroys it.

Records live in results/cloud/.active/<resource_name>.json and are tiny JSON
blobs carrying everything needed to destroy the resource out-of-process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .config import RESULTS_DIR

ACTIVE_DIR = RESULTS_DIR / ".active"


def _record_path(resource_name: str) -> Path:
    safe = resource_name.replace("/", "_")
    return ACTIVE_DIR / f"{safe}.json"


def write_record(record: dict) -> None:
    """Create or update a ledger record (idempotent, keyed by resource_name)."""
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = _record_path(record["resource_name"])
    # Merge with any existing record so we never lose an instance_id.
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            existing.update({k: v for k, v in record.items() if v is not None})
            record = existing
        except (json.JSONDecodeError, OSError):
            pass
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(path)  # atomic on POSIX


def remove_record(resource_name: str) -> None:
    path = _record_path(resource_name)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def list_records() -> List[dict]:
    if not ACTIVE_DIR.exists():
        return []
    out: List[dict] = []
    for f in sorted(ACTIVE_DIR.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out
