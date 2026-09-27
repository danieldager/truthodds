"""Tiny on-disk JSON cache for deterministic eval re-runs.

Keys are hashed; values are JSON-serialisable. Three namespaces:
  - serper:    query string -> list[dict] (Serper results)
  - scrape:    url -> {"text": str | None}
  - summarise: (url, claim_text) -> EvidenceDoc as dict

Set env CLAUDE_PIPELINE_CACHE_BYPASS=1 to skip reads and force fresh calls.
The cache always writes regardless (so a bypassed run refreshes the cache).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_BYPASS = os.environ.get("CLAUDE_PIPELINE_CACHE_BYPASS") == "1"
_DEFAULT_DIR = Path(__file__).parent / ".cache"


def _path(namespace: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _DEFAULT_DIR / namespace / f"{h}.json"


def get(namespace: str, key: str) -> Any | None:
    if _BYPASS:
        return None
    p = _path(namespace, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def set_(namespace: str, key: str, value: Any) -> None:
    p = _path(namespace, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(p)


def make_key(*parts: str) -> str:
    """Join parts with a separator unlikely to appear in any of them."""
    return "\x1f".join(parts)
