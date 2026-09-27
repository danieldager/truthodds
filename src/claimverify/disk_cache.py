"""On-disk JSON cache, same shape as pipeline/disk_cache.py but with a CONFIGURABLE root.

Namespaces used here: serper (parsed results), serper_raw (raw Serper JSON), exa,
scrape (url -> {"text", "source", ...}), llm (exact-match chat response cache).
The default root is src/pipeline/.cache so the scrape/serper entries written by the
production loop and the urn runs are reused; point `set_dir()` elsewhere for a cold run.
Env CLAUDE_PIPELINE_CACHE_BYPASS=1 skips reads (writes still happen).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_BYPASS = os.environ.get("CLAUDE_PIPELINE_CACHE_BYPASS") == "1"
_DIR = Path(__file__).resolve().parent.parent / "pipeline" / ".cache"


def set_dir(path: str | Path) -> None:
    global _DIR
    _DIR = Path(path)


def get_dir() -> Path:
    return _DIR


def _path(namespace: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _DIR / namespace / f"{h}.json"


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
    return "\x1f".join(parts)
