"""Stable identity for a prompt: first 12 hex of sha256 over its texts.

Stamped into each LLM stage's provenance so a corpus can be traced back to the
exact prompt wording that produced it.
"""
from __future__ import annotations

import hashlib


def prompt_hash(*texts: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:12]
