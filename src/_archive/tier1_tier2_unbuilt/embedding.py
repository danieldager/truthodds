"""Shared sentence-embedding singleton.

Used by `verify._is_redundant` for query-similarity checks, and (later) by
`extract` to embed atomic claims for the Tier 1 cache key. Lazy-loaded —
the model only loads on first call.
"""
from __future__ import annotations

import os
import threading

# Avoid the HF tokenizers fork/thread warning + potential deadlock under our ThreadPools.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def embed(text: str) -> np.ndarray:
    """Return a normalised embedding vector. Cosine similarity == dot product.

    Thread-safe lazy load (double-checked locking): concurrent first-callers — e.g.
    verify_run's worker pool — must not race on SentenceTransformer init, which segfaults.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL)
    vec = _model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)
