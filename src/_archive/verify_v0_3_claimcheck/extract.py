"""Claim extraction: post -> list[AtomicClaim].

One LLM call decomposes the post into atomic, independently checkable claims.
Each claim's text is embedded with the shared MiniLM model and the
(text, embedding) pair is the cache key for the rest of the pipeline.
"""
from __future__ import annotations

from pipeline.models import AtomicClaim


def extract(post: str) -> list[AtomicClaim]:
    """Decompose a post into atomic claims. Returns [] for posts with no factual content.

    TODO: lift the decomposition prompt from `_archive/v0_1/pipeline/claim_extraction.py`
    (the `_SYSTEM` constant); drop the `questions` generation step (we don't use
    verifying questions in v0.2). Use `EXTRACTION_MODEL` via the DeepInfra endpoint.
    Embed each claim with `_embed()` before returning.
    """
    raise NotImplementedError


def _embed(text: str) -> list[float]:
    """Embed a claim string with `EMBEDDING_MODEL`.

    TODO: lazy-load sentence-transformers `EMBEDDING_MODEL`. Normalise embeddings
    so downstream cosine == dot. Cache the model instance at module level.
    """
    raise NotImplementedError
