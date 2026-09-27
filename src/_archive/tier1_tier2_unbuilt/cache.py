"""Tier 1: Postgres + pgvector claims cache (growing claims DB / misinformation observatory).

Design: docs/tier1_cache_design.md. Two-stage decision — cosine ANN is a RECALL
filter (stage 1); the bidirectional equivalence GATE is the precision arbiter
(stage 2). Calibration: eval/scripts/cache_eval (gated precision ~0.98 flat across
the cosine sweep, so SIMILARITY_THRESHOLD is set low).

Three tables (every claim is stored — never collapsed; clusters are an overlay):
  clusters       one row per equivalence class: cluster_id, canonical_text,
                 centroid halfvec(384), verdict (jsonb ClaimVerdict), verdict_4class,
                 verdict_confidence (= stopped_reason), member_count, created_at, last_seen_at
  claims         one row per claim ever seen: claim_id, cluster_id, text, embedding
                 halfvec(384), language, source, created_at
  cluster_edges  (cluster_a, cluster_b, relation) — negation_of / related_independent
                 (observatory; populated in a later increment)

lookup(): embed -> ANN nearest claims -> gate vs each candidate cluster's canonical
AND nearest member -> on confirmed equivalence, fold the claim in and return the
verdict if within its (confidence-tiered) TTL, else None. write(): persist a fresh
Tier-2/3 verdict as a new cluster + its first member.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg
from openai import OpenAI
from psycopg.types.json import Jsonb

from config import (
    EMBEDDING_DIM,
    EXTRACTION_API_KEY,
    EXTRACTION_BASE_URL,
    NEON_DATABASE_URL,
)
from pipeline.config import (
    CACHE_GATE_MODEL,
    CACHE_TOP_K,
    RECHECK_AFTER_DAYS,
    RECHECK_AFTER_DAYS_LOWCONF,
    SIMILARITY_THRESHOLD,
)
from pipeline.embedding import embed
from pipeline.models import AtomicClaim, ClaimVerdict

_client: OpenAI | None = None

_GATE_SYSTEM = (
    "You are a strict logical-equivalence judge for a fact-checking cache. Two claims are "
    "EQUIVALENT only if a SINGLE fact-check verdict would apply identically to BOTH — i.e. "
    "they mutually entail each other. ANY of these verdict-flipping differences makes them "
    "NOT equivalent: negation (opposite truth value); different numbers / quantities / dates; "
    "different named entities; different scope or quantifier (some vs all). "
    'Respond with ONLY JSON: {"a_entails_b": bool, "b_entails_a": bool, '
    '"relation": "equivalent"|"negation"|"related"|"unrelated"}'
)


# --------------------------------------------------------------------------- helpers
def _conn() -> psycopg.Connection:
    if not NEON_DATABASE_URL:
        raise RuntimeError("NEON_DATABASE_URL is not set (see .env.example / docker-compose).")
    return psycopg.connect(NEON_DATABASE_URL)


def _vec(embedding: list[float] | np.ndarray) -> str:
    """pgvector literal, e.g. '[0.1,0.2,...]' — cast ::halfvec at the call site."""
    arr = np.asarray(embedding, dtype=np.float32)
    return "[" + ",".join(repr(float(x)) for x in arr) + "]"


def _claim_embedding(claim: AtomicClaim) -> np.ndarray:
    """The claim's (normalized) embedding — use the carried one, else compute it."""
    if claim.embedding:
        v = np.asarray(claim.embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v
    return embed(claim.text)


def _gate(a: str, b: str) -> dict:
    """Bidirectional equivalence judge. equivalent = mutual entailment (a<->b).

    The precision arbiter (catches negation / number / entity / scope flips cosine
    misses). Swappable for a negation-robust fine-tuned NLI run in both orders.
    """
    global _client
    if _client is None:
        _client = OpenAI(base_url=EXTRACTION_BASE_URL, api_key=EXTRACTION_API_KEY)
    msgs = [{"role": "system", "content": _GATE_SYSTEM},
            {"role": "user", "content": f"CLAIM A: {a}\nCLAIM B: {b}"}]
    try:
        r = _client.chat.completions.create(
            model=CACHE_GATE_MODEL, messages=msgs, temperature=0.0, max_tokens=200)
        txt = r.choices[0].message.content or ""
        d = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        equiv = bool(d.get("a_entails_b")) and bool(d.get("b_entails_a"))
        return {"equivalent": equiv, "relation": d.get("relation") or "unrelated"}
    except Exception:  # noqa: BLE001 — a gate error must never reuse a verdict
        return {"equivalent": False, "relation": "error"}


def _ttl_days(confidence: str) -> int:
    """Confident verdicts get the full recheck window; cut-off ones a short one."""
    return RECHECK_AFTER_DAYS if confidence == "confident" else RECHECK_AFTER_DAYS_LOWCONF


# --------------------------------------------------------------------------- schema
def init_schema() -> None:
    """Idempotent: extension + three tables + HNSW index. Run once from setup, not the hot path."""
    dim = EMBEDDING_DIM
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                canonical_text    text NOT NULL,
                centroid          halfvec({dim}) NOT NULL,
                verdict           jsonb NOT NULL,
                verdict_4class    text,
                verdict_confidence text,
                member_count      int NOT NULL DEFAULT 0,
                created_at        timestamptz NOT NULL DEFAULT now(),
                last_seen_at      timestamptz NOT NULL DEFAULT now()
            );""")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                cluster_id  uuid NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
                text        text NOT NULL,
                embedding   halfvec({dim}) NOT NULL,
                language    text,
                source      text,
                created_at  timestamptz NOT NULL DEFAULT now()
            );""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cluster_edges (
                cluster_a  uuid NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
                cluster_b  uuid NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
                relation   text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (cluster_a, cluster_b, relation)
            );""")
        # A1: index every claim embedding (max ANN recall; redundancy = many landing
        # pads per cluster). A2 migration when this index outgrows RAM (~5-10M vectors,
        # see docs/tier1_cache_design.md): drop this index and ANN over clusters.centroid
        # instead (the centroid is already maintained on every write/fold), then fetch
        # members by cluster_id for the gate. Switching is a config/query change, not a
        # data migration — which is exactly why we store every claim.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS claims_embedding_hnsw
            ON claims USING hnsw (embedding halfvec_cosine_ops);""")
        conn.commit()


# --------------------------------------------------------------------------- write
def write(verdict: ClaimVerdict, *, source: str | None = None,
          language: str | None = None) -> str:
    """Persist a fresh Tier-2/3 verdict as a NEW cluster + its first member claim.

    Returns the new cluster_id. Called on a cache MISS (the verify path produced a
    verdict no existing cluster matched). Equivalent recurrences are folded in by
    `lookup`, not here.
    """
    emb = _claim_embedding(verdict.claim)
    vlit = _vec(emb)
    confidence = verdict.stopped_reason or ("tier2" if verdict.tier_resolved == 2 else "")
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO clusters
                 (canonical_text, centroid, verdict, verdict_4class, verdict_confidence,
                  member_count)
               VALUES (%s, %s::halfvec, %s, %s, %s, 1)
               RETURNING cluster_id;""",
            (verdict.claim.text, vlit, Jsonb(verdict.model_dump()),
             verdict.verdict_4class, confidence))
        cluster_id = cur.fetchone()[0]
        cur.execute(
            """INSERT INTO claims (cluster_id, text, embedding, language, source)
               VALUES (%s, %s, %s::halfvec, %s, %s);""",
            (cluster_id, verdict.claim.text, vlit, language, source))
        conn.commit()
    return str(cluster_id)


# --------------------------------------------------------------------------- lookup
def lookup(claim: AtomicClaim, *, source: str | None = None,
           language: str | None = None) -> ClaimVerdict | None:
    """Return a reusable cached verdict for `claim`, or None (caller falls through).

    Stage 1: ANN nearest claims with cosine >= SIMILARITY_THRESHOLD (a recall filter).
    Stage 2: the bidirectional gate against each candidate cluster's canonical AND its
    nearest member. On confirmed equivalence within TTL: fold the claim into the
    cluster (member + centroid running-mean + last_seen) and return the verdict.
    """
    emb = _claim_embedding(claim)
    vlit = _vec(emb)
    max_dist = 1.0 - SIMILARITY_THRESHOLD  # cosine distance = 1 - cosine similarity

    with _conn() as conn, conn.cursor() as cur:
        # Nearest members above the cosine cut; dedup to distinct clusters (nearest first).
        cur.execute(
            """SELECT c.cluster_id, c.text,
                      cl.canonical_text, cl.verdict, cl.verdict_confidence,
                      cl.member_count, cl.centroid, cl.created_at,
                      (c.embedding <=> %s::halfvec) AS dist
               FROM claims c JOIN clusters cl ON c.cluster_id = cl.cluster_id
               WHERE (c.embedding <=> %s::halfvec) <= %s
               ORDER BY dist
               LIMIT %s;""",
            (vlit, vlit, max_dist, CACHE_TOP_K * 4))
        candidates: dict = {}
        for row in cur.fetchall():
            cid = row[0]
            if cid not in candidates:           # keep the nearest member per cluster
                candidates[cid] = row
            if len(candidates) >= CACHE_TOP_K:
                break

        for cid, row in candidates.items():
            (_, member_text, canonical, verdict_json, confidence,
             member_count, centroid_lit, created_at, _dist) = row
            # Stage 2: equivalent vs the canonical OR the nearest member (two-pronged).
            if not (_gate(claim.text, canonical)["equivalent"]
                    or _gate(claim.text, member_text)["equivalent"]):
                continue
            # TTL: stale -> miss (caller re-verifies; don't fold into a stale cluster).
            age = datetime.now(timezone.utc) - created_at
            if age > timedelta(days=_ttl_days(confidence or "")):
                return None
            _fold(conn, cid, claim.text, vlit, emb, member_count, centroid_lit,
                  source, language)
            return ClaimVerdict.model_validate(verdict_json)
    return None


def _fold(conn: psycopg.Connection, cluster_id, text: str, vlit: str,
          emb: np.ndarray, member_count: int, centroid_lit: str,
          source: str | None, language: str | None) -> None:
    """Add the confirmed-equivalent claim to its cluster + update centroid/count/last_seen."""
    old = np.array([float(x) for x in centroid_lit.strip("[]").split(",")], dtype=np.float32)
    new_centroid = (old * member_count + emb) / (member_count + 1)
    n = np.linalg.norm(new_centroid)
    if n:
        new_centroid = new_centroid / n
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO claims (cluster_id, text, embedding, language, source)
               VALUES (%s, %s, %s::halfvec, %s, %s);""",
            (cluster_id, text, vlit, language, source))
        cur.execute(
            """UPDATE clusters
               SET member_count = member_count + 1,
                   centroid = %s::halfvec,
                   last_seen_at = now()
               WHERE cluster_id = %s;""",
            (_vec(new_centroid), cluster_id))
        conn.commit()
