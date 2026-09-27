"""Dev helper — emit 5-row stub parquets matching every upstream stage schema.

Lets stage7/stage6/analyze_funnel be smoke-tested end-to-end before Sessions
1-2 land the real Stage 0/1/2/5 outputs. Writes to ``data/_stub/`` by default
so it never collides with the canonical ``data/*_x862.parquet`` files.

Stub design (5 posts) exercises every branch:
  p1 en  in-scope (embed+llm), 9 claims  -> over-decomposition; FCT matches
  p2 fr  in-scope (embed+llm), 1 claim   -> French FCT match
  p3 es  embed=F / llm=T, has_claim=F    -> embed↔LLM disagreement, no claims
  p4 en  out (sports)                    -> no extraction row
  p5 zxx out (media-only, char_len<5)    -> no extraction row
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.scripts.extraction_grading.extract import OUTPUT_SCHEMA as EXTRACT_SCHEMA  # noqa: E402
from eval.scripts.extraction_grading.judge import PER_CLAIM_SCHEMA  # noqa: E402
from eval.scripts.claim_cascade.stage0_build_posts import (  # noqa: E402
    OUTPUT_SCHEMA as POSTS_SCHEMA,
)

P = ["p1", "p2", "p3", "p4", "p5"]
LANGS = ["en", "fr", "es", "en", "zxx"]

# (claim_text, misinfo_candidate, fable_checkworthy)
P1_CLAIMS = [
    ("Vaccines cause autism.", True, True),
    ("The Earth is flat.", True, False),               # 4-prong only
    ("The 2020 US election was stolen.", True, True),
    ("Tariffs raise consumer prices.", True, False),    # 4-prong only
    ("I had coffee this morning.", False, False),
    ("Cats always win.", False, False),
    ("5G towers spread coronavirus.", False, True),     # FABLE only
    ("The sky is blue.", False, False),
    ("Drinking bleach cures COVID-19.", True, True),
]
P2_CLAIMS = [
    ("Le vaccin contre la COVID-19 contient des puces électroniques.", True, True),
]


def _posts() -> pl.DataFrame:
    rows = []
    for pid, lang in zip(P, LANGS):
        text = {
            "p1": "Big thread on vaccines, elections, tariffs and more.",
            "p2": "Le vaccin contre la COVID-19 contient des puces.",
            "p3": "Reflexiones personales sobre la economía de mi barrio.",
            "p4": "What a game last night, incredible finish! #football",
            "p5": "📷",
        }[pid]
        rows.append({
            "post_id": pid, "rest_id": pid + "000", "text": text,
            "own_text": text, "quoted_text": "", "card_snippet": "",
            "lang": lang, "created_at": "Mon Jun 01 04:58:38 +0000 2026",
            "favorite_count": 10, "retweet_count": 2, "reply_count": 1,
            "quote_count": 0, "bookmark_count": 0, "views_count": 100,
            "author_handle": "user_" + pid, "author_followers": 500,
            "author_verified": False, "has_community_note": False,
            "is_quote_status": False, "char_len": len(text),
        })
    df = pl.DataFrame(rows, schema=POSTS_SCHEMA)
    return df.with_columns(
        pl.col("created_at").str.to_datetime(
            format="%a %b %d %H:%M:%S %z %Y", strict=False, time_zone="UTC"
        )
    )


def _topics() -> pl.DataFrame:
    labels = ["politics", "health", "sports"]
    embed_flags = [True, True, False, False, False]   # p3 embed misses what LLM catches
    rows = []
    for pid, in_embed in zip(P, embed_flags):
        rows.append({
            "post_id": pid,
            "top_topic": "health" if in_embed else "sports",
            "top_cosine": 0.42 if in_embed else 0.11,
            "in_scope_embed": in_embed,
            "margin": 0.08,
            "topic_scores": [0.42, 0.30, 0.11],
            "topic_labels": labels,
            "post_embedding": [0.0] * 384,
        })
    return pl.DataFrame(rows, schema={
        "post_id": pl.String, "top_topic": pl.String, "top_cosine": pl.Float64,
        "in_scope_embed": pl.Boolean, "margin": pl.Float64,
        "topic_scores": pl.List(pl.Float64), "topic_labels": pl.List(pl.String),
        "post_embedding": pl.List(pl.Float64),
    })


def _scope() -> pl.DataFrame:
    llm_flags = [True, True, True, False, False]      # p3 in-scope by LLM
    rows = []
    for pid, in_llm in zip(P, llm_flags):
        rows.append({
            "post_id": pid, "in_scope_llm": in_llm,
            "topic_llm": "health" if in_llm else "sports",
            "reason": "stub", "raw_response": "{}", "latency_s": 0.1, "error": None,
        })
    return pl.DataFrame(rows, schema={
        "post_id": pl.String, "in_scope_llm": pl.Boolean, "topic_llm": pl.String,
        "reason": pl.String, "raw_response": pl.String, "latency_s": pl.Float64,
        "error": pl.String,
    })


def _extractions() -> pl.DataFrame:
    # Only in-scope-by-LLM posts went through Stage 3 (p1, p2, p3).
    rows = [
        {"post_id": "p1", "has_claim": True, "n_claims": len(P1_CLAIMS),
         "claims": [c[0] for c in P1_CLAIMS], "raw_response": "{}",
         "latency_s": 1.0, "error": None},
        {"post_id": "p2", "has_claim": True, "n_claims": len(P2_CLAIMS),
         "claims": [c[0] for c in P2_CLAIMS], "raw_response": "{}",
         "latency_s": 1.0, "error": None},
        {"post_id": "p3", "has_claim": False, "n_claims": 0, "claims": [],
         "raw_response": "{}", "latency_s": 0.8, "error": None},
    ]
    return pl.DataFrame(rows, schema=EXTRACT_SCHEMA)


def _judge() -> pl.DataFrame:
    rows = []
    for pid, claims in (("p1", P1_CLAIMS), ("p2", P2_CLAIMS)):
        for i, (text, misinfo, _fable) in enumerate(claims):
            rows.append({
                "post_id": pid, "claim_index": i, "claim_text": text,
                "fidelity": 4, "decontextualized": 4,
                "verifiability": 5 if misinfo else 2,
                "misinfo_candidate": misinfo,
                "raw_response": "{}", "latency_s": 0.5, "error": None,
            })
    return pl.DataFrame(rows, schema=PER_CLAIM_SCHEMA)


def _fable() -> pl.DataFrame:
    rows = []
    for pid, claims in (("p1", P1_CLAIMS), ("p2", P2_CLAIMS)):
        for i, (text, _misinfo, fable) in enumerate(claims):
            dim = 4 if fable else 2
            total = dim * 5
            rows.append({
                "post_id": pid, "claim_index": i, "claim_text": text,
                "fragmentation": dim, "actionability": dim, "believability": dim,
                "spread_likelihood": dim, "exploitativeness": dim,
                "fable_total": total, "fable_checkworthy": total >= 15,
                "raw_response": "{}", "latency_s": 0.5, "error": None,
            })
    return pl.DataFrame(rows, schema={
        "post_id": pl.String, "claim_index": pl.Int64, "claim_text": pl.String,
        "fragmentation": pl.Int64, "actionability": pl.Int64,
        "believability": pl.Int64, "spread_likelihood": pl.Int64,
        "exploitativeness": pl.Int64, "fable_total": pl.Int64,
        "fable_checkworthy": pl.Boolean, "raw_response": pl.String,
        "latency_s": pl.Float64, "error": pl.String,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "data" / "_stub")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "posts_x862.parquet": _posts(),
        "topics_x862.parquet": _topics(),
        "scope_x862.parquet": _scope(),
        "stage3_extractions.parquet": _extractions(),
        "stage4_judgments.parquet": _judge(),
        "fable_x862.parquet": _fable(),
    }
    for name, df in writers.items():
        df.write_parquet(args.out_dir / name)
        print(f"wrote {df.height:>2} rows -> {args.out_dir / name}")


if __name__ == "__main__":
    main()
