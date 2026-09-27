"""LLM-synthesize a canonical claim per multi-member cluster (alt to the medoid).

Reads novel_claims_<ext>.parquet (from dedup_claims.py, which carries the cluster
`members`). For each multi-member cluster, asks an LLM to write ONE clean,
self-contained, attribution-stripped canonical claim that a single fact-check
would resolve. Singletons keep their (medoid) claim. Writes a parallel
novel_claims_synth_<ext>.parquet with the same schema as the medoid file, so
run_fct.py can query either and we can compare match rates.

  uv run python -m eval.scripts.feed_study.synth_representatives \
      --claims eval/scripts/feed_study/data/sample1000/novel_claims_gpt-oss-120b.parquet
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
_FENCE = re.compile(r"^```.*?\n|```$", re.DOTALL)
_client: OpenAI | None = None

SYNTH_SYSTEM = """You are given several paraphrases/variants of what is essentially the SAME factual claim, each extracted from a social-media post. Write ONE clean, self-contained, canonical version of that claim — the form a fact-checker would check.

Rules:
- Capture the shared factual assertion all variants are about.
- STRIP source attribution ("X said that…", "according to a Twitter user…") UNLESS the attributed speaker is a public figure/institution AND the point of the claim is who said it — in that case keep the speaker.
- Resolve to a single, standalone, verifiable sentence. No commentary.

Output ONLY the canonical claim sentence, nothing else."""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
    return _client


def _retryable(e: Exception) -> bool:
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    s = getattr(e, "status_code", None)
    return isinstance(e, APIError) and isinstance(s, int) and s >= 500


def synth(members: list[str], model: str) -> tuple[str, str | None]:
    msgs = [{"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": "VARIANTS:\n" + "\n".join(f"- {m}" for m in members)}]
    for attempt in range(6):
        try:
            r = get_client().chat.completions.create(model=model, messages=msgs,
                                                     temperature=0.1, max_tokens=400)
            txt = _FENCE.sub("", (r.choices[0].message.content or "").strip()).strip().strip('"')
            return txt, None
        except Exception as e:  # noqa: BLE001
            if attempt == 5 or not _retryable(e):
                return "", f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt + random.random(), 30.0))
    return "", "unreachable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", type=Path, required=True, help="novel_claims_<ext>.parquet (with members)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=8)
    args = ap.parse_args()

    nc = pl.read_parquet(args.claims)
    multi = [(r["cluster_id"], r["members"]) for r in nc.iter_rows(named=True) if r["n_members"] > 1]
    print(f"clusters: {nc.height} | multi-member (to synthesize): {len(multi)}")

    synth_map: dict[int, str] = {}
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(synth, list(m), args.model): cid for cid, m in multi}
        for i, fut in enumerate(as_completed(futs), 1):
            cid = futs[fut]
            txt, err = fut.result()
            if err or not txt:
                n_err += 1
            else:
                synth_map[cid] = txt
            if i % 10 == 0 or i == len(multi):
                print(f"  [{i}/{len(multi)}] synthesized")
    print(f"synthesized {len(synth_map)} (errors/empty: {n_err})")

    # representative = synthesized for multi-member, medoid for singletons (+ fallback)
    out = nc.with_columns(
        pl.col("cluster_id").map_elements(
            lambda c: synth_map.get(c), return_dtype=pl.String).alias("_synth")
    ).with_columns(
        pl.when(pl.col("_synth").is_not_null()).then(pl.col("_synth"))
          .otherwise(pl.col("representative")).alias("representative")
    ).drop("_synth")

    out_path = args.claims.with_name(args.claims.stem + "_synth.parquet")
    out.write_parquet(out_path)
    print(f"wrote {out.height} synth representatives -> {out_path}")
    # show a few synth vs medoid
    for r in out.filter(pl.col("n_members") > 1).head(6).iter_rows(named=True):
        print(f"  synth: {r['representative'][:80]}")


if __name__ == "__main__":
    main()
