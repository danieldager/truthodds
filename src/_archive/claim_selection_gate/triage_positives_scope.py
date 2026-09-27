"""Independent topic-scan of the gate positives, to find OUT-OF-SCOPE fact-checks (entertainment / sports
/ celebrity / product trivia a fact-checker debunked but our societal-consequence gate is designed to drop).

These are effectively mislabels in the positive set; reclassifying them as NEGATIVES de-confounds the
recall metric (clog 260626). Run over ALL positives (dev+test, kept and dropped) — reclassifying only the
gate-dropped ones would circularly inflate the metrics. Uses DeepSeek-V4-Flash (text), a DIFFERENT model
from the gate (Qwen3-VL), to reduce circularity. Conservative: any societal angle, or an image-borne
claim the text can't show, defaults to IN-SCOPE.

  uv run python -m eval.scripts.triage_positives_scope   ->  eval/data/positives_scope_triage.parquet
"""
from __future__ import annotations

import json
import os
import re
import sys

import polars as pl
import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, EXTRACTION_MODEL
from eval.scripts._pool import pooled_checkpointed

URL = f"{EXTRACTION_BASE_URL}/chat/completions"
HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}

SCOPE_SYSTEM = (
    "You label what a social-media post is ABOUT, to decide if it is IN SCOPE for a misinformation "
    "fact-checking pipeline whose scope is matters of SOCIETAL CONSEQUENCE.\n\n"
    "IN-SCOPE topics: politics & elections, government & policy, corruption, war & geopolitics, terrorism, "
    "public health & safety, crime, disasters, science & climate, immigration, the economy.\n"
    "OUT-OF-SCOPE topics: sports results/transfers/players, entertainment/film/TV/music, celebrity "
    "personal life & gossip, consumer products & brands & marketing, personal anecdotes/jokes/trivia.\n\n"
    "Rule: a post is OUT-OF-SCOPE only if its claim is PURELY an out-of-scope topic with NO "
    "societal-consequence angle. If it involves a death, crime, accident, public-health/safety issue, a "
    "political figure acting in office, an election, or public policy, it is IN-SCOPE even if a celebrity "
    "or sport is involved. If the post text is only a brief caption/reaction and the real claim may be in "
    "an attached image you cannot see, default to IN-SCOPE.\n\n"
    'Output strictly valid JSON: {"topic":"<politics|election|policy|corruption|geopolitics|terrorism|'
    'health|safety|crime|disaster|science|immigration|economy|sports|entertainment|celebrity|product|'
    'personal|other>","out_of_scope":true|false,"reason":"<short>"}'
)


def _obj(txt: str) -> dict:
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S), re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _classify(text: str) -> dict:
    body = {"model": EXTRACTION_MODEL,
            "messages": [{"role": "system", "content": SCOPE_SYSTEM},
                         {"role": "user", "content": f"POST: {text}"}],
            "temperature": 0, "response_format": {"type": "json_object"}, "max_tokens": 200}
    g: dict = {}
    for _ in range(2):
        r = requests.post(URL, headers=HDR, json=body, timeout=90)
        r.raise_for_status()
        g = _obj(r.json()["choices"][0]["message"]["content"])
        if g:
            break
    return g


def main() -> None:
    pos = pl.concat([pl.read_parquet("eval/data/positives_dev.parquet").with_columns(pl.lit("dev").alias("fold")),
                     pl.read_parquet("eval/data/positives_test.parquet").with_columns(pl.lit("test").alias("fold"))],
                    how="diagonal_relaxed")
    rows = pos.to_dicts()
    out = "eval/data/positives_scope_triage.parquet"
    done = set(pl.read_parquet(out)["raw_context"].to_list()) if os.path.exists(out) else set()
    todo = [i for i, r in enumerate(rows) if r["raw_context"] not in done]
    print(f"{len(rows)} positives | todo {len(todo)}", flush=True)
    res: dict = {}

    def flush():
        if res:
            new = pl.DataFrame(list(res.values()))
            comb = new if not os.path.exists(out) else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique("raw_context", keep="last")
            comb.write_parquet(out)

    def work(i):
        r = rows[i]
        try:
            g = _classify(r["raw_context"])
            return i, {"raw_context": r["raw_context"], "source": r.get("source"), "fold": r.get("fold"),
                       "topic": g.get("topic"), "out_of_scope": g.get("out_of_scope"), "reason": (g.get("reason") or "")[:200]}
        except Exception as e:
            return i, {"raw_context": r["raw_context"], "out_of_scope": None, "reason": f"ERR {e}"[:150]}

    ab = pooled_checkpointed(todo, work, lambda i, p: res.__setitem__(rows[i]["raw_context"], p), flush, 8, "scope", checkpoint_every=100)
    flush()

    df = pl.read_parquet(out)
    oos = df.filter(pl.col("out_of_scope") == True)  # noqa: E712
    print(f"\nOUT-OF-SCOPE: {oos.height}/{df.height}")
    print("by topic:", oos.group_by("topic").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
    if ab:
        sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
