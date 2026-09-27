"""LLM-as-judge for claim-extraction predictions.

Reads a predictions file written by run_eval.py, picks the first N posts, and
asks a judge model (default: Llama 4 Maverick on Groq — different family from
Kimi K2 and GPT-OSS, reducing self-bias) to score every system's prediction
1-5 on three rubric dimensions:

  * fidelity         — does the claim preserve the verifiable facts from the post?
  * decontextualized — can it be checked standalone (no pronouns, missing entities)?
  * conciseness      — single clear sentence, no extraneous content?

Each post produces one judge call that scores all systems together. Writes raw
judgments to {predictions_stem}.judge.jsonl and prints aggregate means.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_JUDGE = "meta-llama/llama-4-maverick-17b-128e-instruct"
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

RUBRIC = """You are an expert fact-checker evaluating claim-normalization systems. Each system takes a noisy social media post and outputs a single normalized factual claim suitable for downstream verification.

For each candidate, score it on three dimensions, integer 1-5 (1=very poor, 5=excellent):

- fidelity: preserves the verifiable factual content of the post without adding or omitting load-bearing detail
- decontextualized: independently checkable — no pronouns, no unresolved references, named entities and dates spelled out
- conciseness: a single clear sentence, no preamble, no rhetorical fluff, no list/multi-claim sprawl

You will also see a GOLD reference (a human-annotated normalized claim) for calibration. Do NOT just score by similarity to GOLD — score against the rubric. GOLD is one valid answer, not the only one.

Return strict JSON only, with this shape:
{
  "<system_name>": {"fidelity": <1-5>, "decontextualized": <1-5>, "conciseness": <1-5>},
  ...
}
No prose outside the JSON. Include every system listed below."""


def build_prompt(post: str, gold: str, candidates: dict[str, str]) -> list[dict[str, str]]:
    cand_block = "\n".join(f"- {name}: {pred!r}" for name, pred in candidates.items())
    user = (
        f"POST:\n{post}\n\n"
        f"GOLD (reference, one valid normalization):\n{gold}\n\n"
        f"CANDIDATES:\n{cand_block}\n\n"
        f"Return scores for: {sorted(candidates)}"
    )
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": user},
    ]


def parse_judge_json(raw: str, expected_systems: list[str]) -> dict[str, dict[str, int]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Sometimes the judge wraps with prose; grab the outermost JSON object.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in judge response: {raw[:200]}")
    data = json.loads(m.group(0))
    out: dict[str, dict[str, int]] = {}
    for sys_name in expected_systems:
        if sys_name not in data:
            raise ValueError(f"Judge omitted system {sys_name}: {raw[:200]}")
        s = data[sys_name]
        out[sys_name] = {
            "fidelity": int(s["fidelity"]),
            "decontextualized": int(s["decontextualized"]),
            "conciseness": int(s["conciseness"]),
        }
    return out


def judge_one(judge_model: str, post: str, gold: str, candidates: dict[str, str]) -> dict:
    messages = build_prompt(post, gold, candidates)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
        scores = parse_judge_json(raw, sorted(candidates))
        err = None
    except Exception as e:  # noqa: BLE001
        raw = ""
        scores = {}
        err = f"{type(e).__name__}: {e}"
    return {
        "scores": scores,
        "raw": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path,
                    help="path to predictions_<model>_n<N>.jsonl from run_eval.py")
    ap.add_argument("-n", "--n", type=int, default=100,
                    help="number of posts to judge (default 100)")
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="Groq model id for the judge")
    ap.add_argument("--skip-systems", nargs="+", default=["baseline"],
                    help="systems to exclude from judging (default: baseline)")
    args = ap.parse_args()

    gold_path = args.predictions.with_suffix(".gold.jsonl")
    gold_by_id: dict[int, dict] = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            gold_by_id[g["id"]] = g

    # Group predictions by example_id, keep only systems we want to judge
    preds_by_id: dict[int, dict[str, str]] = defaultdict(dict)
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["system"] in args.skip_systems:
                continue
            if r["error"]:
                continue
            preds_by_id[r["id"]][r["system"]] = r["pred"]

    ids = sorted(preds_by_id.keys())[: args.n]
    print(f"Judging {len(ids)} posts × {len(next(iter(preds_by_id.values())))} systems "
          f"with {args.judge}")

    out_path = args.predictions.with_suffix(".judge.jsonl")

    def task(example_id: int) -> dict:
        g = gold_by_id[example_id]
        cands = preds_by_id[example_id]
        result = judge_one(args.judge, g["post"], g["gold"], cands)
        result["id"] = example_id
        return result

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, i) for i in ids]
        for j, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["error"]:
                print(f"  [{j}/{len(ids)}] id={r['id']} ERROR: {r['error']}")
            elif j % 20 == 0 or j == len(ids):
                print(f"  [{j}/{len(ids)}] done")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate
    dims = ["fidelity", "decontextualized", "conciseness"]
    totals: dict[str, dict[str, list[int]]] = defaultdict(lambda: {d: [] for d in dims})
    for r in results:
        if r["error"]:
            continue
        for sys_name, s in r["scores"].items():
            for d in dims:
                totals[sys_name][d].append(s[d])

    print(f"\n{'System':<14} {'fidelity':>9} {'decontext':>10} {'concise':>9} {'mean':>7} {'N':>5}")
    print("-" * 60)
    rows = []
    for sys_name in sorted(totals):
        per_dim = {d: (sum(v) / len(v) if v else 0.0) for d, v in totals[sys_name].items()}
        mean = sum(per_dim.values()) / len(dims)
        n = len(totals[sys_name]["fidelity"])
        rows.append((sys_name, per_dim, mean, n))
    # Sort by mean descending
    rows.sort(key=lambda x: -x[2])
    for sys_name, per_dim, mean, n in rows:
        print(f"{sys_name:<14} {per_dim['fidelity']:>9.3f} {per_dim['decontextualized']:>10.3f} "
              f"{per_dim['conciseness']:>9.3f} {mean:>7.3f} {n:>5}")

    print(f"\nWrote {len(results)} judgments → {out_path}")
    print(f"Done in {time.time() - t0:.1f}s with judge={args.judge}")


if __name__ == "__main__":
    main()
