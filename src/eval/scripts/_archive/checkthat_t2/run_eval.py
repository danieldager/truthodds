"""Run all systems on N posts from the CheckThat! Task 2 English dev split.

Writes predictions to results/predictions_<n>.jsonl with one row per
(system, example_id) pair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the project root importable so we can reuse openai client config
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from eval.scripts.checkthat_t2.load_dataset import load_eng_dev  # noqa: E402
from eval.scripts.checkthat_t2.prompts import SYSTEMS, baseline, extract_claim  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-oss-120b"
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)


def call(messages: list[dict[str, str]], model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=4000,  # reasoning models (GPT-OSS, Qwen3) spend budget on hidden reasoning first
    )
    return resp.choices[0].message.content or ""


def run_one(system_name: str, example_id: int, post: str, model: str) -> dict:
    builder = SYSTEMS[system_name]
    t0 = time.time()
    if builder is None:  # baseline
        pred_raw = baseline(post)
        pred = pred_raw
        err = None
    else:
        try:
            pred_raw = call(builder(post), model)
            pred = extract_claim(system_name, pred_raw)
            err = None
        except Exception as e:  # noqa: BLE001
            pred_raw = ""
            pred = ""
            err = f"{type(e).__name__}: {e}"
    return {
        "system": system_name,
        "id": example_id,
        "pred": pred,
        "pred_raw": pred_raw,
        "latency_s": round(time.time() - t0, 3),
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=5, help="number of dev posts")
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument(
        "-s", "--systems", nargs="+", default=list(SYSTEMS.keys()),
        help="subset of systems to run",
    )
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Groq model id")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    examples = load_eng_dev(limit=args.n)
    model_slug = args.model.replace("/", "_")
    out_path = args.output or Path(__file__).parent / "results" / f"predictions_{model_slug}_n{args.n}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(s, i, ex.post) for s in args.systems for i, ex in enumerate(examples)]
    print(f"Running {len(jobs)} jobs ({len(args.systems)} systems × {len(examples)} posts) on {args.model} → {out_path}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, s, i, p, args.model) for s, i, p in jobs]
        for j, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["error"]:
                print(f"  [{j}/{len(jobs)}] {r['system']}#{r['id']} ERROR: {r['error']}")
            elif j % 20 == 0 or j == len(jobs):
                print(f"  [{j}/{len(jobs)}] done")

    # Also dump gold for the scorer's convenience
    gold = [{"id": i, "post": ex.post, "gold": ex.gold} for i, ex in enumerate(examples)]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_path.with_suffix(".gold.jsonl"), "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"\nDone in {time.time() - t0:.1f}s. {len(results)} predictions written.")


if __name__ == "__main__":
    main()
