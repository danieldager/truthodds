"""Dump a traced verify run into per-post readable audit files (for auditor agents).

Usage (from src/):
    uv run python -m scripts.dump_verify_traces --dir <trace_dir> --out <dump_dir>

Each post_NN_<handle>.txt carries: post + claims + final ledger/verdict, then the full
step-by-step trace (queries, all hits + snippets, triage, scrapes, READ with the numbered
article text, STEP with the evidence table it saw), then round metadata (drops, picks).
system_prompts.txt gets the current pipeline prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import verify_tweet_claims as pv
from pipeline.verify_tweet_claims import rank_hits


def dump(trace_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = {p["post_id"]: i for i, p in enumerate(
        json.loads((trace_dir / "sample_posts.json").read_text()))}
    recs, seen = [], set()
    for shard in sorted(trace_dir.glob("results-*.jsonl")):
        for ln in shard.open():
            r = json.loads(ln)
            if r.get("ok") and r["post_id"] not in seen:
                seen.add(r["post_id"])
                recs.append(r)
    recs.sort(key=lambda r: order.get(r["post_id"], 999))

    for i, rec in enumerate(recs, 1):
        res = rec["result"]
        L = [f"POST #{i}  id={rec['post_id']}  @{res['handle']}  domain={res['domain']}",
             f"posted: {res['date']}   NG score: {res.get('ng_score')}   lean: {res.get('lean')}",
             f"loop {res.get('loop_version', '?')} | elapsed {rec['elapsed_s']}s | "
             f"rounds {len(res['rounds'])} | targeted claims {res.get('targeted')}",
             f"\nPOST TEXT:\n{res['text']}", "\nCLAIMS (the checkworthy ledger):"]
        for j, c in enumerate(res["claims"], 1):
            L.append(f"  {j}. {c['c']}")
        L.append(f"\nFINAL LEDGER: {res['ledger']}   coerced_open: {res['coerced_open']}")
        L.append(f"FINAL VERDICT: {json.dumps(res['verdict'], ensure_ascii=False)}")
        L.append("\n" + "=" * 90 + "\nSTEP-BY-STEP TRACE\n" + "=" * 90)
        for t in res["trace"]:
            if t["kind"] == "llm":
                lab = t["label"]
                if lab == "open" or lab.startswith("triage") or lab == "exa-query":
                    L.append(f"\n--- {lab.upper()} ({t['secs']}s) ---")
                    L.append(f"response: {json.dumps(t['response'], ensure_ascii=False)}")
                elif lab.startswith("read-"):
                    L.append(f"\n--- READ {lab} ({t['secs']}s) ---")
                    u = t["user"]
                    art = "ARTICLE" + u.split("ARTICLE", 1)[-1].split("POST (posted by", 1)[0]
                    L.append(art.strip()[:12000])
                    L.append(f"READ response: {json.dumps(t['response'], ensure_ascii=False)}")
                elif lab.startswith("step") or lab == "verdict-retry":
                    L.append(f"\n--- {'STEP ' + lab if lab.startswith('step') else 'VERDICT RE-ASK'} ({t['secs']}s) ---")
                    u = t["user"]
                    if "PRIOR QUERIES:" in u:
                        L.append("PRIOR QUERIES:" + u.split("PRIOR QUERIES:", 1)[1]
                                 .split("EVIDENCE TABLE:", 1)[0].rstrip())
                    if "EVIDENCE TABLE:" in u:
                        L.append("EVIDENCE TABLE (as fed to STEP):" +
                                 u.split("EVIDENCE TABLE:", 1)[1]
                                 .split("Update the ledger", 1)[0].rstrip()[:14000])
                    L.append(f"STEP response: {json.dumps(t['response'], ensure_ascii=False)}")
            elif t["kind"] in ("serper", "exa"):
                L.append(f"\n--- {t['kind'].upper()} SEARCH ({t['secs']}s) ---")
                L.append(f"query: {t['query']}\nexcluded domains: {t.get('exclude_domains')}")
                L.append("results in RANKED order (the id space triage picks and R-ids use):")
                for j, h in enumerate(rank_hits(t["hits"]), 1):
                    L.append(f"  [{j}] {h.get('url')}  ({h.get('date') or 'no date'})")
                    L.append(f"      snippet: {(h.get('snippet') or '')[:350]}")
            elif t["kind"] == "scrape":
                L.append(f"  scrape {'OK' if t['ok'] else 'FAILED'} {t['chars']}ch {t['url']}")
        L.append("\n" + "=" * 90 + "\nROUND METADATA (drops, triage picks)\n" + "=" * 90)
        for rd in res["rounds"]:
            L.append(json.dumps({k: rd.get(k) for k in
                                 ("round", "provider", "query", "dropped", "triage",
                                  "action", "step_query", "step_targets")}, ensure_ascii=False))
        (out_dir / f"post_{i:02d}_{res['handle']}.txt").write_text("\n".join(L))

    (out_dir / "system_prompts.txt").write_text(
        "QUERY_SYSTEM:\n" + pv.QUERY_SYSTEM + "\n\nTRIAGE_SYSTEM:\n" + pv.TRIAGE_SYSTEM +
        "\n\nREAD_SYSTEM:\n" + pv.READ_SYSTEM + "\n\nRESOLVE_SYSTEM:\n" + pv.RESOLVE_SYSTEM +
        "\n\nEXA_QUERY_SYSTEM:\n" + pv.EXA_QUERY_SYSTEM)
    return len(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = dump(Path(args.dir), Path(args.out))
    print(f"dumped {n} posts -> {args.out}")


if __name__ == "__main__":
    main()
