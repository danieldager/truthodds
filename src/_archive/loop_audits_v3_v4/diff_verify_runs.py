"""Diff two verify-loop runs over the same post set (docs/verify_loop_versions.md workflow).

Usage (from src/):
    uv run python -m scripts.diff_verify_runs <trace_dir_a> <trace_dir_b>

Prints per-post verdict/nudge/ledger/round changes plus aggregate movement, so a loop
iteration can be judged against its predecessor before touching a bigger set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d: str) -> dict[str, dict]:
    out = {}
    for shard in sorted(Path(d).glob("results-*.jsonl")):
        for ln in shard.open():
            r = json.loads(ln)
            if r.get("ok"):
                out[r["post_id"]] = r["result"]
    return out


def vstr(res: dict) -> str:
    v = res["verdict"] or {}
    return f"{v.get('veracity')}/{v.get('misinfo_type')}/{'NUDGE' if v.get('nudge') else 'pass'}"


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    print(f"{len(common)} common posts ({len(a)} in A, {len(b)} in B)\n")
    verdict_moves, nudge_flips, ledger_moves = [], [], []
    for pid in common:
        ra, rb = a[pid], b[pid]
        va, vb = (ra["verdict"] or {}), (rb["verdict"] or {})
        if va.get("veracity") != vb.get("veracity") or va.get("nudge") != vb.get("nudge"):
            verdict_moves.append(pid)
            if va.get("nudge") != vb.get("nudge"):
                nudge_flips.append(pid)
        led_diff = {c: (sa, rb["ledger"].get(c)) for c, sa in ra["ledger"].items()
                    if sa != rb["ledger"].get(c)}
        if led_diff:
            ledger_moves.append((pid, led_diff))
        if va.get("veracity") != vb.get("veracity") or va.get("nudge") != vb.get("nudge") or led_diff:
            print(f"@{ra['handle']} ({pid})")
            print(f"  post: {ra['text'][:110]}".replace("\n", " "))
            print(f"  verdict: {vstr(ra)} -> {vstr(rb)} | rounds {len(ra['rounds'])} -> {len(rb['rounds'])}")
            if led_diff:
                print(f"  ledger: {led_diff}")
            print()
    n_rounds_a = sum(len(r["rounds"]) for r in a.values()) / max(len(a), 1)
    n_rounds_b = sum(len(r["rounds"]) for r in b.values()) / max(len(b), 1)
    print(f"summary: {len(verdict_moves)} verdict changes, {len(nudge_flips)} nudge flips, "
          f"{len(ledger_moves)} posts with ledger movement")
    print(f"mean rounds: {n_rounds_a:.2f} -> {n_rounds_b:.2f}")
    for tag, recs in (("A", a), ("B", b)):
        vh: dict = {}
        for r in recs.values():
            k = (r["verdict"] or {}).get("veracity")
            vh[k] = vh.get(k, 0) + 1
        nudges = sum(1 for r in recs.values() if (r["verdict"] or {}).get("nudge"))
        print(f"{tag}: veracity {dict(sorted(vh.items(), key=lambda x: (x[0] is None, x[0])))} "
              f"| nudges {nudges}")


if __name__ == "__main__":
    main()
