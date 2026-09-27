"""Merge returned `labels_<id>.json` files into one table + agreement report.

Annotators email back the JSON the tool downloads. Drop them all in a folder and:

  uv run collect.py --in returned --out labels.jsonl

Outputs:
  - labels.jsonl  : one row per (post_id, annotator) with gate/V/R/H.
  - stdout report : per-dimension Cohen's kappa (pairwise) + Fleiss' kappa on the
                    OVERLAP subset (posts labelled by >=2 annotators), a list of
                    disagreements to adjudicate, and median seconds/post per pass.

No deps beyond the stdlib — kappa is computed directly so this stays portable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DIMS = ["gate", "V", "R", "H"]


def load_returned(in_dir: Path) -> list[dict]:
    files = sorted(in_dir.glob("labels_*.json"))
    if not files:
        raise SystemExit(f"no labels_*.json in {in_dir}")
    out = []
    for fp in files:
        d = json.loads(fp.read_text())
        if "labels" not in d or "annotator" not in d:
            raise SystemExit(f"{fp.name}: not a results file (missing labels/annotator)")
        out.append(d)
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for a list of (rater_a_label, rater_b_label)."""
    if not pairs:
        return None
    n = len(pairs)
    cats = sorted({x for p in pairs for x in p})
    po = sum(a == b for a, b in pairs) / n
    pa = {c: sum(a == c for a, _ in pairs) / n for c in cats}
    pb = {c: sum(b == c for _, b in pairs) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(rows: list[list[str]]) -> float | None:
    """Fleiss' kappa. rows = per-item list of category labels from >=2 raters."""
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return None
    cats = sorted({x for r in rows for x in r})
    # require constant rater count for the classic formula
    k = len(rows[0])
    if any(len(r) != k for r in rows) or k < 2:
        return None
    N = len(rows)
    p = {c: 0.0 for c in cats}
    P = []
    for r in rows:
        counts = {c: r.count(c) for c in cats}
        for c in cats:
            p[c] += counts[c]
        P.append((sum(v * v for v in counts.values()) - k) / (k * (k - 1)))
    for c in cats:
        p[c] /= (N * k)
    Pbar = sum(P) / N
    Pe = sum(v * v for v in p.values())
    return 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("labels.jsonl"))
    args = ap.parse_args()

    results = load_returned(args.in_dir)
    annotators = [r["annotator"] for r in results]
    print(f"loaded {len(results)} annotators: {', '.join(annotators)}\n")

    # long-form table + per-(post,dim) label map
    rows = []
    by_post: dict[str, dict[str, dict]] = defaultdict(dict)  # post_id -> annotator -> {dim:val}
    for r in results:
        a = r["annotator"]
        for pid, lab in r["labels"].items():
            rec = {"post_id": pid, "annotator": a,
                   **{d: lab.get(d) for d in DIMS}}
            rows.append(rec)
            by_post[pid][a] = lab
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")

    overlap = {pid: m for pid, m in by_post.items() if len(m) >= 2}
    print(f"overlap subset (labelled by >=2): {len(overlap)} posts\n")

    # ---- agreement per dimension ----
    print("Cohen's kappa (pairwise) / Fleiss' kappa (all raters), per dimension:")
    for d in DIMS:
        fleiss_rows = []
        for pid, m in overlap.items():
            labs = [m[a].get(d) for a in m if m[a].get(d) is not None]
            if len(labs) >= 2:
                fleiss_rows.append(labs)
        line = f"  {d:5s}"
        for x, y in combinations(annotators, 2):
            pairs = [(by_post[pid][x].get(d), by_post[pid][y].get(d))
                     for pid in overlap
                     if x in by_post[pid] and y in by_post[pid]
                     and by_post[pid][x].get(d) is not None
                     and by_post[pid][y].get(d) is not None]
            k = cohen_kappa(pairs)
            if k is not None:
                line += f"   {x}-{y}={k:+.2f}"
        fk = fleiss_kappa(fleiss_rows)
        if fk is not None:
            line += f"   Fleiss={fk:+.2f}"
        print(line)

    # ---- disagreements to adjudicate ----
    print("\nDisagreements on overlap (to adjudicate):")
    ndis = 0
    for pid, m in overlap.items():
        for d in DIMS:
            vals = {a: m[a].get(d) for a in m if m[a].get(d) is not None}
            if len(set(vals.values())) > 1:
                ndis += 1
                if ndis <= 25:
                    print(f"  {pid}  {d}: " + ", ".join(f"{a}={v}" for a, v in vals.items()))
    print(f"  ... {ndis} dimension-level disagreements total"
          + (" (showing first 25)" if ndis > 25 else ""))

    # ---- image-dependent rate ----
    print("\nImage-dependent flags (text alone insufficient):")
    for r in results:
        flagged = sum(1 for lab in r["labels"].values() if lab.get("needs_image"))
        n = len(r["labels"])
        print(f"  {r['annotator']}: {flagged}/{n} ({100*flagged/n:.0f}%)" if n else f"  {r['annotator']}: 0")

    # ---- speed ----
    print("\nMedian seconds/post per pass:")
    for r in results:
        per = defaultdict(list)
        for t in r.get("timings", []):
            if t.get("ms"):
                per[t["pass"]].append(t["ms"])
        bits = []
        for pass_name in ("gate", "cats"):
            v = sorted(per.get(pass_name, []))
            if v:
                bits.append(f"{pass_name}={v[len(v)//2]/1000:.1f}s")
        print(f"  {r['annotator']}: " + ("  ".join(bits) if bits else "(no timings)"))


if __name__ == "__main__":
    main()
