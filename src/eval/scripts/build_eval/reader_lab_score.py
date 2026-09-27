"""Score reader-lab variants against document-level gold labels.

    uv run python -m eval.scripts.build_eval.reader_lab_score --gold <dir with labels_*.json> \
        --lab eval/data/reader_lab/outlet eval/data/reader_lab/timeline

Gold: labels_*.json keyed by pair_id "<source>|<claim_id>|<rank>" with {"direction": flag}.
Each lab jsonl (<prompt>__<model>.jsonl) carries per document old (production) and new
flags, so the production reader is scored as the variant "prod" from any one file.
Reports, per variant: exact flag agreement, polarity agreement (for = 5/4, none = 3/X/I,
against = 2/1), and the two polarity inversions that matter, gold-for read as against and
gold-against read as for.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

POL = {"5": "for", "4": "for", "3": "none", "X": "none", "I": "none", "2": "against", "1": "against"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--lab", nargs="+", required=True)
    ap.add_argument("--by-group", action="store_true")
    a = ap.parse_args()
    gold = {}
    for f in glob.glob(str(Path(a.gold) / "labels_*.json")):
        for k, v in json.load(open(f)).items():
            gold[k] = v["direction"]
    print(f"gold pairs {len(gold)}  flags {dict(Counter(gold.values()))}")

    preds = defaultdict(dict)   # variant -> pair_id -> flag
    groups = {}
    for labdir in a.lab:
        src = Path(labdir).name
        for f in sorted(glob.glob(str(Path(labdir) / "*.jsonl"))):
            var = Path(f).stem
            for line in open(f):
                r = json.loads(line)
                for d in r["docs"]:
                    pid = f"{src}|{r['claim_id']}|{d['rank']}"
                    if pid not in gold:
                        continue
                    preds[var][pid] = d["new"]
                    preds["prod"][pid] = d["old"]
                    groups[pid] = r.get("group")

    def report(name, sel):
        rows = []
        for var in sorted(preds, key=lambda v: (v != "prod", v)):
            ps = {k: v for k, v in preds[var].items() if k in sel}
            if not ps:
                continue
            n = len(ps)
            exact = sum(gold[k] == v for k, v in ps.items()) / n
            pol = sum(POL[gold[k]] == POL[v] for k, v in ps.items()) / n
            gf = [k for k in ps if POL[gold[k]] == "for"]
            ga = [k for k in ps if POL[gold[k]] == "against"]
            f2a = sum(POL[ps[k]] == "against" for k in gf) / max(1, len(gf))
            a2f = sum(POL[ps[k]] == "for" for k in ga) / max(1, len(ga))
            a_rec = sum(POL[ps[k]] == "against" for k in ga) / max(1, len(ga))
            f_rec = sum(POL[ps[k]] == "for" for k in gf) / max(1, len(gf))
            rows.append((var, n, exact, pol, len(gf), f2a, f_rec, len(ga), a2f, a_rec))
        print(f"\n{name}")
        print(f"{'variant':34s} {'n':>5} {'exact':>6} {'polar':>6} | {'gold for':>8} {'read agnst':>10} {'read for':>8} | {'gold agnst':>10} {'read for':>8} {'read agnst':>10}")
        for var, n, exact, pol, nf, f2a, frec, na, a2f, arec in rows:
            print(f"{var:34s} {n:5d} {exact:6.3f} {pol:6.3f} | {nf:8d} {f2a:10.3f} {frec:8.3f} | {na:10d} {a2f:8.3f} {arec:10.3f}")

    report("ALL GOLD PAIRS", set(gold))
    if a.by_group:
        for g in sorted({v for v in groups.values()}, key=str):
            report(f"group {g}", {k for k, v in groups.items() if v == g})


if __name__ == "__main__":
    main()
