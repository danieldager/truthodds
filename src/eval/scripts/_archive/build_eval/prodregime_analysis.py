"""Silent-block view + genuineness sampling for the prodregime run.

silentblock — of the shipped-ref bucket-A false claims (no refuting read; miss_decomp.json) and the
  160 all-I true claims, how many GAIN >=1 kept refuting read under each condition, and how many CROSS
  that condition's own nested threshold (a cross on a false claim = a catch; on a true claim = a false
  alarm).

gensample — draw the genuineness frame: for S and E-bridge, up to 30 random NEW kept refuting reads
  (flag 1/2 on a doc whose URL is NOT in the shipped dossier for that claim) on false claims and up to
  30 on true claims, rendered for hand adjudication -> gen_frame.jsonl (+ readable .txt).

genscore — after hand tags are filled into gen_frame.jsonl ("genuine": true/false), compute genuine
  rate and the selectivity ratio (genuine refutes per false claim / per true claim), per arm.

  uv run python -m eval.scripts.build_eval.prodregime_analysis silentblock --run full
  uv run python -m eval.scripts.build_eval.prodregime_analysis gensample --run full
  uv run python -m eval.scripts.build_eval.prodregime_analysis genscore
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval import graded_urn as gu  # noqa: E402
from eval.scripts.build_eval import prodregime_score as ps  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/e1_prodregime"
SP = None  # scratchpad dir (bucketA_false.json / allI_true.json); set from --scratch in main()
E1_DOCS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
CONDS = ["S", "E-plain", "E-bridge", "U-plain", "U-bridge"]


def shipped_dossier() -> dict[str, set]:
    """review_url -> set of doc URLs in the shipped-reference dossier (results-00.jsonl)."""
    d = {}
    for l in E1_DOCS.open():
        r = json.loads(l)
        d[r["review_url"]] = {x.get("url") for x in (r.get("results") or []) if x.get("url")}
    return d


def load_full(run):
    return {json.loads(l)["review_url"]: json.loads(l)
            for l in (OUT_DIR / f"results_{run}.jsonl").open()}


def cond_docs(rec, cond):
    """kept docs contributing to a condition, as (url, direction) after leak control."""
    def rd(d, w):
        return (d.get(f"read_{w}") or {}).get("direction")
    out = []
    if cond == "S":
        for d in rec["arm_S"]["docs"]:
            if ps.kept(d):
                out.append((d["url"], rd(d, "plain")))
    elif cond == "E-plain":
        for d in rec["arm_E"]["docs"]:
            if ps.kept(d):
                out.append((d["url"], rd(d, "plain")))
    elif cond == "E-bridge":
        for d in rec["arm_E"]["docs"]:
            if ps.kept(d):
                out.append((d["url"], rd(d, "bridge")))
    elif cond in ("U-plain", "U-bridge"):
        seen = set()
        ec = "plain" if cond == "U-plain" else "bridge"
        for d, w in [(x, "plain") for x in rec["arm_S"]["docs"]] + \
                    [(x, ec) for x in rec["arm_E"]["docs"]]:
            if d["url"] in seen:
                continue
            seen.add(d["url"])
            if ps.kept(d):
                out.append((d["url"], rd(d, w)))
            if len(seen) >= 20:
                break
    return out


def silentblock(run):
    gu.FLAGS = gu.FLAGS6
    bucketA = set(json.load(open(SP / "bucketA_false.json")))
    allI = set(json.load(open(SP / "allI_true.json")))
    pop = fit_urn.load_population(ps.POP)
    full = load_full(run)
    # per-condition fitted weights + nested threshold from the aligned scorer artefacts
    m = json.loads((OUT_DIR / "prodregime_metrics.json").read_text())["conditions"]

    def scored(cond):
        rows = fit_urn.load_headline(OUT_DIR / f"cond_{cond}.jsonl", population=pop)
        w = {k: m[cond]["weights"][k] for k in gu.FLAGS6}
        thr = m[cond]["threshold"]
        return {r["review_url"]: (sum(w[k] * n for k, n in zip(gu.FLAGS6, gu.flag_counts(r))), r["y"])
                for r in rows}, thr

    print(f"\nSILENT-BLOCK VIEW (run={run})")
    print(f"  bucket-A false (no refuting read under shipped ref): {len(bucketA)}")
    print(f"  all-I true (shipped ref): {len(allI)}\n")
    print(f"{'cond':10} | A-false gain>=1 refute | A-false cross thr | allI-true gain>=1 refute | "
          f"allI-true cross thr (false alarm)")
    for cond in CONDS:
        sc, thr = scored(cond)
        gA = cA = gT = cT = 0
        for cid in bucketA:
            docs = cond_docs(full.get(cid, {"arm_S": {"docs": []}, "arm_E": {"docs": []}}), cond) \
                if cid in full else []
            if any(dr in ("1", "2") for _, dr in docs):
                gA += 1
            if cid in sc and sc[cid][0] <= thr:
                cA += 1
        for cid in allI:
            docs = cond_docs(full[cid], cond) if cid in full else []
            if any(dr in ("1", "2") for _, dr in docs):
                gT += 1
            if cid in sc and sc[cid][0] <= thr:
                cT += 1
        print(f"{cond:10} | {gA:4d}/{len(bucketA)} ({gA/len(bucketA):5.1%}) | "
              f"{cA:4d}/{len(bucketA)} ({cA/len(bucketA):5.1%}) | "
              f"{gT:3d}/{len(allI)} ({gT/len(allI):5.1%}) | "
              f"{cT:3d}/{len(allI)} ({cT/len(allI):5.1%})")


def gensample(run):
    rng = random.Random(20260915)
    doss = shipped_dossier()
    full = load_full(run)
    ver = {cid: r["veracity"] for cid, r in full.items()}
    frame = []
    for arm in ("S", "E-bridge"):
        for side, pred in (("false", lambda v: v <= 2), ("true", lambda v: v >= 4)):
            cands = []
            for cid, rec in full.items():
                if not pred(ver[cid]):
                    continue
                which = "plain" if arm == "S" else "bridge"
                docs = rec["arm_S"]["docs"] if arm == "S" else rec["arm_E"]["docs"]
                for d in docs:
                    if not ps.kept(d):
                        continue
                    rd = (d.get(f"read_{which}") or {}).get("direction")
                    if rd in ("1", "2") and d["url"] not in doss.get(cid, set()):
                        cands.append((cid, d, which, rd))
            rng.shuffle(cands)
            for cid, d, which, rd in cands[:30]:
                rec = full[cid]
                frame.append({
                    "arm": arm, "side": side, "review_url": cid,
                    "veracity": ver[cid], "claim": rec["claim"],
                    "target": rec["arm_E"].get("target") if arm == "E-bridge" else None,
                    "bearing": rec["arm_E"].get("bearing") if arm == "E-bridge" else None,
                    "url": d["url"], "domain": d["domain"], "date": d.get("date"),
                    "flag": rd, "reason": (d.get(f"read_{which}") or {}).get("reason"),
                    "sents": " ".join(d.get("sents") or [])[:1400],
                    "genuine": None})   # <- hand-fill true/false
    (OUT_DIR / "gen_frame.jsonl").write_text("\n".join(json.dumps(f) for f in frame))
    with (OUT_DIR / "gen_frame.txt").open("w") as fh:
        for i, f in enumerate(frame):
            fh.write(f"\n{'='*100}\n[{i}] arm={f['arm']} side={f['side']} v{f['veracity']} flag={f['flag']}\n")
            fh.write(f"CLAIM: {f['claim']}\n")
            if f["target"]:
                fh.write(f"TARGET: {f['target']}\nBEARING: {f['bearing']}\n")
            fh.write(f"DOC: {f['domain']} ({f['date']}) {f['url']}\n")
            fh.write(f"READ reason: {f['reason']}\n")
            fh.write(f"SENTS: {f['sents']}\n")
    from collections import Counter
    c = Counter((f["arm"], f["side"]) for f in frame)
    print(f"gen_frame: {len(frame)} reads -> {OUT_DIR/'gen_frame.jsonl'} (+ .txt)")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")


def genscore():
    frame = [json.loads(l) for l in (OUT_DIR / "gen_frame.jsonl").read_text().splitlines() if l.strip()]
    full = load_full("full")
    ver = {cid: r["veracity"] for cid, r in full.items()}
    nF = sum(1 for v in ver.values() if v <= 2)
    nT = sum(1 for v in ver.values() if v >= 4)
    from collections import Counter, defaultdict
    tag = defaultdict(lambda: [0, 0])   # (arm,side) -> [genuine, total]
    for f in frame:
        if f["genuine"] is None:
            continue
        t = tag[(f["arm"], f["side"])]
        t[1] += 1
        t[0] += 1 if f["genuine"] else 0
    print("\nGENUINENESS (hand-adjudicated NEW refuting reads):")
    for arm in ("S", "E-bridge"):
        gf = tag[(arm, "false")]; gt = tag[(arm, "true")]
        rf = gf[0] / gf[1] if gf[1] else float("nan")
        rt = gt[0] / gt[1] if gt[1] else float("nan")
        print(f"  {arm}: false genuine {gf[0]}/{gf[1]} ({rf:.0%}) | true genuine {gt[0]}/{gt[1]} ({rt:.0%})")
        # selectivity: genuine-refute rate per claim (extrapolated from sample rates x observed refute/claim)
    print("\n(selectivity computed in the final report from these rates x observed refutes/claim)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["silentblock", "gensample", "genscore"])
    ap.add_argument("--run", default="full")
    ap.add_argument("--scratch", help="scratchpad dir holding bucketA_false.json / allI_true.json")
    a = ap.parse_args()
    global SP
    if a.scratch:
        SP = Path(a.scratch)
    if a.cmd == "silentblock":
        silentblock(a.run)
    elif a.cmd == "gensample":
        gensample(a.run)
    else:
        genscore()


if __name__ == "__main__":
    main()
