"""Stronger-reader re-read scores: Flash-v5, Flash-v5b/v5c, V4-Pro-v5, Kimi-v5/v5b.
Faithful extension of the Arm-S scorer to the requestion re-read manifest.
 T1 fix/break on 120 judged, per cell: right-by-blind, corrected/broken vs Flash-v5, NET.
 T2 flip rates on the 200+200 sample per cell (all cols, identical docs = the 520 sub).
 T3 flag distribution (on the 520 sub for all cols; on the full 4556 for the full-coverage cols).
 Frozen instrument: honest population-wide swap-in (actual flag where the model READ the doc,
 per-cell rate-based draw for occurrences it did not read incl. the non-manifest agreeing
 reads), FROZEN training weights, both regimes, 200 draws; + baseline.

  uv run python -m eval.scripts.build_eval.stronger_reader_scores

$0 -- rescores cached requestion re-reads, no API calls. (The v5c Kimi re-read was never
run; the column is skipped automatically.)"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eval.scripts.build_eval import fit_urn, graded_urn
from eval.scripts.build_eval.e1_figures import _auc_np
from eval.scripts.build_eval import reader_error_lib as L

FLAGS7 = graded_urn.FLAGS7            # ("5","4","3","X","I","2","1")
FIDX = {f: i for i, f in enumerate(FLAGS7)}
K = fit_urn.K_FOLDS
PAD_TO = fit_urn.PAD_TO
WEIGHTS = "eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json"
TRAIN = "eval/data/urn_runs/e1_ctx/results-00.jsonl"
PROD = "eval/data/urn_runs/e1_prodregime/cond_S_v5_full3000_v5.jsonl"
REQUESTION = "eval/data/reader_lab/requestion"
IDMAP = "eval/data/reader_lab/requestion/rq_idmap.json"
CLAIMS_SUB = "eval/data/reader_lab/requestion/rq_claims_sub.json"
POPULATION = "eval/data/populations/fc_gold_bal3000.parquet"
OUT = "eval/data/reader_lab/requestion/stronger_reader_scores.json"
NDRAW, SEED = 200, 920
DESIRED = ["flash", "flash_v5b", "flash_v5c", "v4pro", "kimi", "kimi_v5b", "kimi_v5c"]
COVER = {"flash": "full", "flash_v5b": "full", "flash_v5c": "full", "v4pro": "full",
         "kimi": "sub", "kimi_v5b": "sub", "kimi_v5c": "sub"}
CELLW = {"Q-refute-on-true": "refute", "Q-support-on-false": "support",
         "C-refute-on-false": "refute", "C-support-on-true": "support"}
AUDIT_CELL = {"R-T": "Q-refute-on-true", "S-F": "Q-support-on-false",
              "R-F": "C-refute-on-false", "S-T": "C-support-on-true"}
AGREE = {0: ("1", "2"), 1: ("5", "4")}
BASE_REF = {"training": {"auc": 0.860, "recall": 0.382, "false_flagged": 573, "true_flagged": 31},
            "production": {"auc": 0.927, "recall": 0.372, "false_flagged": 558, "true_flagged": 30}}

# Populated in main() (no file reads at import time, per the build_eval convention).
idmap, FLAGOF, COV, VBY, RATES = {}, {}, {}, {}, {}
FULL, SUB, JUDGED, SAMPLE, COMMON_SUB = set(), set(), set(), set(), set()
MODEL_ORDER = []
WV = None
bal = None


def direction(fl):
    if fl in ("1", "2"):
        return "refute"
    if fl in ("4", "5"):
        return "support"
    return "neither"


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_model(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m[r["claim_id"]] = r["new_flags"][0]
    return m


def dircell(m, cid):
    return direction(FLAGOF[m][cid])


def blinddir(cid):
    return direction(idmap[cid]["hand"]["blind_flag"])


def flashdir(cid):
    return direction(idmap[cid]["flag"])


# =================== TASK 1: 120 judged, per cell ===================
# right_by_blind = model direction == blind direction; corrected/broken measured against Flash-v5:
# corrected = flash-v5 disagreed with blind AND model agrees; broken = flash-v5 agreed AND model does not.
def t1_block(cids):
    res = {}
    for m in MODEL_ORDER:
        mc = [c for c in cids if c in COV[m]]
        rb = corr = brk = 0
        for cid in mc:
            bd = blinddir(cid); md = dircell(m, cid); fd = flashdir(cid)
            if md == bd:
                rb += 1
            if fd != bd and md == bd:
                corr += 1
            if fd == bd and md != bd:
                brk += 1
        n = len(mc)
        res[m] = {"n": n, "right_by_blind": rb, "right_rate": round(rb / n, 4) if n else None,
                  "wilson_right": wilson(rb, n), "corrected_vs_flashv5": corr,
                  "broken_vs_flashv5": brk, "net_vs_flashv5": corr - brk}
    return res


# =================== TASK 2: flip rates on 200+200 sample per cell ===================
def t2_block(cids, cell):
    wrong = CELLW[cell]; res = {}
    for m in MODEL_ORDER:
        mc = [c for c in cids if c in COV[m]]
        flipped = sum(1 for c in mc if dircell(m, c) != wrong)
        neither = sum(1 for c in mc if dircell(m, c) == "neither")
        opp = sum(1 for c in mc if dircell(m, c) not in (wrong, "neither"))
        n = len(mc)
        res[m] = {"n": n, "flipped": flipped, "flip_rate": round(flipped / n, 4) if n else None,
                  "wilson": wilson(flipped, n), "to_neither": neither, "to_opposite": opp}
    return res


# =================== TASK 3: flag distribution ===================
def dist(cidset, m):
    cc = [c for c in cidset if c in COV[m]]
    d = collections.Counter(FLAGOF[m][c] for c in cc)
    n = len(cc)
    return {"counts": {f: d[f] for f in FLAGS7}, "frac": {f: round(d[f] / n, 4) for f in FLAGS7}}


# =================== frozen instrument core ===================
def frozen_metrics(C, y, fold):
    s = WV @ C
    auc = float(_auc_np(s[y == 1], s[y == 0]))
    flagged = np.zeros(len(y), bool); thrs = []
    for k in range(K):
        te = fold == k
        if not te.any() or te.all():
            continue
        _, _, thr = fit_urn.recall_at_fpr(list(zip(s[~te].tolist(), y[~te].tolist())), 0.02)
        flagged[te] = s[te] <= thr; thrs.append(float(thr))
    return {"auc": round(auc, 4), "recall": round(float(flagged[y == 0].mean()), 4),
            "false_flagged": int(flagged[y == 0].sum()), "true_flagged": int(flagged[y == 1].sum()),
            "threshold": round(float(np.mean(thrs)), 4)}


def base_matrix(dr):
    nn = len(dr); C = np.zeros((nn, 7)); y = np.zeros(nn, int); fold = np.zeros(nn, int)
    for i, d in enumerate(dr):
        y[i] = d["y"]; fold[i] = d["fold"]
        cnt = collections.Counter(d["real_flags"]); cnt["I"] += max(0, PAD_TO - len(d["real_flags"]))
        for f, c in cnt.items():
            C[i, FIDX[f]] += c
    return C.T, y, fold


# per-cell fix/break RATES + replacement dists, from each model's own coverage pool
def occ_cell(fl, y):
    d = direction(fl)
    if d == "neither":
        return None
    if y == 0:   # FALSE claim
        return "C-refute-on-false" if d == "refute" else "Q-support-on-false"
    return "C-support-on-true" if d == "support" else "Q-refute-on-true"     # TRUE claim


def cell_rate(m, cid_pool, cell):
    wrong = CELLW[cell]
    flipped = [cid for cid in cid_pool if dircell(m, cid) != wrong]
    repl = collections.Counter(FLAGOF[m][cid] for cid in flipped)
    tot = sum(repl.values())
    flags = list(repl); probs = [repl[f] / tot for f in flags] if tot else []
    return {"n": len(cid_pool), "n_flip": len(flipped),
            "p": (len(flipped) / len(cid_pool) if cid_pool else 0.0),
            "repl_flags": flags, "repl_probs": probs,
            "repl_dist": {f: round(repl[f] / tot, 4) for f in flags} if tot else {}}


# honest population-wide: actual flag where the model READ the (claim,regime) doc; rate-based draw
# for occurrences the model did not read (incl. non-manifest agreeing reads). Within-claim
# correlation preserved for the read docs.
def build_vby(m):
    vby = collections.defaultdict(list)
    read = COV[m]
    for cid, mm in idmap.items():
        if cid not in read:
            continue
        for reg in mm["regimes"]:
            vby[(mm["claim_id"], reg)].append((mm["flags_by_regime"][reg], FLAGOF[m][cid]))
    return vby


def honest_matrix(dr, reg, m, rng):
    vby = VBY[m]; rates = RATES[m]
    nn = len(dr); C = np.zeros((nn, 7)); y = np.zeros(nn, int); fold = np.zeros(nn, int)
    n_actual = n_rate = 0
    for i, d in enumerate(dr):
        y[i] = d["y"]; fold[i] = d["fold"]; yy = d["y"]
        flags = collections.Counter(d["real_flags"])
        subs = vby.get((d["review_url"], reg), [])
        removed = collections.Counter(); adds = []
        for fl, nf in subs:
            if flags[fl] - removed[fl] > 0:
                removed[fl] += 1; adds.append(nf); n_actual += 1
        post = flags - removed
        base = collections.Counter()
        for nf in adds:
            base[nf] += 1
        for fl, cnt in post.items():
            for _ in range(cnt):
                cell = occ_cell(fl, yy); nf = fl
                if cell is not None:
                    r = rates[cell]
                    if r["p"] > 0 and r["repl_probs"] and rng.random() < r["p"]:
                        nf = r["repl_flags"][rng.choice(len(r["repl_flags"]), p=r["repl_probs"])]
                    n_rate += 1
                base[nf] += 1
        base["I"] += max(0, PAD_TO - len(d["real_flags"]))
        for ff, cc in base.items():
            if cc > 0:
                C[i, FIDX[ff]] += cc
    return C.T, y, fold, n_actual, n_rate


def main():
    global idmap, FLAGOF, COV, VBY, RATES, FULL, SUB, JUDGED, SAMPLE, COMMON_SUB, MODEL_ORDER, WV, bal
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--train", type=Path, default=TRAIN)
    ap.add_argument("--prod", type=Path, default=PROD)
    ap.add_argument("--requestion", type=Path, default=REQUESTION)
    ap.add_argument("--idmap", type=Path, default=IDMAP)
    ap.add_argument("--claims-sub", type=Path, default=CLAIMS_SUB)
    ap.add_argument("--population", type=Path, default=POPULATION)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting the pinned frozen deliverable")
    args = ap.parse_args()
    if Path(args.out).resolve() == Path(OUT).resolve() and not args.force:
        raise SystemExit(f"refusing to overwrite the pinned frozen deliverable {OUT}: run with "
                         f"--out PATH to reproduce the recalls and write elsewhere, or --force "
                         f"to regenerate it in place.")

    WV = np.array([json.loads(Path(args.weights).read_text())["weights"][f] for f in FLAGS7])
    rq = str(args.requestion)
    paths = {"flash_v5b": f"{rq}/v5b__DeepSeek-V4-Flash.jsonl",
             "flash_v5c": f"{rq}/v5c__DeepSeek-V4-Flash.jsonl",
             "v4pro":     f"{rq}/v5__DeepSeek-V4-Pro.jsonl",
             "kimi":      f"{rq}/v5__Kimi-K2.6_sub.jsonl",
             "kimi_v5b":  f"{rq}/v5b__Kimi-K2.6_sub.jsonl",
             "kimi_v5c":  f"{rq}/v5c__Kimi-K2.6_sub.jsonl"}

    idmap = json.load(open(args.idmap))
    FULL = set(idmap)                                     # 4556
    SUB = set(c["claim_id"] for c in json.load(open(args.claims_sub)))   # canonical 520
    FLAGOF = {"flash": {cid: idmap[cid]["flag"] for cid in idmap}}
    for m, p in paths.items():
        if Path(p).exists():
            FLAGOF[m] = load_model(p)
        else:
            print(f"SKIP {m}: {p} missing")
    MODEL_ORDER = [m for m in DESIRED if m in FLAGOF]
    # The v5 Kimi _sub file was later appended with the paused extension (888 recs) and is missing
    # 20 of the 520 originals; restrict its coverage to the sub docs it actually holds.
    FLAGOF["kimi"] = {c: v for c, v in FLAGOF["kimi"].items() if c in SUB}
    # per-model coverage set (docs the column actually has a flag for)
    for m in MODEL_ORDER:
        COV[m] = FULL if COVER[m] == "full" else (set(FLAGOF[m]) & SUB)
    for m in MODEL_ORDER:
        if COVER[m] == "full":
            assert FULL == set(FLAGOF[m]), f"{m} not full coverage"
    JUDGED = {c for c in FULL if idmap[c]["hand_judged"]}
    SAMPLE = SUB - JUDGED
    assert len(JUDGED) == 120, len(JUDGED)
    print(f"FULL {len(FULL)}  SUB {len(SUB)}  judged {len(JUDGED)}  sample {len(SAMPLE)}")
    print("coverage: " + "  ".join(f"{m}={len(COV[m])}" for m in MODEL_ORDER))
    # common sub set across all columns (for the strict identical-doc comparison)
    COMMON_SUB = SUB & COV["kimi"] if "kimi" in COV else SUB
    print(f"COMMON_SUB (all columns present) = {len(COMMON_SUB)}")

    out = {"n_full": len(FULL), "n_sub": len(SUB), "n_judged": len(JUDGED), "n_sample": len(SAMPLE),
           "weights_file": str(args.weights), "seed": SEED, "ndraws": NDRAW, "models": MODEL_ORDER}

    JC = {ac: [c for c in JUDGED if idmap[c]["hand"]["audit_cell"] == ac] for ac in AUDIT_CELL}
    out["task1_judged"] = {ac: t1_block(JC[ac]) for ac in AUDIT_CELL}
    out["task1_judged"]["questionable"] = t1_block(JC["R-T"] + JC["S-F"])
    out["task1_judged"]["control"] = t1_block(JC["R-F"] + JC["S-T"])
    out["task1_judged"]["ALL"] = t1_block(sorted(JUDGED))

    out["task2_sample"] = {}
    for cell in CELLW:
        cids = [c for c in SAMPLE if idmap[c]["cell"] == cell]
        out["task2_sample"][cell] = t2_block(cids, cell)

    out["task3_flag_distribution"] = {
        "on_sub_520": {"n": len(SUB), **{m: dist(SUB, m) for m in MODEL_ORDER}},
        "on_full_4556": {"n": len(FULL),
                         **{m: dist(FULL, m) for m in MODEL_ORDER if COVER[m] == "full"}},
    }

    for m in MODEL_ORDER:
        pool_ids = COV[m]
        RATES[m] = {cell: cell_rate(m, [c for c in pool_ids if idmap[c]["cell"] == cell], cell)
                    for cell in CELLW}
    out["frozen_rates"] = {m: {cell: {kk: RATES[m][cell][kk] for kk in ("n", "n_flip", "p", "repl_dist")}
                               for cell in CELLW} for m in MODEL_ORDER}

    for m in MODEL_ORDER:
        VBY[m] = build_vby(m)

    bal = set(pd.read_parquet(args.population)["claim_id"].tolist())
    out["frozen"] = {"baseline": {}}
    for m in MODEL_ORDER:
        out["frozen"][m] = {}
    for label, path, reg in [("training", args.train, "train"), ("production", args.prod, "prod")]:
        dr = L.load_docs(Path(path), bal)
        Cb, yb, fb = base_matrix(dr)
        out["frozen"]["baseline"][label] = {"n": len(dr), "ref": BASE_REF[label],
                                            **frozen_metrics(Cb, yb, fb)}
        print(f"{label} baseline {frozen_metrics(Cb, yb, fb)}")
        for m in MODEL_ORDER:
            acc = {kk: [] for kk in ("auc", "recall", "false_flagged", "true_flagged")}
            rng = np.random.default_rng(SEED); na = nr = 0
            for _ in range(NDRAW):
                C, y2, f2, na, nr = honest_matrix(dr, reg, m, rng)
                fm = frozen_metrics(C, y2, f2)
                for kk in acc:
                    acc[kk].append(fm[kk])
            res = {kk: {"mean": round(float(np.mean(acc[kk])), 4),
                        "p2.5": round(float(np.percentile(acc[kk], 2.5)), 4),
                        "p97.5": round(float(np.percentile(acc[kk], 97.5)), 4)} for kk in acc}
            res["n_actual_inject"] = na; res["n_rate_draw"] = nr
            out["frozen"][m][label] = res
            print(f"  {label} {m}: recall {res['recall']['mean']} AUC {res['auc']['mean']} "
                  f"F{res['false_flagged']['mean']:.0f} T{res['true_flagged']['mean']:.0f} "
                  f"(actual {na} rate {nr})")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
