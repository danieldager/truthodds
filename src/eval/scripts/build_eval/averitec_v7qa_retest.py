"""Issue #28 re-test: the AVeriTeC dev 7-flag urn AUC under read-v5 vs v7qa flags.

The pinned issue-28 number (7-flag AUC 0.868, binary map A, n=500 gold rows) is a
TRANSFER: the frozen fc-gold ladder weights (`e1_ctx/model_ladder/ladder.json`,
read-v5 reads) applied unchanged to the AVeriTeC run, scored exactly as
`score_frozen_urn.py` does -- sum(count x weight), NO pad-to-10 -- and ranked
against gold Supported = pass. This script reproduces that to the digit and then
re-runs it on the v7qa re-read of the same stored documents
(`reader_lab/refit_v7qa/averitec/`), under three weight regimes:

  (a) frozen read-v5 ladder weights on read-v5 flags       -> must give 0.8682
  (b) the v7qa fc-gold refit weights on v7qa flags          (transfer, the fair
      reader-consistent comparison)
  (c) a 7-flag fit on AVeriTeC ITSELF, in-sample and 5-fold oof, both readers
      (the ceiling either reader could reach on this dataset)

Recall at the FPR of the claim-level verifier arms (all10 0.385, top3 0.475) is
reported beside each AUC, same convention as `compare_averitec.py`.

    uv run python -m eval.scripts.build_eval.averitec_v7qa_retest
        [--reads v7qa=<jsonl> read-v5@gpt-oss=<jsonl> ...]

Each `--reads label=path` is a reader_lab re-read of the SAME stored documents; every
one gets the (a)/(c) rows and a paired dAUC against the pinned read-v5@Flash column,
so several readers can be compared in one table. Default is the v7qa re-read alone.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

FLAGS = ("5", "4", "3", "X", "I", "2", "1")
GOLD = Path("eval/scripts/verification_grading/data/claims_dev_500_gold.parquet")
RUN = Path("eval/data/urn_runs/averitec_dev/scores.jsonl")
READS = Path("eval/data/reader_lab/refit_v7qa/averitec/v7qa__gpt-oss-120b+rlow.jsonl")
LADDER = Path("eval/data/urn_runs/e1_ctx/model_ladder/ladder.json")
REFIT_V7QA = Path("eval/data/populations/refit_results_v7qa.json")
ARM_FPR = {"all10": 0.38524590163934425, "top3": 0.47540983606557374}
BOOT_REPS, BOOT_SEED = 2000, 707
K_FOLDS = 5


def counts(sub: dict | None) -> dict[str, collections.Counter]:
    """claim_id -> flag counts; `sub` maps claim_id -> {rank: v7qa flag}."""
    out = {}
    for line in RUN.open():
        r = json.loads(line)
        cid, c = r["review_url"], collections.Counter()
        for d in r.get("results") or []:
            f = (d.get("read") or {}).get("direction")
            if f is None:
                continue
            if sub is not None:
                f = sub.get(cid, {}).get(d["rank"], f)   # no stored region -> source flag
            if f in FLAGS:
                c[f] += 1
        out[cid] = c
    return out


def score(c, w, pad):
    s = sum(c.get(f, 0) * w[f] for f in FLAGS)
    return s + (max(0, pad - sum(c.values())) * w["I"] if pad else 0.0)


def auc(s, y):
    """Mann-Whitney, positives = gold pass (compare_averitec.auc_pass)."""
    pos, neg = y.sum(), (~y).sum()
    o = np.argsort(s, kind="mergesort")
    ranks, ss, i = np.empty(len(s), float), s[o], 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def threshold_at(s, y, target):
    """Largest threshold whose FPR (flag = s <= thr) stays at or below target."""
    ok = [float(t) for t in np.unique(s) if ((s <= t) & y).sum() / max(1, y.sum()) <= target]
    return max(ok) if ok else float("-inf")


def boot_ci(s, y, reps=BOOT_REPS, seed=BOOT_SEED):
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    v = np.array([auc(s[r], y[r]) for r in idx])
    return np.percentile(v[np.isfinite(v)], [2.5, 97.5])


def fit_graded(cnts, y, pad):
    """graded_urn.fit_graded: per-flag log-LR, Laplace +1 / +7."""
    t, f = collections.Counter(), collections.Counter()
    for c, yy in zip(cnts, y):
        d = {k: c.get(k, 0) for k in FLAGS}
        if pad:
            d["I"] += max(0, pad - sum(c.values()))
        (t if yy else f).update(d)
    tt, tf = sum(t.values()), sum(f.values())
    return {k: math.log(((t[k] + 1) / (tt + 7)) / ((f[k] + 1) / (tf + 7))) for k in FLAGS}


def fold_of(cid):
    return int(hashlib.blake2b(str(cid).encode(), digest_size=8).hexdigest(), 16) % K_FOLDS


def row(name, s, y):
    a = auc(s, y)
    lo, hi = boot_ci(s, y)
    line = f"{name:<50s} AUC {a:.4f} [{lo:.4f}, {hi:.4f}]"
    for arm, tgt in ARM_FPR.items():
        flag = s <= threshold_at(s, y, tgt)
        line += (f" | recall@{arm} FPR {tgt:.3f}: {(flag & ~y).sum() / (~y).sum():.4f}"
                 f" (fpr {(flag & y).sum() / y.sum():.4f})")
    print(line, flush=True)
    return {"auc": a, "auc_ci95": [lo, hi]}


def load_sub(path: Path):
    sub, n_docs = {}, 0
    for line in Path(path).open():
        r = json.loads(line)
        sub[r["claim_id"]] = {d["rank"]: d["new"] for d in r["docs"]}
        n_docs += len(r["docs"])
    return sub, n_docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", nargs="*", default=[f"v7qa={READS}"],
                    help="label=path reader_lab re-reads of the same stored documents")
    ap.add_argument("--out", type=Path, default=RUN.parent / "averitec_v7qa_retest_metrics.json",
                    help="where to persist the run's AUCs (JSON, next to the AVeriTeC run)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing metrics file")
    a_ = ap.parse_args()
    metrics = {"generated": "averitec_v7qa_retest.py", "gold": str(GOLD),
               "reads": {}, "transfer": {}, "fitted_on_averitec": {}}
    w_v5 = json.loads(LADDER.read_text())["models"]["7-flag"]["weights"]
    w_v7 = next(c for c in json.loads(REFIT_V7QA.read_text())["cells"]
                if c["variant"] == "graded_urn_7flag")["weights"]
    sets = []
    for spec in a_.reads:
        label, path = spec.split("=", 1)
        sub, n_docs = load_sub(Path(path))
        sets.append((label, counts(sub), sub, n_docs))
    c5 = counts(None)
    g = pd.read_parquet(GOLD)
    g["claim_id"] = g["claim_id"].astype(str)
    ids, y = list(g["claim_id"]), (g["gold_label"] == "Supported").to_numpy()
    print(f"gold rows {len(g)} (Supported {y.sum()} / rest {(~y).sum()})")
    for label, _, sub, n_docs in sets:
        print(f"  {label} re-read: {len(sub)} claims, {n_docs} docs")
    t5 = sum(c5.values(), collections.Counter())
    print("flag totals    read-v5@Flash  " + "  ".join(f"{f}: {t5[f]}" for f in FLAGS))
    for label, cnt, _, _ in sets:
        t = sum(cnt.values(), collections.Counter())
        print(f"flag totals    {label:<14s} " + "  ".join(f"{f}: {t[f]}" for f in FLAGS))

    def vec(cnt, w, pad=0):
        return np.array([score(cnt.get(i, collections.Counter()), w, pad) for i in ids])

    print("\n(a) read-v5@Flash flags, frozen fc-gold ladder weights (the pinned procedure)")
    a = vec(c5, w_v5)
    metrics["transfer"]["readv5_flash_frozen_ladder_no_pad"] = row("transfer, no pad", a, y)

    idx = np.random.default_rng(BOOT_SEED).integers(0, len(y), size=(BOOT_REPS, len(y)))

    def paired(name, b):
        v = np.array([auc(a[r], y[r]) - auc(b[r], y[r]) for r in idx])
        print(f"  paired dAUC (read-v5@Flash - {name}) {auc(a, y) - auc(b, y):+.4f} 95% CI "
              f"[{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]")

    for label, cnt, _, _ in sets:
        print(f"\n(b) {label} flags under transferred weights")
        if label == "v7qa":
            b = vec(cnt, w_v7, 10)
            metrics["transfer"]["v7qa_fcgold_refit_pad10"] = row(
                "v7qa fc-gold refit weights, pad-to-10 (reader-consistent)", b, y)
            row("v7qa fc-gold refit weights, no pad", vec(cnt, w_v7), y)
        else:
            b = vec(cnt, w_v5)
            metrics["transfer"][f"{label}_frozen_ladder_no_pad"] = row(
                "frozen ladder weights, no pad (same prompt as the ladder)", b, y)
        row("under the OLD ladder weights, no pad (control)", vec(cnt, w_v5), y)
        paired(label, b)

    print("\n(c) 7-flag fit on AVeriTeC itself")
    folds = np.array([fold_of(i) for i in ids])
    for tag, cnt in [("read-v5@Flash", c5)] + [(l, c) for l, c, _, _ in sets]:
        for pad in (0, 10):
            cl = [cnt.get(i, collections.Counter()) for i in ids]
            w_in = fit_graded(cl, y, pad)
            row(f"{tag} in-sample (pad {pad})", vec(cnt, w_in, pad), y)
            s = np.zeros(len(ids))
            for k in range(K_FOLDS):
                wk = fit_graded([c for c, f in zip(cl, folds) if f != k], y[folds != k], pad)
                s[folds == k] = [score(c, wk, pad) for c, f in zip(cl, folds) if f == k]
            oof = row(f"{tag} 5-fold oof (pad {pad})", s, y)
            metrics["fitted_on_averitec"][f"{tag}_oof_pad{pad}"] = oof
            print("    in-sample weights " + " ".join(f"{k}{w_in[k]:+.3f}" for k in FLAGS))

    metrics["reads"] = {label: {"claims": len(sub), "docs": n_docs}
                        for label, _, sub, n_docs in sets}
    metrics["gold_rows"] = int(len(g))
    metrics["n_supported"] = int(y.sum())
    # gate: the four pinned issue-28 AUCs must reproduce
    PINS = {"transfer.readv5_flash_frozen_ladder_no_pad": 0.8682,
            "transfer.v7qa_fcgold_refit_pad10": 0.8039,
            "fitted_on_averitec.read-v5@Flash_oof_pad0": 0.8606,
            "fitted_on_averitec.v7qa_oof_pad10": 0.8092}
    ok = True
    for dotted, pin in PINS.items():
        sec, key = dotted.split(".", 1)
        got = metrics[sec].get(key, {}).get("auc")
        hit = got is not None and abs(got - pin) < 5e-5
        ok = ok and hit
        print(f"PIN {dotted}: got {got} vs {pin} -> {'OK' if hit else 'MISS/absent'}")
    metrics["pins_reproduced"] = ok

    if a_.out.exists() and not a_.force:
        print(f"\n(not writing: {a_.out} exists; pass --force to overwrite)")
    else:
        a_.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {a_.out}")


if __name__ == "__main__":
    main()
