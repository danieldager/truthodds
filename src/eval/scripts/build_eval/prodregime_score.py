"""Score the prodregime run: six conditions, silent-block view, genuineness sampling frames.

Conditions (all six-flag, on fc_gold_rep1500, media-axis excluded, padded to PAD_TO, cluster folds):
  shipped   — the shipped-with-ceiling reference: existing read-v6.1 frozen3280 reads (reproduces
              headline_metrics.json AUC 0.8512 / recall@2%FPR 41.5% / thr -3.5839).
  S         — production Serper query, ceiling OFF, read-v6.1 plain.
  E-plain   — Exa two-hop retrieval, read-v6.1 plain.
  E-bridge  — Exa two-hop retrieval, read-v6.1 with the target/bearing bridge.
  U-plain   — S ∪ E deduped by URL (<=20), all read plain.
  U-bridge  — S ∪ E deduped by URL (<=20), S read plain, E read with the bridge.

Leak control on S/E/U: drop fc_domain docs, drop echo-yes docs. Reuses fit_urn.load_headline for
padding / media exclusion / clusters / folds, and fit_urn's vectorised fit_np/oof_np/nested_threshold
for the point estimates + a paired clustered bootstrap (dAUC vs S and vs shipped, 2000 reps).

  uv run python -m eval.scripts.build_eval.prodregime_score --run full

Writes eval/data/urn_runs/e1_prodregime/{cond_<c>.jsonl, prodregime_metrics.json}. Nothing shipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval import graded_urn as gu  # noqa: E402
from eval.scripts.build_eval.e1_figures import _auc_np  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/e1_prodregime"
POP = SRC / "eval/data/populations/fc_gold_rep1500.parquet"
FROZEN = SRC / "eval/data/urn_runs/e1_ctx_v61/results-v6.1-frozen3280.jsonl"
DIRS = {"5", "4", "3", "X", "I", "2", "1"}
CONDS = ["shipped", "S", "E-plain", "E-bridge", "U-plain", "U-bridge"]


def kept(d: dict) -> bool:
    return d.get("read_status") == "prepped" and d.get("echo") != "yes"


def cond_dirs(rec: dict, cond: str) -> list[str]:
    """The read flags contributed by a condition for one claim, after leak control."""
    def rd(d, which):
        r = d.get(f"read_{which}") or {}
        return r.get("direction")
    out = []
    if cond == "S":
        for d in rec["arm_S"]["docs"]:
            if kept(d) and rd(d, "plain") in DIRS:
                out.append(rd(d, "plain"))
    elif cond == "E-plain":
        for d in rec["arm_E"]["docs"]:
            if kept(d) and rd(d, "plain") in DIRS:
                out.append(rd(d, "plain"))
    elif cond == "E-bridge":
        for d in rec["arm_E"]["docs"]:
            if kept(d) and rd(d, "bridge") in DIRS:
                out.append(rd(d, "bridge"))
    elif cond in ("U-plain", "U-bridge"):
        seen = set()
        ec = "plain" if cond == "U-plain" else "bridge"
        pairs = [(d, "plain") for d in rec["arm_S"]["docs"]] + \
                [(d, ec) for d in rec["arm_E"]["docs"]]
        for d, which in pairs:
            if d["url"] in seen:
                continue
            seen.add(d["url"])
            if kept(d) and rd(d, which) in DIRS:
                out.append(rd(d, which))
            if len(seen) >= 20:
                break
    return out


def build_cond_jsonl(full: list[dict], cond: str) -> Path:
    p = OUT_DIR / f"cond_{cond}.jsonl"
    with p.open("w") as f:
        for rec in full:
            results = [{"read": {"direction": x}} for x in cond_dirs(rec, cond)]
            f.write(json.dumps({
                "review_url": rec["review_url"], "veracity": rec["veracity"],
                "publisher_site": rec["publisher_site"],
                "claim_resolved": rec.get("claim"), "claim_text": rec.get("claim_text"),
                "results": results}) + "\n")
    return p


def c6(rows: list[dict]) -> np.ndarray:
    return np.array([gu.flag_counts(r) for r in rows], dtype=float).T


def _silent_mask(C: np.ndarray) -> np.ndarray:
    """all-I claims: every read slot is I (no 5/4/3/X/2/1 flag). C is 7 x n in FLAGS7 order."""
    iI = gu.FLAGS7.index("I")
    other = [i for i in range(len(gu.FLAGS7)) if i != iI]
    return C[other, :].sum(0) == 0


def _recall_curve(s: np.ndarray, y: np.ndarray, budgets) -> dict:
    """recall at matched FPR budgets, threshold picked in-sample on the oof scores."""
    pairs = list(zip(s.tolist(), y.tolist()))
    out = {}
    for b in budgets:
        rec, fpr, thr = fit_urn.recall_at_fpr(pairs, b)
        out[f"{b:.2f}"] = {"recall": float(rec), "fpr": float(fpr), "threshold": float(thr)}
    return out


def _fit_block(C, y, mid, fold, cl, reps):
    """Point estimates + cluster-bootstrap CIs for one (sub)population. 7-flag."""
    s = fit_urn.oof_np(C, y, mid, fold)
    auc = float(_auc_np(s[y == 1], s[y == 0]))
    nest = fit_urn.nested_threshold(C, y, mid, fold)
    w = fit_urn.fit_np(C, y, mid, np.arange(C.shape[1]))
    _, _, gthr = fit_urn.recall_at_fpr(list(zip(s.tolist(), y.tolist())), 0.02)
    curve = _recall_curve(s, y, [0.01, 0.02, 0.05, 0.10])
    rng = np.random.default_rng(fit_urn.BOOT_SEED)
    A, W = [], []
    for _ in range(reps):
        bi = fit_urn.boot_idx(rng, y, cl, True)
        yy = y[bi]
        sb = fit_urn.oof_np(C, y, mid, fold, bi)
        A.append(_auc_np(sb[yy == 1], sb[yy == 0]))
        W.append(fit_urn.fit_np(C, y, mid, bi))
    A = np.array(A); W = np.array(W)
    return {
        "n": int(C.shape[1]), "n_true": int((y == 1).sum()), "n_false": int((y == 0).sum()),
        "auc_oof": auc, "auc_ci95": [float(x) for x in np.percentile(A, [2.5, 97.5])],
        "recall_at_2pct_fpr_nested": nest["recall"], "fpr_nested": nest["fpr"],
        "threshold_nested_mean": float(np.mean(nest["thresholds_by_fold"])),
        "threshold_global_2pct": float(gthr),
        "recall_at_fpr": curve,
        "weights": {k: float(v) for k, v in zip(gu.FLAGS7, w)},
        "weights_ci95": {k: [float(x) for x in np.percentile(W[:, i], [2.5, 97.5])]
                         for i, k in enumerate(gu.FLAGS7)},
    }


def score_arm_s_v5(run_tag: str, population_path: Path, weights_out: Path, reps: int) -> None:
    """Arm-S-only, 7-flag (read-v5) scoring of results_<run_tag>.jsonl on `population_path`.

    Reuses the graded_urn 7-flag fitter (gu.flag_counts + FLAGS7) and fit_urn's vectorised
    fit_np / oof_np / nested_threshold / recall_at_fpr / cluster bootstrap (nothing rewritten).
    Variants: plain and never-flag-on-silence (all-I claims excluded, silent counts reported).
    Also refits on the rep1500 subset for continuity with yesterday's Arm-S numbers.
    """
    import datetime
    gu.FLAGS = gu.FLAGS7
    read_hash = "92404a300e14"   # prompt_hash(reader_lab_prompts.PROMPTS['v5']); recorded in meta
    seen = {}
    for l in (OUT_DIR / f"results_{run_tag}.jsonl").open():
        r = json.loads(l)
        seen[r["review_url"]] = r
    full = list(seen.values())
    print(f"loaded {len(full)} claims from results_{run_tag}.jsonl (last-wins)")

    condp = OUT_DIR / f"cond_S_v5_{run_tag}.jsonl"
    with condp.open("w") as f:
        for rec in full:
            results = [{"read": {"direction": x}} for x in cond_dirs(rec, "S")]
            f.write(json.dumps({
                "review_url": rec["review_url"], "veracity": rec["veracity"],
                "publisher_site": rec["publisher_site"],
                "claim_resolved": rec.get("claim"), "claim_text": rec.get("claim_text"),
                "results": results}) + "\n")

    def compute(pop):
        rows = fit_urn.load_headline(condp, population=pop)
        order = sorted({r["review_url"] for r in rows})
        idx = {r["review_url"]: r for r in rows}
        rows = [idx[u] for u in order]
        C = np.array([gu.flag_counts(r) for r in rows], float).T
        y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
        fold = np.array([r["fold"] for r in rows]); cl = fit_urn.cluster_index(rows)
        plain = _fit_block(C, y, mid, fold, cl, reps)
        # never-flag-on-silence: drop all-I claims, recompute on the rest
        sil = _silent_mask(C)
        silent_counts = {
            "true": int(((y == 1) & sil).sum()),
            "false": int(((y == 0) & ~mid & sil).sum()),
            "mid": int((mid & sil).sum()), "total": int(sil.sum())}
        ns = ~sil
        Cn, yn, midn, foldn = C[:, ns], y[ns], mid[ns], fold[ns]
        cln = fit_urn.cluster_index([rows[i] for i in range(len(rows)) if ns[i]])
        nfs = _fit_block(Cn, yn, midn, foldn, cln, reps)
        nfs["silent_counts"] = silent_counts
        return {"plain": plain, "never_flag_on_silence": nfs}

    pop_bal = fit_urn.load_population(population_path)
    res_bal = compute(pop_bal)
    pop_rep = fit_urn.load_population(SRC / "eval/data/populations/fc_gold_rep1500.parquet")
    res_rep = compute(pop_rep)

    out = {
        "slice": str(population_path), "run_tag": run_tag,
        "reader": "read-v5", "read_hash": read_hash, "flags": list(gu.FLAGS7),
        "reps": reps, "date": datetime.date.today().isoformat(),
        "bal3000": res_bal, "rep1500_subset": res_rep,
        "continuity_note": ("yesterday Arm-S read-v6.1 six-flag on rep1500: AUC 0.9188, "
                            "recall@2%FPR 0.347"),
    }
    OUT_DIR.joinpath("prodregime_v5_metrics.json").write_text(json.dumps(out, indent=2))
    # the headline weights file the brief asks for (bal3000, plain)
    pb = res_bal["plain"]
    weights_out.write_text(json.dumps({
        "reader": "read-v5", "read_hash": read_hash, "flags": list(gu.FLAGS7),
        "slice": "fc_gold_bal3000", "date": datetime.date.today().isoformat(),
        "n": pb["n"], "n_true": pb["n_true"], "n_false": pb["n_false"],
        "weights": pb["weights"], "weights_ci95": pb["weights_ci95"],
        "threshold_nested_mean": pb["threshold_nested_mean"],
        "threshold_global_2pct": pb["threshold_global_2pct"],
        "auc_oof": pb["auc_oof"], "auc_ci95": pb["auc_ci95"],
        "recall_at_2pct_fpr_nested": pb["recall_at_2pct_fpr_nested"],
        "recall_at_fpr": pb["recall_at_fpr"],
        "never_flag_on_silence": {
            "auc_oof": res_bal["never_flag_on_silence"]["auc_oof"],
            "recall_at_2pct_fpr_nested": res_bal["never_flag_on_silence"]["recall_at_2pct_fpr_nested"],
            "silent_counts": res_bal["never_flag_on_silence"]["silent_counts"]},
    }, indent=2))

    def show(label, r):
        p, n = r["plain"], r["never_flag_on_silence"]
        print(f"\n== {label} (7-flag read-v5, Arm S) ==")
        print(f"  n={p['n']} ({p['n_true']}T/{p['n_false']}F) | plain oof AUC {p['auc_oof']:.4f} "
              f"{p['auc_ci95']} | nested rec@2%FPR {p['recall_at_2pct_fpr_nested']:.4f} "
              f"(fpr {p['fpr_nested']:.4f})")
        print("  recall@FPR:", {k: round(v["recall"], 4) for k, v in p["recall_at_fpr"].items()})
        print(f"  NFS: AUC {n['auc_oof']:.4f} | nested rec@2%FPR {n['recall_at_2pct_fpr_nested']:.4f}"
              f" | silent {n['silent_counts']}")
        print("  weights:", {k: round(v, 3) for k, v in p["weights"].items()})

    show("bal3000", res_bal)
    show("rep1500 subset", res_rep)
    print(f"\nwrote {OUT_DIR / 'prodregime_v5_metrics.json'} and {weights_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full", help="results_<run>.jsonl tag")
    ap.add_argument("--reps", type=int, default=fit_urn.BOOT_REPS)
    ap.add_argument("--mode", default="sixflag", choices=["sixflag", "arm_s_v5"],
                    help="sixflag = the original six-condition scorer; arm_s_v5 = Arm-S-only "
                         "7-flag read-v5 fit on --population")
    ap.add_argument("--population", type=Path, default=SRC / "eval/data/populations/fc_gold_bal3000.parquet",
                    help="population parquet for --mode arm_s_v5")
    ap.add_argument("--weights-out", type=Path,
                    default=OUT_DIR / "weights_v5_production_bal3000.json")
    a = ap.parse_args()
    if a.mode == "arm_s_v5":
        score_arm_s_v5(a.run, a.population, a.weights_out, a.reps)
        return
    gu.FLAGS = gu.FLAGS6
    pop = fit_urn.load_population(POP)
    # last-wins per review_url: a rerun appends a fresh record, so the LAST one is authoritative
    _seen = {}
    for l in (OUT_DIR / f"results_{a.run}.jsonl").open():
        r = json.loads(l)
        _seen[r["review_url"]] = r
    full = list(_seen.values())
    print(f"loaded {len(full)} claims from results_{a.run}.jsonl (last-wins)")

    paths = {"shipped": FROZEN}
    for cond in CONDS:
        if cond == "shipped":
            continue
        paths[cond] = build_cond_jsonl(full, cond)

    # load each condition through the production headline loader (media excluded, padded, clustered)
    rows_by = {c: fit_urn.load_headline(paths[c], population=pop) for c in CONDS}
    # align on the common review_url set + fixed order (folds/clusters identical across conds)
    common = set.intersection(*[{r["review_url"] for r in rows_by[c]} for c in CONDS])
    order = sorted(common)
    print(f"aligned claim set n={len(order)} (each cond padded to {fit_urn.PAD_TO} slots)")
    idx_of = {c: {r["review_url"]: r for r in rows_by[c]} for c in CONDS}
    base = [idx_of["S"][u] for u in order]
    y = np.array([r["y"] for r in base]); mid = np.array([r["mid"] for r in base])
    fold = np.array([r["fold"] for r in base]); cl = fit_urn.cluster_index(base)
    C = {c: np.array([gu.flag_counts(idx_of[c][u]) for u in order], float).T for c in CONDS}

    # point estimates per condition
    metrics = {}
    for c in CONDS:
        s = fit_urn.oof_np(C[c], y, mid, fold)
        auc = _auc_np(s[y == 1], s[y == 0])
        nest = fit_urn.nested_threshold(C[c], y, mid, fold)
        w = fit_urn.fit_np(C[c], y, mid, np.arange(len(order)))
        totals = C[c].sum(1)
        docs_per = float(C[c].sum() / len(order))
        metrics[c] = {
            "n": len(order), "auc_oof": float(auc),
            "recall_at_2pct_fpr": nest["recall"], "fpr": nest["fpr"],
            "threshold": float(np.mean(nest["thresholds_by_fold"])),
            "lr_plus": (nest["recall"] / nest["fpr"]) if nest["fpr"] else float("inf"),
            "weights": {k: float(v) for k, v in zip(gu.FLAGS6, w)},
            "flag_totals": {k: int(t) for k, t in zip(gu.FLAGS6, totals)},
            "docs_per_claim_scored": round(docs_per, 2)}

    # bootstrap: per-flag weight CIs (cluster) + paired dAUC vs S and vs shipped
    rng = np.random.default_rng(fit_urn.BOOT_SEED)
    W = {c: [] for c in CONDS}
    A = {c: [] for c in CONDS}
    for _ in range(a.reps):
        bi = fit_urn.boot_idx(rng, y, cl, True)
        yy = y[bi]
        for c in CONDS:
            W[c].append(fit_urn.fit_np(C[c], y, mid, bi))
            s = fit_urn.oof_np(C[c], y, mid, fold, bi)
            A[c].append(_auc_np(s[yy == 1], s[yy == 0]))
    for c in CONDS:
        Wc = np.array(W[c]); Ac = np.array(A[c])
        metrics[c]["weights_ci"] = {k: [float(x) for x in np.percentile(Wc[:, i], [2.5, 97.5])]
                                    for i, k in enumerate(gu.FLAGS6)}
        metrics[c]["auc_ci95"] = [float(x) for x in np.percentile(Ac, [2.5, 97.5])]
    for c in CONDS:
        for ref in ("S", "shipped"):
            d = np.array(A[c]) - np.array(A[ref])
            metrics[c][f"dauc_vs_{ref}"] = {
                "delta": float(np.mean(np.array(A[c])) - np.mean(np.array(A[ref]))),
                "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
    (OUT_DIR / "prodregime_metrics.json").write_text(json.dumps(
        {"population": str(POP), "n": len(order), "reps": a.reps,
         "flags": list(gu.FLAGS6), "conditions": metrics}, indent=2))

    # print table
    print(f"\n{'cond':10} {'n':>5} {'AUC':>7} {'AUC 95% CI':>18} {'rec@2%FPR':>10} {'FPR':>6} "
          f"{'thr':>8} {'docs/cl':>7}  dAUC_vs_S [CI]           dAUC_vs_shipped [CI]")
    for c in CONDS:
        m = metrics[c]
        ci = m["auc_ci95"]; ds = m["dauc_vs_S"]; dh = m["dauc_vs_shipped"]
        print(f"{c:10} {m['n']:5d} {m['auc_oof']:7.4f} [{ci[0]:.4f},{ci[1]:.4f}] "
              f"{m['recall_at_2pct_fpr']:10.4f} {m['fpr']:6.4f} {m['threshold']:+8.3f} "
              f"{m['docs_per_claim_scored']:7.2f}  "
              f"{ds['delta']:+.4f} [{ds['ci95'][0]:+.4f},{ds['ci95'][1]:+.4f}]  "
              f"{dh['delta']:+.4f} [{dh['ci95'][0]:+.4f},{dh['ci95'][1]:+.4f}]")
    print(f"\nwrote {OUT_DIR / 'prodregime_metrics.json'}")

    # weight table for S (candidate new baseline)
    print("\nWeight table — Arm S (six-flag, 95% cluster bootstrap CI):")
    for k in gu.FLAGS6:
        w = metrics["S"]["weights"][k]; lo, hi = metrics["S"]["weights_ci"][k]
        print(f"  {k}: {w:+.3f} [{lo:+.3f}, {hi:+.3f}]  {gu.FLAG_DESC[k]}")


if __name__ == "__main__":
    main()
