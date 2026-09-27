"""Render the markdown tables for bucket_opt_results.md from the run JSONs."""
import json
from pathlib import Path

OUT = Path("eval/data/urn_runs/e1_ctx/model_ladder/bucket_opt")


def main():
    D = json.loads((OUT / "bucket_opt_diag.json").read_text())
    R = json.loads((OUT / "bucket_opt_results.json").read_text())
    X = json.loads((OUT / "bucket_opt_extra_axes.json").read_text())
    S = R["structures"]

    L = []
    A = L.append

    A("### Table 1 — the 28 cells, diagnosed\n")
    A("Laplace log-LR per document, 95% interval from the pinned 2,000-rep bootstrap. "
      "`hw` is the interval half-width. Sorted by document share.\n")
    A("| cell | T docs | F docs | % docs | log-LR | 95% CI | hw |")
    A("|---|---:|---:|---:|---:|:---:|---:|")
    for c in D["cells"]:
        A(f"| `{c['cell'].replace('|', chr(92)+'|')}` | {c['docs_true']:,} | {c['docs_false']:,} | {c['pct_docs']:.2f} | "
          f"{c['loglr']:+.3f} | [{c['boot_ci'][0]:+.2f}, {c['boot_ci'][1]:+.2f}] | {c['boot_hw']:.3f} |")
    A("")
    A(f"Analytic delta-method SE understates the bootstrap by a median factor of "
      f"**{D['inflation']:.3f}** (documents inside one claim are correlated); that factor is "
      f"applied wherever the constraint is evaluated inside a fold. "
      f"**{len(D['thin_cells_hw035'])} of 28** cells miss `hw <= 0.35`: "
      + ", ".join(f"`{c}`" for c in D["thin_cells_hw035"]) + ".")
    A(f"\n**{D['n_overlapping_pairs']} of 378** cell pairs have overlapping 95% intervals, "
      "so most of the 28-cell grid is not separating anything.\n")

    A("\n### Table 2 — every structure, same population, folds, seed and bootstrap\n")
    A("Out-of-fold AUC on the pinned fc-gold n=3,274; dAUC paired against 7-flag; "
      "`hw` = min / median / max 95% half-width across the structure's own weights; "
      "transfer = urn-fitted, gold-evaluated at eps = 0.10, no refit.\n")
    A("| structure | params | oof AUC | 95% CI | dAUC vs 7-flag | rec@2% FPR | hw min/med/max | transfer AUC |")
    A("|---|---:|---:|:---:|:---:|---:|:---:|---:|")
    for n, r in S.items():
        k = r["k_eff_full"]
        k = f"{k:.1f}" if isinstance(k, float) and k != int(k) else str(int(k))
        tr = r.get("transfer_auc")
        A(f"| {n} | {k} | **{r['auc_oof']:.4f}** | [{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}] | "
          f"{r['dauc_vs_7flag']:+.4f} [{r['dauc_ci'][0]:+.4f}, {r['dauc_ci'][1]:+.4f}] | "
          f"{r['recall_2pct']*100:.1f}% | "
          f"{r.get('hw_min', float('nan')):.2f} / {r.get('hw_med', float('nan')):.2f} / "
          f"{r.get('hw_max', float('nan')):.2f} | "
          f"{f'{tr:.4f}' if tr else 'n/a'} |")

    A("\n\n### Table 3 — what the support constraint costs\n")
    A("Merge until every bucket's 95% half-width is at or below `h`, nested inside every fold.\n")
    A("| h | scope | buckets | oof AUC | rec@2% FPR | max hw reached |")
    A("|---:|---|---:|---:|---:|---:|")
    for k, v in R["support_constraint_sweep"].items():
        h, scope = k.split(" ", 1)
        A(f"| {h[2:]} | {scope} | {v['k_full']} | {v['auc_oof']:.4f} | "
          f"{v['recall_2pct']*100:.1f}% | {v['max_hw_achieved']:.3f} |")

    A("\n\n### Table 4 — the two untested raw axes\n")
    A("`rank` (retrieval position 1-10) and `provenance` (snippet vs scraped full read) are on "
      "every saved document. NewsGuard numeric bins are not retested: `quality_urn.py` already "
      "recorded that verdict (ng covers ~37% of documents).\n")
    A("| structure | base grid | oof AUC | 95% CI | dAUC vs 7-flag | rec@2% FPR |")
    A("|---|---:|---:|:---:|:---:|---:|")
    for n, r in X.items():
        A(f"| {n} | {r['n_base_cells']} | {r['auc_oof']:.4f} | "
          f"[{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}] | "
          f"{r['dauc_vs_7flag']:+.4f} [{r['dauc_ci'][0]:+.4f}, {r['dauc_ci'][1]:+.4f}] | "
          f"{r['recall_2pct']*100:.1f}% |")

    A("\n\n### The 10-bucket structure the constrained search chose\n")
    A("Weights and their 2,000-rep bootstrap intervals with that partition frozen. "
      "The tier axis survives in exactly three places; four of the seven flags keep no "
      "source split at all.\n")
    r10 = S["merge within-flag, inner-CV"]
    cnt = {c["cell"]: (c["docs_true"], c["docs_false"]) for c in D["cells"]}
    grp = [g.split("+") for g in r10["note_full"].split(" ; ")]
    tot = sum(sum(cnt[c]) for c in cnt)
    rowsw = []
    for g in grp:
        t = sum(cnt[c][0] for c in g); f = sum(cnt[c][1] for c in g)
        rowsw.append((g, t, f))
    import math
    NT = sum(t for _, t, _ in rowsw); NF = sum(f for _, _, f in rowsw); K = len(rowsw)
    for i, (g, t, f) in enumerate(rowsw):
        rowsw[i] = (g, t, f, math.log(((t+1)/(NT+K))/((f+1)/(NF+K))))
    order = sorted(range(K), key=lambda i: rowsw[i][3])          # params are weight-sorted
    ci = {order[i]: r10["params_ci"][i] for i in range(K)}
    A("| bucket | T docs | F docs | log-LR | 95% CI | hw |")
    A("|---|---:|---:|---:|:---:|---:|")
    for i, (g, t, f, w) in enumerate(rowsw):
        lo, hi = ci[i]
        name = g[0].split(" | ")[0] + " | " + ("all tiers" if len(g) == 4 else
                ("+".join(x.split(" | ")[1] for x in g)))
        A(f"| `{name.replace('|', chr(92)+'|')}` | {t:,} | {f:,} | {w:+.3f} | [{lo:+.2f}, {hi:+.2f}] | {(hi-lo)/2:.3f} |")

    A("\n\n### The partitions the search actually chose\n")
    for n in ("merge within-flag, inner-CV", "merge any, inner-CV",
              "merge within-flag, support", "merge any, support",
              "merge any, support+LRT"):
        r = S[n]
        A(f"\n**{n}** — {r['k_eff_full']} buckets on the full data, "
          f"per-fold bucket counts {r['fold_k_eff']}, "
          f"{'IDENTICAL' if r.get('partition_stable_across_folds') else 'NOT identical'} "
          f"across the five outer folds.\n")
        A("```")
        for g in r["note_full"].split(" ; "):
            A("  " + g)
        A("```")

    (OUT / "bucket_opt_tables.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
