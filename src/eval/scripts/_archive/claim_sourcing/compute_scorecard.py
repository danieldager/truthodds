"""Compute LABEL-FREE INVARIANTS for a verify run and append a scorecard to a versioned ledger.

Reads the checkpointed JSONL shards of a verify run (results-*.jsonl), computes metrics that
need NO gold labels — label mix, round/exit behaviour, evidence sourcing quality, guard/refusal
firings, drop reasons, per-source sourcing, and a cost proxy — prints a human summary, appends
one JSON line to a versioned ledger, and regenerates three trend plots across ledger entries.

Everything is derived from the records themselves (per-claim `ledger`/`close_round`/`tries`,
per-round `provider`/`docs`/`dropped`, per-entry `evidence`, `guard_events`, `verdict`). Claim
metadata (ng_score, topic, flags) is joined from the claims parquet via each claim's `claim_id`.

  cd src && uv run python eval/scripts/claim_sourcing/compute_scorecard.py \
      --label "v7.2-dev500a" \
      [--run-dir eval/data/survey_claims/verify_v7_dev500] \
      [--claims eval/data/survey_claims/dev500_claims.parquet] \
      [--ledger eval/data/survey_claims/scorecards/ledger.jsonl] \
      [--plots-dir eval/data/survey_claims/scorecards/] \
      [--stability N]     # re-run N random posts twice through the LIVE loop (see stability_check)

--stability re-runs claims through the real verify loop, so it is behind the flag and OFF by
default. Every other metric is a pure read of the existing records.
"""
import argparse, glob, json, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[3]
LABELS = ["supported", "refuted", "conflicting", "unsupported"]
REFUSAL_GUARDS = ["close-below-bar", "close-unbacked", "conflict-one-sided",
                  "unsupported-untargeted", "budget-exhausted"]


def load_records(run_dir):
    recs = []
    for f in sorted(glob.glob(str(Path(run_dir) / "results-*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def pctl(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def cidx(claim_id):
    """Ledger/evidence key for a claim = the numeric suffix of its claim_id ('<post>:3' -> '3')."""
    return str(claim_id).split(":")[-1]


def snippet_dependent(evs):
    """Mirror the close bar: a close is snippet-DEPENDENT when there is no full-read PRIMARY/NG>=90
    anchor AND fewer than 2 full-read directional voices. Recomputed from the claim's evidence."""
    full = [e for e in evs if not e.get("snippet_only")]
    anchor = any((e.get("rel") == "PRIMARY" or (e.get("ng") or 0) >= 90) for e in full)
    voices = sum(1 for e in full if e.get("stance") in
                 ("supports", "refutes", "partially-supports", "partially-refutes"))
    return (not anchor) and voices < 2


def compute(recs, claims_df, label, run_dir):
    ok = [r for r in recs if r.get("ok")]
    meta_cid = {row.claim_id: row for row in claims_df.itertuples()}

    lab = Counter()               # label -> count
    nudge = 0
    close_round = Counter()       # exit round -> count
    close_provider = Counter()    # provider at close -> count
    stopped = Counter()
    rounds_per_post = []
    closed_12 = 0
    n_claims = 0
    ev_per_claim = []
    fullread = snippet = 0
    reltier = Counter()
    republication = 0
    stance_c = Counter()
    snip_dep = 0
    guards = Counter()
    dropreason = Counter()
    per_source = defaultdict(lambda: {"n": 0, "nonsup": 0, "ng": []})
    tot_rounds = tot_docs = tot_ev = 0
    elapsed = []
    loop_versions = Counter()

    for r in ok:
        res = r["result"]
        elapsed.append(r.get("elapsed_s"))
        loop_versions[res.get("loop_version")] += 1
        rnds = res.get("rounds", [])
        rounds_per_post.append(len(rnds))
        tot_rounds += len(rnds)
        stopped[res.get("stopped")] += 1
        if res.get("verdict", {}).get("nudge"):
            nudge += 1
        prov_by_round = {rd.get("round"): rd.get("provider") for rd in rnds}
        for rd in rnds:
            tot_docs += len(rd.get("docs", []))
            for dd in rd.get("dropped", []):
                dropreason[dd.get("reason")] += 1
        for ge in res.get("guard_events", []):
            guards[ge.get("guard")] += 1

        ledger = res.get("ledger", {})
        crmap = res.get("close_round", {})
        # evidence grouped by claim index
        ev_by = defaultdict(list)
        for e in res.get("evidence", []):
            tot_ev += 1
            ev_by[str(e.get("claim_id"))].append(e)
            if e.get("snippet_only"):
                snippet += 1
            else:
                fullread += 1
            reltier[e.get("rel")] += 1
            stance_c[e.get("stance")] += 1
            if e.get("republication"):
                republication += 1

        handle = res.get("handle")
        for c in res.get("claims", []):
            k = cidx(c["claim_id"])
            if k not in ledger:
                continue
            n_claims += 1
            l = ledger[k]
            lab[l] += 1
            cr = crmap.get(k)
            if cr is not None:
                close_round[cr] += 1
                close_provider[prov_by_round.get(cr, "?")] += 1
                if cr <= 2:
                    closed_12 += 1
            evs = ev_by.get(k, [])
            ev_per_claim.append(len(evs))
            if cr is not None and snippet_dependent(evs):
                snip_dep += 1
            per_source[handle]["n"] += 1
            if l != "supported":
                per_source[handle]["nonsup"] += 1
            m = meta_cid.get(c["claim_id"])
            if m is not None and pd.notna(getattr(m, "ng_score", None)):
                per_source[handle]["ng"].append(float(m.ng_score))

    n_claims = max(n_claims, 1)
    # per-source table + spearman(ng, non-support share) over sources with >=5 claims
    src_rows = []
    for h, d in per_source.items():
        if d["n"] == 0:
            continue
        src_rows.append({"handle": h, "n_claims": d["n"],
                         "nonsup_share": d["nonsup"] / d["n"],
                         "ng": float(np.mean(d["ng"])) if d["ng"] else None})
    src_rows.sort(key=lambda x: x["nonsup_share"], reverse=True)
    big = [s for s in src_rows if s["n_claims"] >= 5 and s["ng"] is not None]
    rho = spearman([s["ng"] for s in big], [s["nonsup_share"] for s in big]) if len(big) >= 3 else None

    refusals = {g: guards.get(g, 0) for g in REFUSAL_GUARDS}
    return {
        "meta": {
            "label": label, "run_dir": str(run_dir), "git_rev": git_rev(),
            "loop_version": dict(loop_versions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_posts": len(ok), "n_claims": sum(lab.values()),
            "errors": sum(1 for r in recs if not r.get("ok")),
        },
        "labels": {
            "counts": {k: lab.get(k, 0) for k in LABELS},
            "shares": {k: lab.get(k, 0) / n_claims for k in LABELS},
            "nudge_count": nudge, "nudge_rate": nudge / max(len(ok), 1),
        },
        "rounds": {
            "exit_round_dist": {str(k): v for k, v in sorted(close_round.items())},
            "stopped_reasons": dict(stopped),
            "mean_rounds_per_post": float(np.mean(rounds_per_post)) if rounds_per_post else None,
            "median_rounds_per_post": float(np.median(rounds_per_post)) if rounds_per_post else None,
            "closes_by_provider": dict(close_provider),
            "closes_by_round": {str(k): v for k, v in sorted(close_round.items())},
            "pct_closed_rounds_1_2": closed_12 / n_claims,
        },
        "evidence": {
            "per_claim_mean": float(np.mean(ev_per_claim)) if ev_per_claim else None,
            "per_claim_p50": pctl(ev_per_claim, 50), "per_claim_p90": pctl(ev_per_claim, 90),
            "fullread_share": fullread / max(fullread + snippet, 1),
            "snippet_share": snippet / max(fullread + snippet, 1),
            "rel_tier_shares": {k: v / max(sum(reltier.values()), 1) for k, v in reltier.items()},
            "stance_counts": dict(stance_c),
            "republication_flag_count": republication,
            "snippet_dependent_closes": snip_dep,
            "snippet_dependent_close_rate": snip_dep / n_claims,
        },
        "guards": {"by_type": dict(guards), "refusals": refusals},
        "drops": dict(dropreason),
        "per_source": {
            "top5_nonsup": src_rows[:5], "bottom5_nonsup": src_rows[-5:],
            "spearman_ng_vs_nonsup": rho, "n_sources_ge5": len(big),
        },
        "cost_proxy": {
            "total_rounds": tot_rounds, "total_docs_read": tot_docs,
            "total_evidence_entries": tot_ev,
            "mean_elapsed_s": float(np.mean([e for e in elapsed if e is not None]))
            if any(e is not None for e in elapsed) else None,
        },
    }


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for r, i in enumerate(order):
                rk[i] = r
            return rk
        rx, ry = ranks(x), ranks(y)
        n = len(x)
        if n < 2:
            return None
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else None


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(SRC), text=True).strip()
    except Exception:
        return None


def print_summary(sc):
    m, L, R, E = sc["meta"], sc["labels"], sc["rounds"], sc["evidence"]
    print(f"\n=== scorecard {m['label']}  (git {m['git_rev']}, loop {m['loop_version']}) ===")
    print(f"posts={m['n_posts']}  claims={m['n_claims']}  errors={m['errors']}")
    print("labels:  " + "  ".join(f"{k}={L['counts'][k]} ({L['shares'][k]:.1%})" for k in LABELS))
    print(f"nudge:   {L['nudge_count']} posts ({L['nudge_rate']:.1%})")
    print(f"rounds:  mean/post={R['mean_rounds_per_post']:.2f}  median={R['median_rounds_per_post']:.0f}"
          f"  closed_r1-2={R['pct_closed_rounds_1_2']:.1%}")
    print(f"         exit-round {R['exit_round_dist']}")
    print(f"         stopped {R['stopped_reasons']}  by-provider {R['closes_by_provider']}")
    print(f"evidence: /claim mean={E['per_claim_mean']:.1f} p50={E['per_claim_p50']:.0f} "
          f"p90={E['per_claim_p90']:.0f}  fullread={E['fullread_share']:.1%}")
    print("         rel " + " ".join(f"{k}={v:.0%}" for k, v in E["rel_tier_shares"].items()))
    print(f"         republication={E['republication_flag_count']}  "
          f"snippet-dependent closes={E['snippet_dependent_closes']} ({E['snippet_dependent_close_rate']:.1%})")
    print(f"guards:  by-type {sc['guards']['by_type']}")
    print(f"         refusals {sc['guards']['refusals']}")
    print(f"drops:   {sc['drops']}")
    ps = sc["per_source"]
    print(f"per-source: spearman(ng, non-sup)={ps['spearman_ng_vs_nonsup']} "
          f"over {ps['n_sources_ge5']} sources >=5 claims")
    print("         top non-sup: " + ", ".join(f"{s['handle']}={s['nonsup_share']:.0%}(n{s['n_claims']})"
                                                for s in ps["top5_nonsup"]))
    c = sc["cost_proxy"]
    print(f"cost:    rounds={c['total_rounds']} docs={c['total_docs_read']} ev={c['total_evidence_entries']} "
          f"mean_elapsed_s={c['mean_elapsed_s']:.1f}")


def append_ledger(sc, ledger_path):
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(sc) + "\n")
    return [json.loads(l) for l in open(ledger_path) if l.strip()]


def make_plots(entries, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = [e["meta"]["label"] for e in entries]
    grey = [str(g) for g in (0.15, 0.4, 0.6, 0.8)]
    out = []

    # (1) label-share stacked bars per version
    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.1), 4))
    bottom = np.zeros(len(xs))
    for lab, col in zip(LABELS, grey):
        vals = np.array([e["labels"]["shares"][lab] for e in entries])
        ax.bar(xs, vals, bottom=bottom, label=lab, color=col, edgecolor="white")
        bottom += vals
    ax.set_ylabel("share of claims"); ax.set_title("Label mix per version")
    ax.legend(fontsize=8, ncol=4); plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    p = plots_dir / "labels_stacked.png"; fig.savefig(p, dpi=120); plt.close(fig); out.append(p)

    # (2) nudge rate + refusal rates lines
    fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.1), 4))
    ax.plot(xs, [e["labels"]["nudge_rate"] for e in entries], "o-k", label="nudge rate")
    for g, col in zip(REFUSAL_GUARDS, ["0.3", "0.45", "0.6", "0.72", "0.82"]):
        ax.plot(xs, [e["guards"]["refusals"].get(g, 0) / max(e["meta"]["n_claims"], 1) for e in entries],
                "s--", color=col, label=g, markersize=4)
    ax.set_ylabel("rate (per post / per claim)"); ax.set_title("Nudge + refusal rates")
    ax.legend(fontsize=7); plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    p = plots_dir / "nudge_refusal_rates.png"; fig.savefig(p, dpi=120); plt.close(fig); out.append(p)

    # (3) exit-round distribution grouped bars per version
    maxr = max((int(k) for e in entries for k in e["rounds"]["exit_round_dist"]), default=1)
    rounds = list(range(1, min(maxr, 8) + 1))
    fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.1), 4))
    w = 0.8 / max(len(xs), 1)
    for i, e in enumerate(entries):
        tot = max(e["meta"]["n_claims"], 1)
        vals = [e["rounds"]["exit_round_dist"].get(str(r), 0) / tot for r in rounds]
        ax.bar(np.arange(len(rounds)) + i * w, vals, w, label=e["meta"]["label"],
               color=grey[i % len(grey)], edgecolor="white")
    ax.set_xticks(np.arange(len(rounds)) + 0.4 - w / 2); ax.set_xticklabels(rounds)
    ax.set_xlabel("exit round"); ax.set_ylabel("share of claims")
    ax.set_title("Exit-round distribution per version")
    ax.legend(fontsize=7); fig.tight_layout()
    p = plots_dir / "exit_round_dist.png"; fig.savefig(p, dpi=120); plt.close(fig); out.append(p)
    return out


def stability_check(run_dir, claims_path, n, seed=42):
    """Re-run N random posts twice through the LIVE verify loop and report label agreement.

    NOTE: this drives the real loop (Serper/Exa/LLM calls) — it is only reached behind --stability
    and is intentionally NOT exercised in the default scorecard path. Wired to the same three pieces
    run_tweet_verify.py uses: load_verify_posts -> run_posts(verify_post) -> read back the shards.
    """
    import asyncio, random, tempfile
    sys.path.insert(0, str(SRC)); sys.path.insert(0, str(Path(__file__).parent))
    from build_verify_input import load_verify_posts
    from pipeline.harness import run_posts
    from pipeline.pools import OrchestrationConfig, make_pools
    from pipeline.verify_tweet_claims import verify_post

    posts = [p for p in load_verify_posts(claims_path) if any(c["cw"] for c in p["claims"])]
    random.Random(seed).shuffle(posts)
    posts = posts[:n]

    async def _pass(tag):
        pools = make_pools(OrchestrationConfig())
        try:
            d = tempfile.mkdtemp(prefix=f"stab_{tag}_")
            await run_posts(posts, verify_post, pools, d, name=f"stab-{tag}")
            return {r["post_id"]: r["result"].get("ledger", {}) for r in load_records(d)}
        finally:
            await pools.close()

    a = asyncio.run(_pass("a")); b = asyncio.run(_pass("b"))
    same = tot = 0
    for pid, la in a.items():
        for k, v in la.items():
            tot += 1
            if b.get(pid, {}).get(k) == v:
                same += 1
    return {"n_posts": len(posts), "n_claims": tot,
            "label_agreement": same / tot if tot else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = SRC / "eval/data/survey_claims"
    ap.add_argument("--run-dir", default=str(base / "verify_v7_dev500"))
    ap.add_argument("--claims", default=str(base / "dev500_claims.parquet"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--ledger", default=str(base / "scorecards/ledger.jsonl"))
    ap.add_argument("--plots-dir", default=str(base / "scorecards"))
    ap.add_argument("--stability", type=int, default=0,
                    help="re-run N random posts twice through the LIVE loop (label agreement)")
    args = ap.parse_args()

    recs = load_records(args.run_dir)
    if not recs:
        sys.exit(f"no results-*.jsonl records in {args.run_dir}")
    claims_df = pd.read_parquet(args.claims)
    sc = compute(recs, claims_df, args.label, args.run_dir)

    if args.stability:
        sc["stability"] = stability_check(args.run_dir, args.claims, args.stability)

    print_summary(sc)
    entries = append_ledger(sc, args.ledger)
    plots = make_plots(entries, args.plots_dir)
    print(f"\nledger: {args.ledger}  ({len(entries)} entries)")
    for p in plots:
        print(f"plot:   {p}")


if __name__ == "__main__":
    main()
