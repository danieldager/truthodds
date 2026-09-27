"""Fit the Truth Odds signal weights and report the headline metrics.

Closes pre-run audit finding #68: until now every published AUC / recall number
came from throwaway scratch scripts, under slot conventions that were never
recorded, and no two of them agreed.

    uv run python -m eval.scripts.build_eval.fit_urn -i eval/data/urn_runs/e1/results-00.jsonl

Model. Each retrieved document is one voice. A voice is supporting, refuting, or
silent (neutral / irrelevant / junk). Per Truth Odds the score of a claim is the
sum of per-voice log-likelihood ratios:

    score = n_t * w_t + n_f * w_f + n_empty * w_e

where w_x = log( P(voice is x | claim TRUE) / P(voice is x | claim FALSE) ).

Everything reported is OUT OF FOLD. Weights are fitted on K-1 folds and the
held-out fold is scored with them; the pooled held-out scores are what AUC,
recall and LR+ are computed on. In-sample numbers are printed alongside only so
the optimism gap is visible.

Slot convention (revised, Daniel 2026-09-08). Every claim is scored over PAD_TO
= 10 slots. Slots that produced no read flag -- fewer than ten results returned,
an empty search, a fetch that failed -- count as SILENT documents (flag I). This
extends the convention already used for short result lists to the empty one, so
a claim that read nothing stays in the population as ten silences rather than
being dropped. `--no-pad` reproduces the old N=documents-returned convention.

Folds are assigned by blake2b of the claim's cluster_id (fc_gold_v3: two
fact-checks of one claim share a cluster; review_url only when the id is missing),
so they are stable across runs, independent of row order (see the 2026-07-27
shuffle bug) and cluster-disjoint (provenance audit 2026-09-08).

Intervals (2026-09-08). The headline AUC and weights carry 95% CLUSTER bootstrap
intervals (resample cluster_id, 2,000 reps, seed 707); the old class-stratified
row bootstrap is run alongside and the design effect (variance ratio
cluster / row) is reported. The operating point is NESTED: within each fold the
threshold is picked at FPR <= 2% on the four training folds (inner out-of-fold
scores) and applied to the held-out fold; the pooled held-out recall and realised
FPR are the headline, the old in-sample choice stays under `insample`.

Media-provenance exclusion (Daniel 2026-08-20). Claims whose JUDGED AXIS is
media_authenticity are excluded from the fit and the headline by default. The
fact-checker graded whether a photo or video shows what it is presented as showing;
our reader sees text only, so the gold label is about a proposition the system never
assessed. `claim_screen`'s media filter (build_calval_splits.py) cannot catch these:
it keys on claim_type, derived from the claim TEXT, and extraction has already
stripped the media framing ("this video shows a 7.1 quake in Miyazaki" arrives as
"7.1-Magnitude earthquake hits Miyazaki Prefecture"). 336 of 4,035 claims (8.3%) are
media-provenance and ALL of them carry a non-media claim_type. `--keep-media-axis`
reproduces the pre-2026-08-20 population.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DIRECTIONAL = ("supports", "refutes")
# read-v5 (2026-08-04): READ emits flags, not direction words. The committed
# 3-voice urn model is unchanged — flags map to voices (5/4 support, 1/2 refute,
# 3/X/I silent); raw per-flag counts ride along on each row for the graded
# Phase-3 refit. v4 direction words still map 1:1 so old runs stay loadable.
FLAG_TO_VOICE = {"5": "supports", "4": "supports", "1": "refutes", "2": "refutes",
                 "3": "neutral", "X": "neutral", "I": "irrelevant",
                 "supports": "supports", "refutes": "refutes", "neutral": "neutral",
                 "irrelevant": "irrelevant", "junk": "junk"}
STATES = ("supports", "refutes", "neutral", "irrelevant", "junk")
K_FOLDS = 5
# Slots per claim. Serper is asked for ten results; every slot that did not come
# back with a read flag is a silent document (see load()). Daniel 2026-09-08.
PAD_TO = 10
BOOT_REPS, BOOT_SEED = 2000, 707
JUDGED_AXIS = Path("eval/data/judged_axis_llm.parquet")
FC_GOLD = Path("eval/data/fc_gold_v3.parquet")
# THE SHIPPED CONSTANTS. Seven-flag since 2026-09-14 (graded_urn --ship), so
# `overall.weights` here is keyed by FLAG, not by voice; the three-voice fit on the
# same frozen population lives in E1_METRICS_CLUSTERED and fit_urn still writes it.
E1_METRICS_PATH = Path("eval/data/urn_runs/e1_ctx/headline_metrics.json")
E1_METRICS_CLUSTERED = Path("eval/data/urn_runs/e1_ctx/headline_metrics_clustered.json")
POPULATION_FC_GOLD = Path("eval/data/populations/fc_gold.parquet")
MEDIA_AXIS = "media_authenticity"

# ---- label-validity exclusions applied to the CN-false urn ------------------
# THE ONE PLACE these paths are named. Both urn loaders (fit_two_urn.load_urn,
# transfer_ladder.load_urn) call extra_exclusion_paths(); nothing else should
# hardcode a file here.
#
# DECIDED 2026-08-28 (Daniel): the media-provenance purge is applied BY DEFAULT
# under the MAJORITY-of-three-judges rule (250 of 1,969 urn claims). His reason:
# "better to be safe" — a mislabelled FALSE corrupts the fit worse than a lost
# claim costs us, and the CN corpus backfills almost free.
# The mixed-framing screen (mixed_framing_exclusions.json, 241 claims) is NOT
# in this list: it is a separate screen and was not part of that decision.
_C2 = Path("eval/data/urn_runs/c2_false")
DEFAULT_EXTRA_EXCLUSIONS = [_C2 / "media_provenance_exclusions.json"]

# ---- the same screen, applied to the fc-gold side ---------------------------
# The 2026-08-28 morning purge judged the CN corpus in FULL but fc-gold only as a
# 300-claim sample, so the gold side was unfinished. The afternoon sweep covered
# the whole gold FALSE side (veracity <= 3 of the pinned kept cut, n=1,772) with
# the SAME detector, the SAME three judges and the SAME majority rule.
#
# WHY A SEPARATE LIST AND A SEPARATE JOIN KEY. The CN files are matched on
# claim[:80] because the CN urn loader has no stable per-claim id. fc-gold does:
# review_url. Joining gold on its id is exact and cannot collide with a CN claim
# that happens to share an 80-char prefix, so the two screens can never bleed
# into each other's corpus.
_E1 = Path("eval/data/urn_runs/e1_ctx")
DEFAULT_GOLD_EXCLUSIONS = [_E1 / "media_provenance_exclusions.json"]


def extra_exclusion_paths() -> list[Path]:
    """$EXTRA_EXCLUSIONS overrides the default list:
    unset            -> DEFAULT_EXTRA_EXCLUSIONS (purge ON)
    "" / none / off  -> nothing (reproduces the pre-purge numbers)
    "a.json:b.json"  -> exactly those files
    """
    v = __import__("os").environ.get("EXTRA_EXCLUSIONS")
    if v is None:
        return list(DEFAULT_EXTRA_EXCLUSIONS)
    if v.strip().lower() in ("", "none", "off"):
        return []
    return [Path(p) for p in v.split(":") if p]


def gold_exclusion_paths() -> list[Path]:
    """fc-gold twin of extra_exclusion_paths(), with one extra hatch.

    $EXTRA_EXCLUSIONS unset       -> DEFAULT_GOLD_EXCLUSIONS (gold purge ON)
    $EXTRA_EXCLUSIONS ""/none/off -> nothing (turns EVERY label-validity screen off)
    $GOLD_EXCLUSIONS  ""/none/off -> nothing on the GOLD side only, so the CN purge
                                     can stay on while the gold purge is off. That
                                     is the before/after contrast for the gold sweep.
    $GOLD_EXCLUSIONS "a.json:b.json" -> exactly those files
    """
    env = __import__("os").environ
    v0 = env.get("EXTRA_EXCLUSIONS")
    if v0 is not None and v0.strip().lower() in ("", "none", "off"):
        return []
    v = env.get("GOLD_EXCLUSIONS")
    if v is None:
        return list(DEFAULT_GOLD_EXCLUSIONS)
    if v.strip().lower() in ("", "none", "off"):
        return []
    return [Path(p) for p in v.split(":") if p]


def gold_excluded_ids() -> set[str]:
    """review_urls the gold-side label-validity screens exclude. Missing file = none."""
    out: set[str] = set()
    for p in gold_exclusion_paths():
        if p.exists():
            out |= {e["claim_id"] for e in json.loads(p.read_text())
                    if e.get("excluded", True)}
    return out


def load_judged_axis() -> dict[str, str]:
    """review_url -> the proposition the FACT-CHECKER settled (derive_judged_axis.py)."""
    if not JUDGED_AXIS.exists():
        print(f"WARNING: {JUDGED_AXIS} missing — media-provenance claims cannot be excluded")
        return {}
    d = pl.read_parquet(JUDGED_AXIS)
    return {r["review_url"]: r["judged_axis_llm"] for r in d.iter_rows(named=True)}


def fold_of(review_url: str, k: int) -> int:
    return int(hashlib.blake2b(review_url.encode(), digest_size=8).hexdigest(), 16) % k


def load_clusters() -> dict[str, str]:
    """review_url -> cluster_id (fc_gold_v3). Missing file = every claim its own cluster."""
    if not FC_GOLD.exists():
        print(f"WARNING: {FC_GOLD} missing — folds fall back to review_url")
        return {}
    d = pl.read_parquet(FC_GOLD, columns=["review_url", "cluster_id"])
    return {u: str(c) for u, c in zip(d["review_url"], d["cluster_id"]) if c is not None}


def load_population(path: Path | None, side: str | None = None) -> set[str] | None:
    """claim_ids of a population parquet (column `claim_id`, optional `side`
    false/true for the two urns). None = no restriction."""
    if path is None:
        return None
    d = pl.read_parquet(path)
    if side is not None:
        d = d.filter(pl.col("side") == side)
    return set(d["claim_id"].to_list())


def load(path: Path, slots: int | None, demote_blanket: bool = False,
         axis: dict[str, str] | None = None,
         population: set[str] | None = None, pad: bool = True) -> list[dict]:
    """One record per gold-directional claim: label, voice counts, provenance.

    demote_blanket implements the silence-as-refutation SENSITIVITY (smoke audit
    2026-08-04): reads whose directional flag cites nearly the whole document are
    the signature of a reader inferring refutation from a document's SILENCE on
    the claim rather than from anything it asserts. No read-time discriminator
    separates those from genuine anchor refutations (anchor-coverage and
    query-term-coverage tests both measured, both overlap), so instead of
    correcting them we bound their effect: demoting them to silent voices and
    refitting says how much of the headline rests on silence. Silence already has
    its own fitted channel (n_e), so this returns that mass to where the model
    can price it."""
    out = []
    gold_excl = gold_excluded_ids()
    clusters = load_clusters()
    for line in path.open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        if r.get("review_url") in gold_excl:
            continue
        if population is not None and r.get("review_url") not in population:
            continue
        counts, flags = collections.Counter(), collections.Counter()
        for d in r.get("results") or []:
            read = d.get("read") or {}
            direction = read.get("direction")
            voice = FLAG_TO_VOICE.get(direction)
            if voice:
                if demote_blanket and read.get("qc_flag") == "blanket-citation":
                    voice = "neutral"      # silent channel, not a refuting voice
                counts[voice] += 1
                flags[direction] += 1
        # Pad rule (Daniel 2026-09-08). Every claim is scored over PAD_TO slots.
        # "Documents read" here means documents that carried a READ FLAG: a
        # search that returned nothing and a fetch that died before the reader
        # saw the page both leave a slot with no flag, and both count as
        # MISSING, hence silent. That is the same convention already applied
        # when Serper returns eight results instead of ten, where the two
        # missing slots ride in n_e. So a claim with no reads at all stays in
        # the population as PAD_TO silences (flag I) instead of being dropped.
        # A claim that somehow read more than PAD_TO documents is padded by zero.
        returned = sum(counts.values())
        if pad:
            n_pad = max(0, PAD_TO - returned)
            counts["irrelevant"] += n_pad
            flags["I"] += n_pad
            returned += n_pad
        elif not returned:
            continue
        n_slots = slots if slots else returned
        out.append({
            # Eval convention (Daniel 2026-08-19): mixed (3) counts as FALSE in
            # evaluation -- a half-true claim flagged is not a product error.
            # It stays OUT of the fit (no binary direction to learn from).
            "y": 1 if v >= 4 else 0,
            "mid": v == 3,
            "n_t": counts["supports"],
            "n_f": counts["refutes"],
            "n_e": max(0, n_slots - counts["supports"] - counts["refutes"]),
            "flags": dict(flags),
            "publisher": r.get("publisher_site") or "?",
            "ceiling_src": r.get("ceiling_src") or "none",
            "screen_verdict": r.get("screen_verdict") or "unscreened",
            "judged_axis": (axis or {}).get(r["review_url"]) or "untagged",
            "review_url": r["review_url"],
            "claim": r.get("claim_resolved") or r.get("claim_text") or "",
            "cluster": clusters.get(r["review_url"], r["review_url"]),
            "fold": fold_of(clusters.get(r["review_url"], r["review_url"]), K_FOLDS),
        })
    return out


def load_headline(path: Path, slots: int | None = None,
                  demote_blanket: bool = False,
                  population: set[str] | None = None,
                  pad: bool = True) -> list[dict]:
    """The headline population: load() with media-provenance claims removed.

    Every consumer of the headline numbers (fit_urn's own report, e1_figures,
    the artifacts) must go through this, or they silently disagree about which
    claims are in the corpus."""
    axis = load_judged_axis()
    return [r for r in load(path, slots, demote_blanket, axis, population, pad)
            if r["judged_axis"] != MEDIA_AXIS]


def fit(rows: list[dict]) -> dict[str, float]:
    """Per-voice log-LR, Laplace-smoothed. Empty pools give w=0, not an infinity.

    Mixed claims never enter the fit: their documents support the true half and
    refute the false half, so they carry no binary direction to count."""
    rows = [r for r in rows if not r.get("mid")]
    w = {}
    for key in ("n_t", "n_f", "n_e"):
        t = sum(r[key] for r in rows if r["y"] == 1)
        f = sum(r[key] for r in rows if r["y"] == 0)
        tot_t = sum(r["n_t"] + r["n_f"] + r["n_e"] for r in rows if r["y"] == 1)
        tot_f = sum(r["n_t"] + r["n_f"] + r["n_e"] for r in rows if r["y"] == 0)
        if not tot_t or not tot_f:
            w[key] = 0.0
            continue
        w[key] = math.log(((t + 1) / (tot_t + 3)) / ((f + 1) / (tot_f + 3)))
    return w


def score(row: dict, w: dict[str, float]) -> float:
    return row["n_t"] * w["n_t"] + row["n_f"] * w["n_f"] + row["n_e"] * w["n_e"]


def auc(pairs: list[tuple[float, int]]) -> float:
    """Mann-Whitney U. Ties count half."""
    pos = sorted(s for s, y in pairs if y == 1)
    neg = sorted(s for s, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for s in pos:
        lo, hi = bisect.bisect_left(neg, s), bisect.bisect_right(neg, s)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def recall_at_fpr(pairs: list[tuple[float, int]], max_fpr: float) -> tuple[float, float, float]:
    """Flag claims scoring at or below a threshold. Returns (recall, fpr, threshold).

    Sweeps every achievable operating point and takes the highest recall whose
    FPR stays within budget. FPR is over TRUE claims (a flagged true post is the
    error the product cannot afford).
    """
    trues = sorted(s for s, y in pairs if y == 1)
    falses = sorted(s for s, y in pairs if y == 0)
    if not trues or not falses:
        return float("nan"), float("nan"), float("nan")
    best = (0.0, 0.0, float("-inf"))
    for thr in sorted({s for s, _ in pairs}):
        fpr = bisect.bisect_right(trues, thr) / len(trues)
        if fpr > max_fpr:
            continue
        rec = bisect.bisect_right(falses, thr) / len(falses)
        if rec > best[0]:
            best = (rec, fpr, thr)
    return best


def out_of_fold(rows: list[dict]) -> list[tuple[float, int]]:
    scored = []
    for f in range(K_FOLDS):
        train = [r for r in rows if r["fold"] != f]
        test = [r for r in rows if r["fold"] == f]
        if not train or not test:
            continue
        w = fit(train)
        scored += [(score(r, w), r["y"]) for r in test]
    return scored


# ---- vectorised twins, shared by graded_urn / model_ladder / fit_two_urn ----
def fit_np(C: np.ndarray, y: np.ndarray, mid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """fit() on a K x n count matrix over the columns idx (duplicates allowed, so a
    bootstrap draw is just an idx): Laplace +1 per bucket, +K on each class total.
    Mixed never enters."""
    keep = idx[~mid[idx]]
    p, n = keep[y[keep] == 1], keep[y[keep] == 0]
    cT, cF = C[:, p].sum(1), C[:, n].sum(1)
    K = C.shape[0]
    return np.log(((cT + 1) / (cT.sum() + K)) / ((cF + 1) / (cF.sum() + K)))


def oof_np(C, y, mid, fold, idx=None) -> np.ndarray:
    """Out-of-fold scores of the columns idx (default: all), each fold scored by
    the weights fitted on the other folds of idx."""
    idx = np.arange(C.shape[1]) if idx is None else idx
    s = np.full(len(idx), np.nan)
    ff = fold[idx]
    for k in range(K_FOLDS):
        m = ff == k
        if not m.any() or m.all():
            continue
        s[m] = fit_np(C, y, mid, idx[~m]) @ C[:, idx[m]]
    return s


def boot_idx(rng, y: np.ndarray, cl: np.ndarray, clustered: bool) -> np.ndarray:
    """One bootstrap draw. Row: resample within gold class (the pre-2026-09-08
    design). Cluster: resample clusters with replacement; every row of a drawn
    cluster comes along, as many times as the cluster was drawn."""
    if clustered:
        G = int(cl.max()) + 1
        return np.repeat(np.arange(len(y)), np.bincount(rng.choice(G, G), minlength=G)[cl])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def cluster_index(rows: list[dict], key: str = "cluster") -> np.ndarray:
    return np.unique([str(r[key]) for r in rows], return_inverse=True)[1]


def _spread(thrs: list[float]) -> dict:
    """-inf = a fold with no operating point inside budget (tiny strata)."""
    fin = [t for t in thrs if math.isfinite(t)]
    return {"min": float(min(thrs)), "max": float(max(thrs)),
            "std": float(np.std(fin)) if fin else float("nan")}


def nested_threshold(C, y, mid, fold, max_fpr: float = 0.02) -> dict:
    """Nested operating point. For each outer fold: inner out-of-fold scores on
    the four training folds pick the threshold (recall maximiser at FPR <= budget),
    the model fitted on those four folds scores the held-out fold, the threshold
    is applied there. Pooled held-out recall / realised FPR are what is reported."""
    idx = np.arange(C.shape[1])
    flagged = np.zeros(len(idx), bool)
    thrs = []
    for k in range(K_FOLDS):
        te = fold == k
        if not te.any() or te.all():
            continue
        tr = idx[~te]
        s_in = oof_np(C, y, mid, fold, tr)
        _, _, thr = recall_at_fpr(list(zip(s_in.tolist(), y[tr].tolist())), max_fpr)
        flagged[te] = (fit_np(C, y, mid, tr) @ C[:, idx[te]]) <= thr
        thrs.append(float(thr))
    return {"recall": float(flagged[y == 0].mean()), "fpr": float(flagged[y == 1].mean()),
            "thresholds_by_fold": thrs, "threshold_spread": _spread(thrs)}


def nested_fixed(pairs: list[tuple[float, int]], fold: list[int], max_fpr: float = 0.02) -> dict:
    """nested_threshold for FIXED weights (no refit, e.g. the two-urn transfer):
    the threshold for fold k is picked on the other folds' scores and applied to k."""
    s = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    fold = np.asarray(fold)
    flagged = np.zeros(len(s), bool)
    thrs = []
    for k in range(K_FOLDS):
        te = fold == k
        if not te.any() or te.all():
            continue
        _, _, thr = recall_at_fpr(list(zip(s[~te].tolist(), y[~te].tolist())), max_fpr)
        flagged[te] = s[te] <= thr
        thrs.append(float(thr))
    return {"recall": float(flagged[y == 0].mean()), "fpr": float(flagged[y == 1].mean()),
            "thresholds_by_fold": thrs, "threshold_spread": _spread(thrs)}


def bootstrap(rows: list[dict], reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """Cluster bootstrap (headline) and the old class-stratified row bootstrap
    side by side, same seed, on the oof AUC and the full-fit weights. Design
    effect = var(cluster reps) / var(row reps)."""
    C = np.array([[r["n_t"], r["n_f"], r["n_e"]] for r in rows], float).T
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows]); cl = cluster_index(rows)
    out = {"reps": reps, "seed": seed, "n_clusters": int(cl.max()) + 1}
    aucs, ws = {}, {}
    for design in ("cluster", "row"):
        rng = np.random.default_rng(seed)
        a, w = [], []
        for _ in range(reps):
            idx = boot_idx(rng, y, cl, design == "cluster")
            w.append(fit_np(C, y, mid, idx))
            s, yy = oof_np(C, y, mid, fold, idx), y[idx]
            a.append(auc(list(zip(s.tolist(), yy.tolist()))))
        aucs[design], ws[design] = np.array(a), np.array(w)
    keys = ("n_t", "n_f", "n_e")
    out["auc_ci95"] = [float(x) for x in np.percentile(aucs["cluster"], [2.5, 97.5])]
    out["auc_ci95_row"] = [float(x) for x in np.percentile(aucs["row"], [2.5, 97.5])]
    out["design_effect_auc"] = float(aucs["cluster"].var() / aucs["row"].var())
    out["weights_ci"] = {k: [float(x) for x in np.percentile(ws["cluster"][:, i], [2.5, 97.5])]
                         for i, k in enumerate(keys)}
    out["weights_ci_row"] = {k: [float(x) for x in np.percentile(ws["row"][:, i], [2.5, 97.5])]
                             for i, k in enumerate(keys)}
    out["design_effect_weights"] = {k: float(ws["cluster"][:, i].var() / ws["row"][:, i].var())
                                    for i, k in enumerate(keys)}
    return out


def report(name: str, rows: list[dict], indent: str = "", boot: bool = False) -> dict:
    if len({r["y"] for r in rows}) < 2 or len(rows) < 20:
        print(f"{indent}{name:34s} n={len(rows):4d}  (too few / single-class, skipped)")
        return {}
    oof = out_of_fold(rows)
    w_all = fit(rows)
    ins = [(score(r, w_all), r["y"]) for r in rows]
    a_oof, a_ins = auc(oof), auc(ins)
    rec_i, fpr_i, thr_i = recall_at_fpr(oof, 0.02)
    C = np.array([[r["n_t"], r["n_f"], r["n_e"]] for r in rows], float).T
    nest = nested_threshold(C, np.array([r["y"] for r in rows]),
                            np.array([r["mid"] for r in rows]),
                            np.array([r["fold"] for r in rows]))
    rec, fpr = nest["recall"], nest["fpr"]
    lr_plus = (rec / fpr) if fpr > 0 else float("inf")
    n_t = sum(r["y"] for r in rows)
    print(f"{indent}{name:34s} n={len(rows):4d} (T {n_t} / F {len(rows)-n_t})  "
          f"AUC {a_oof:.3f} oof / {a_ins:.3f} in-sample   "
          f"recall@FPR<=2% {rec:.3f} nested (FPR {fpr:.3f}) / {rec_i:.3f} in-sample   "
          f"LR+ {lr_plus:.1f}")
    out = {"n": len(rows), "n_true": n_t, "auc_oof": a_oof, "auc_insample": a_ins,
           "recall_at_2pct_fpr": rec, "fpr": fpr,
           "thresholds_by_fold": nest["thresholds_by_fold"],
           "threshold_spread": nest["threshold_spread"], "lr_plus": lr_plus,
           "insample": {"recall_at_2pct_fpr": rec_i, "fpr": fpr_i, "threshold": thr_i,
                        "lr_plus": (rec_i / fpr_i) if fpr_i > 0 else float("inf")},
           "weights": w_all}
    if boot:
        print(f"{indent}  bootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}, cluster + row) ...")
        b = bootstrap(rows)
        out.update(b)
        print(f"{indent}  AUC {a_oof:.4f} [{b['auc_ci95'][0]:.4f}, {b['auc_ci95'][1]:.4f}] cluster / "
              f"[{b['auc_ci95_row'][0]:.4f}, {b['auc_ci95_row'][1]:.4f}] row   "
              f"design effect {b['design_effect_auc']:.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path, help="write the summary as JSON")
    ap.add_argument("--slots", type=int, default=None,
                    help="fix N slots per claim (default: documents actually returned)")
    ap.add_argument("--keep-media-axis", action="store_true",
                    help="keep claims the fact-checker graded on media provenance "
                         "(reproduces the pre-2026-08-20 population)")
    ap.add_argument("--demote-blanket", action="store_true",
                    help="silence-as-refutation sensitivity: count blanket-citation "
                         "directional reads as silent voices instead")
    ap.add_argument("--population", type=Path, default=None,
                    help="parquet with a claim_id column: restrict to those claims")
    ap.add_argument("--no-pad", action="store_true",
                    help=f"do not pad each claim to {PAD_TO} slots with silent "
                         "documents; drops claims that read nothing (pre-2026-09-08)")
    args = ap.parse_args()

    axis = load_judged_axis()
    rows = load(args.input, args.slots, args.demote_blanket, axis,
                load_population(args.population), pad=not args.no_pad)
    media = [r for r in rows if r["judged_axis"] == MEDIA_AXIS]
    if not args.keep_media_axis:
        rows = [r for r in rows if r["judged_axis"] != MEDIA_AXIS]
    print(f"media-provenance claims (judged_axis == {MEDIA_AXIS}): {len(media)} "
          f"{'KEPT (--keep-media-axis)' if args.keep_media_axis else 'EXCLUDED from fit + headline'}")
    print(f"claims: {len(rows)}  |  slot convention: "
          f"{'N=' + str(args.slots) if args.slots else ('N=returned (--no-pad)' if args.no_pad else f'N={PAD_TO}, unflagged slots silent')}"
          f"{'  |  blanket-citation reads DEMOTED to silent' if args.demote_blanket else ''}\n")

    summary = {"input": str(args.input), "slots": args.slots or "returned",
               "pad_to": None if args.no_pad else PAD_TO,
               "demote_blanket": args.demote_blanket,
               "media_axis_excluded": (0 if args.keep_media_axis else len(media)),
               "population": str(args.population) if args.population else None,
               "folds": K_FOLDS, "fold_key": "cluster_id",
               "overall": report("OVERALL", rows, boot=True)}

    w = summary["overall"].get("weights", {})
    if w:
        ci = summary["overall"]["weights_ci"]
        print("\nfitted weights (all claims), 95% cluster-bootstrap CI: "
              + "  ".join(f"{lab} {w[k]:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]"
                          for k, lab in (("n_t", "supporting"), ("n_f", "refuting"),
                                         ("n_e", "silent"))) + " per voice\n")

    # The two confounds the pre-run audit named as threats to the headline.
    print("by date-ceiling provenance (audit finding #70):")
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src["lag-derived" if "lag" in r["ceiling_src"] else r["ceiling_src"]].append(r)
    summary["by_ceiling_src"] = {k: report(k, v, "  ")
                                 for k, v in sorted(by_src.items(), key=lambda x: -len(x[1]))}

    # Claim-screen strata (2026-08-04). "ok" is the headline population; the
    # ill-posed classes are reported beside it so a gold-misaligned claim shows
    # up as its own number instead of quietly moving the headline.
    screened = [r for r in rows if r["screen_verdict"] != "unscreened"]
    if screened:
        print("\nby claim-screen verdict (headline = 'ok' only):")
        by_scr = collections.defaultdict(list)
        for r in screened:
            by_scr[r["screen_verdict"]].append(r)
        summary["by_screen_verdict"] = {
            k: report(k, v, "  ")
            for k, v in sorted(by_scr.items(), key=lambda x: -len(x[1]))}
        summary["headline_screened_ok"] = report(
            "OVERALL (screen=ok)", by_scr.get("ok", []), "  ")

    # The excluded stratum, reported so its size and behaviour stay visible.
    if media:
        print(f"\nmedia-provenance stratum ({'kept above' if args.keep_media_axis else 'EXCLUDED above'}):")
        summary["media_axis_stratum"] = report("judged_axis == media_authenticity", media, "  ")

    print("\nby judged axis (the proposition the fact-checker settled):")
    by_axis = collections.defaultdict(list)
    for r in rows:
        by_axis[r["judged_axis"]].append(r)
    summary["by_judged_axis"] = {k: report(k, v, "  ")
                                 for k, v in sorted(by_axis.items(), key=lambda x: -len(x[1]))}

    print("\nby publisher (audit finding #78):")
    by_pub = collections.defaultdict(list)
    for r in rows:
        by_pub[r["publisher"]].append(r)
    summary["by_publisher"] = {k: report(k, v, "  ")
                               for k, v in sorted(by_pub.items(), key=lambda x: -len(x[1]))[:5]}

    if args.out:
        if args.out.resolve() == E1_METRICS_PATH.resolve():
            sys.exit(f"refusing to write a 3-voice fit to {E1_METRICS_PATH}: it holds the "
                     f"SHIPPED 7-flag constants (graded_urn --ship). Write "
                     f"{E1_METRICS_CLUSTERED} instead.")
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
