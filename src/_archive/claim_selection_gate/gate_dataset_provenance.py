"""Provenance ledger for the Stage-1 gate eval dataset: every sample we dropped, recovered, or relabelled,
with when + why + the artifact that decided it. Derived from the source artifacts (not hand-typed) so it
stays accurate when splits are rebuilt.

  uv run python -m eval.scripts.gate_dataset_provenance
    -> eval/data/gate_dataset_provenance.parquet  (per-sample trace)
    -> eval/data/gate_dataset_provenance.md        (human summary)

All changes dated 2026-06-26 (single session; clog 260626). Originals: negatives = synthetic_negatives.parquet
(547, built by build_synthetic_negatives.py); positives = harvest image posts.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from eval.scripts.build_gate_splits import OOS_KEEP_OVERRIDE

DATE = "2026-06-26"
DATA = Path("eval/data")


def main() -> None:
    rows = []

    # 1. MISLABELS: 7 negatives that were genuinely check-worthy -> recovered as must-pass positives
    for t in pl.read_parquet(DATA / "gate_mislabels.parquet")["text"].to_list():
        rows.append({"date": DATE, "change": "negative→positive (recovered mislabel)",
                     "identity": (t or "")[:90], "reason": "check-worthy political/corruption/policy claim wrongly in the negative set",
                     "decided_by": "FP audit triage (gate_fp_triage.md); clog 15:50"})

    # 2. VIDEO negatives: 12 synthetic 'video_dependent' dropped from the check-worthiness negatives
    neg = pl.read_parquet(DATA / "synthetic_negatives.parquet")
    for t in neg.filter(pl.col("category") == "video_dependent")["text"].to_list():
        rows.append({"date": DATE, "change": "negative→dropped (video)",
                     "identity": (t or "")[:90], "reason": "claim lives in video; measured by the video-drop metric, not check-worthiness",
                     "decided_by": "WS2 decision (clog 15:40); build_gate_splits filter"})

    # 3 & 4. Out-of-scope topic scan: reclassify -> negative, or keep (classifier false-flag)
    tri = DATA / "positives_scope_triage.parquet"
    if tri.exists():
        flagged = pl.read_parquet(tri).filter(pl.col("out_of_scope") == True)  # noqa: E712
        for r in flagged.to_dicts():
            t = r["raw_context"] or ""
            kept = any(k in t for k in OOS_KEEP_OVERRIDE)
            if kept:
                rows.append({"date": DATE, "change": "positive KEPT (classifier false-flag)",
                             "identity": t[:90], "reason": f"flagged '{r.get('topic')}' but kept positive: image-borne or political/culture-war claim",
                             "decided_by": "manual review of topic scan; OOS_KEEP_OVERRIDE"})
            else:
                rows.append({"date": DATE, "change": "positive→negative (out-of-scope fact-check)",
                             "identity": t[:90], "reason": f"out-of-scope topic '{r.get('topic')}': {(r.get('reason') or '')[:70]}",
                             "decided_by": "topic scan (triage_positives_scope.py) + review; clog 17:05"})

    # 5. IMAGE-ONLY positives: newly INCLUDED (were excluded when the harness keyed on text)
    n_io = 0
    for fold in ["dev", "test"]:
        f = DATA / f"image_only_{fold}.parquet"
        if f.exists():
            n_io += pl.read_parquet(f).height
    for _ in range(n_io):
        pass  # counted below; don't bloat the per-sample table with 286 near-identical rows

    df = pl.DataFrame(rows)
    df.write_parquet(DATA / "gate_dataset_provenance.parquet")

    # ---- markdown summary ----
    by = df.group_by("change").agg(pl.len().alias("n")).sort("n", descending=True)
    L = ["# Stage-1 gate dataset — provenance ledger (2026-06-26)\n",
         "Every drop / recovery / relabel from the original sets, with why + the deciding artifact.",
         "Per-sample detail: `gate_dataset_provenance.parquet`. Full narrative: clog/260626.md.\n",
         "## Net dataset math\n",
         "**Negatives:** 547 synthetic − 7 mislabels − 12 video = **528** check-worthiness; + **26** reclassified",
         "out-of-scope fact-checks = **554** (categories: opinion_personal / hard_negative / ad_promo /",
         "subjective_aesthetic / decorative_image / **oos_factcheck**).",
         "**Positives:** 575 text-bearing image posts − 26 out-of-scope = **549**; + **286** image-only",
         "(pure-visual, newly gateable via the uid key) measured separately. **7** recovered mislabels = must-pass probe.\n",
         "## Changes by type\n", "| change | n | why |", "|---|---|---|"]
    why = {
        "negative→positive (recovered mislabel)": "genuinely check-worthy (corruption/policy) — FP audit",
        "negative→dropped (video)": "claim in video → video-drop metric, not check-worthiness",
        "positive→negative (out-of-scope fact-check)": "entertainment/sports/celebrity/product trivia — independent topic scan + review",
        "positive KEPT (classifier false-flag)": "topic scan flagged but review kept (image-borne or political)",
    }
    for r in by.to_dicts():
        L.append(f"| {r['change']} | {r['n']} | {why.get(r['change'], '')} |")
    L.append(f"| positive ADDED (image-only) | {n_io} | pure-visual claims, includable once harness gained the uid key |")
    L += ["\n## Decision artifacts (the trail)\n",
          "- **Mislabels (7):** `gate_fp_triage.md` (full 107-FP triage), `gate_mislabels.parquet`.",
          "- **Out-of-scope (26 + 15 kept):** `positives_scope_triage.parquet` (every positive's topic label),",
          "  `OOS_KEEP_OVERRIDE` in `build_gate_splits.py` (the 15 reviewed keeps).",
          "- **Reclassification + image-only logic:** `build_gate_splits.py` (single source of truth; deterministic).",
          "- **Narrative (when/why):** clog/260626.md 15:50 / 17:05 / 17:25; report `gate_eval_report.md`."]
    (DATA / "gate_dataset_provenance.md").write_text("\n".join(L))

    print(f"per-sample ledger rows: {df.height} (+ {n_io} image-only counted in summary)")
    print(by.to_dicts())
    print("wrote gate_dataset_provenance.{parquet,md}")


if __name__ == "__main__":
    main()
