"""Freeze the Truth Odds populations as id files.

Until today (Daniel, 2026-09-08) 4,035 / 3,211 / 1,719 / 1,999 existed only as
filters inside `fit_urn.load`, `fit_two_urn.load_urn` and `model_ladder`, plus
two exclusion JSONs and two environment variables (dataset_improvements.md
block A, first item). This writes each one out once, with the rule chain and the
sha256 of every file it consumed.

    uv run python -m eval.scripts.build_eval.freeze_populations

Outputs under eval/data/populations/: fc_gold.parquet, cn_false.parquet,
x_feed.parquet, manifest.json. One version per dataset (Daniel
2026-09-08); the 4,035-row fc-gold companion is gone.

The zero-doc rule is RETIRED (Daniel 2026-09-08). A claim whose search returned
nothing is ten silent documents, not an absent claim, so `fit_urn.load` pads
every claim to PAD_TO=10 slots and no population drops a claim for having no
reads. The one exception on the gold side is the 37 hand-gated claims
(e1_gate_decisions.json, 2026-08-04): they were never screened and never
searched, and Daniel's ruling keeps them out.

Two rules go BEYOND what the default loaders apply today, both decided 2026-09-08:
the CN urn drops claims whose note never locked (tier != gold), and the timeline
urn gets the residue screen plus the two CN-side parity screens (media locus,
duplicate claim) that the true side never had.

Rows are sorted by claim_id and columns are fixed, so a re-run is byte-identical.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.cn_false_stratum import MEDIA_RE, _jaccard, _tokens  # noqa: E402
from eval.scripts.build_eval.fit_urn import FLAG_TO_VOICE  # noqa: E402

E1 = Path("eval/data/urn_runs/e1_ctx")
C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline")
CN = Path("eval/data/community_notes")
OUT = Path("eval/data/populations")

E1_RESULTS = E1 / "results-00.jsonl"
E1_MEDIA = E1 / "media_provenance_exclusions.json"
JUDGED_AXIS = Path("eval/data/judged_axis_llm.parquet")
FC_GOLD_V3 = Path("eval/data/fc_gold_v3.parquet")
C2_SCORES = [C2 / "scores.jsonl", C2 / "scores_ext.jsonl"]
C2_FIT_EX = C2 / "fit_exclusions.json"
C2_MEDIA = C2 / "media_provenance_exclusions.json"
CN_CLUSTERS = CN / "cn_gold_clusters.parquet"
CN_POSTS = [Path("eval/data/tweet_corpus/cn_false_urn_posts.parquet"),
            Path("eval/data/tweet_corpus/cn_false_urn_ext_posts.parquet")]
TL_SCORES = TL / "scores.jsonl"
TL_SCREEN = TL / "tl_screen_nocontext.parquet"

MEDIA_AXIS = "media_authenticity"
DUP_JACCARD = 0.85   # cn_false_stratum.screen's threshold


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_flags(rec: dict) -> int:
    """Documents carrying a read flag. `fit_urn.load` keeps a claim iff this > 0."""
    return sum(1 for d in (rec.get("results") or [])
               if FLAG_TO_VOICE.get((d.get("read") or {}).get("direction")))


def excluded_prefixes(path: Path) -> set[str]:
    """claim[:80] keys, the CN-side join (fit_two_urn.load_urn)."""
    return {e["claim"][:80] for e in json.loads(path.read_text())
            if e.get("excluded", True)}


def write(df: pl.DataFrame, name: str, cols: list[str]) -> int:
    df = df.select(cols).sort("claim_id")
    df.write_parquet(OUT / name)
    print(f"  wrote {name}: {df.height} rows")
    return df.height


def fc_gold() -> dict:
    rows, n_gate, n_zero = [], 0, 0
    for line in E1_RESULTS.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        # Hand gate (Daniel 2026-08-04, e1_gate_decisions.json): these claims
        # were pulled before the claim screen ran, so they carry
        # screen_verdict == "unscreened" and no search was ever issued. Every
        # unscreened row in the draw is one of them. They stay excluded; the
        # remaining zero-doc claims are kept and padded to ten silences.
        if (r.get("screen_verdict") or "unscreened") == "unscreened":
            n_gate += 1
            continue
        if not n_flags(r):
            n_zero += 1
        rows.append(r)
    print(f"  fc-gold: {n_gate} hand-gated (unscreened) dropped, "
          f"{n_zero} zero-doc claims KEPT (padded to 10 silences)")
    axis = {a["review_url"]: a["judged_axis_llm"]
            for a in pl.read_parquet(JUDGED_AXIS).iter_rows(named=True)}
    clusters = {g["review_url"]: g["cluster_id"]
                for g in pl.read_parquet(FC_GOLD_V3).iter_rows(named=True)}
    unjoined = [r["review_url"] for r in rows if r["review_url"] not in clusters]
    if unjoined:
        print(f"  WARNING: {len(unjoined)} rows have no cluster_id in fc_gold_v3")
    df = pl.DataFrame([{
        "claim_id": r["review_url"],
        "cluster_id": clusters.get(r["review_url"]),
        "veracity": int(r["veracity"]),
        "rating_subtype": r.get("rating_subtype"),
        "judged_axis_llm": axis.get(r["review_url"]) or "untagged",
        "y": 1 if r["veracity"] >= 4 else 0,
        "mid": r["veracity"] == 3,
    } for r in rows])
    cols = ["claim_id", "cluster_id", "veracity", "rating_subtype",
            "judged_axis_llm", "y", "mid"]
    n_all = df.height

    media_ids = {e["claim_id"] for e in json.loads(E1_MEDIA.read_text())
                 if e.get("excluded", True)}
    cut = df.filter(pl.col("judged_axis_llm") != MEDIA_AXIS)
    n_axis = df.height - cut.height
    cut2 = cut.filter(pl.col("rating_subtype") != "mixed")
    n_mixed = cut.height - cut2.height
    cut3 = cut2.filter(~pl.col("claim_id").is_in(list(media_ids)))
    n_purge = cut2.height - cut3.height
    n_pin = write(cut3, "fc_gold.parquet", cols)
    print(f"  fc-gold removals: media_axis {n_axis}, mixed {n_mixed}, purge {n_purge}")

    inputs = [E1_RESULTS, JUDGED_AXIS, FC_GOLD_V3, E1_MEDIA]
    return {
        "rows": n_pin,
        "rules": [
            "E1 draw, urn_runs/e1_ctx/results-00.jsonl (4,157 scored rows)",
            "keep veracity in 1..5",
            f"minus hand-gated claims, screen_verdict == unscreened ({n_gate})",
            f"zero-doc claims KEPT ({n_zero}); fit_urn pads every claim to 10 "
            "slots with silent documents — the zero-doc rule is retired "
            "(Daniel 2026-09-08)",
            "cluster_id joined from fc_gold_v3.parquet on review_url",
            f"minus judged_axis_llm == media_authenticity ({n_axis})",
            f"minus rating_subtype == mixed ({n_mixed})",
            f"minus e1_ctx/media_provenance_exclusions.json, excluded=true ({n_purge})",
        ],
        "inputs": {str(p): sha256(p) for p in inputs},
    }


def cn_false() -> dict:
    fit_ex = excluded_prefixes(C2_FIT_EX)
    media_ex = excluded_prefixes(C2_MEDIA)
    rows, n_fit, n_media, n_zero = [], 0, 0, 0
    for p in C2_SCORES:
        for line in p.open():
            r = json.loads(line)
            claim = (r.get("claim_resolved") or r.get("claim_text") or "")[:80]
            if claim in fit_ex:
                n_fit += 1
                continue
            if claim in media_ex:
                n_media += 1
                continue
            if not n_flags(r):
                n_zero += 1      # kept: padded to 10 silences by fit_two_urn.load_urn
            rows.append(r)
    print(f"  CN removals: fit_exclusions {n_fit}, media purge {n_media}; "
          f"zero-doc {n_zero} KEPT (padded to 10 silences)")

    cl = pl.read_parquet(CN_CLUSTERS).select(["noteId", "tweetId", "tier"])
    tier_by_note = {c["noteId"]: c["tier"] for c in cl.iter_rows(named=True)}
    tier_by_tweet: dict[str, str] = {}
    for c in cl.iter_rows(named=True):
        tier_by_tweet.setdefault(c["tweetId"], c["tier"])
    handles: dict[str, str] = {}
    for p in CN_POSTS:
        for x in pl.read_parquet(p).select(["post_id", "handle"]).iter_rows(named=True):
            handles.setdefault(x["post_id"], x["handle"])

    kept, n_prov, n_untiered = [], 0, 0
    for r in rows:
        tier = tier_by_note.get(r.get("noteId")) or tier_by_tweet.get(r.get("post_id"))
        if tier is None:
            n_untiered += 1
        elif tier != "gold":
            n_prov += 1
            continue
        kept.append({"claim_id": r["review_url"], "post_id": r.get("post_id"),
                     "author": handles.get(r.get("post_id")), "side": "false"})
    print(f"  CN removals: provisional note tier {n_prov}" +
          (f", {n_untiered} rows had no note tier (kept)" if n_untiered else ""))
    n_author = sum(1 for k in kept if not k["author"])
    if n_author:
        print(f"  WARNING: {n_author} CN rows have no author from the posts draws")
    n = write(pl.DataFrame(kept), "cn_false.parquet",
              ["claim_id", "post_id", "author", "side"])
    return {
        "rows": n,
        "rules": [
            "urn_runs/c2_false/scores.jsonl + scores_ext.jsonl (2,086 scored claims)",
            f"minus fit_exclusions.json, claim[:80] prefix match ({n_fit})",
            f"minus c2_false/media_provenance_exclusions.json, excluded=true ({n_media})",
            f"zero-doc claims KEPT ({n_zero}); the urn loader pads every claim "
            "to 10 slots with silent documents — the zero-doc rule is retired "
            "(Daniel 2026-09-08)",
            f"minus claims whose Community Note never locked, tier != gold ({n_prov}) "
            "— NEW 2026-09-08, not applied by fit_two_urn.load_urn",
        ],
        "inputs": {str(p): sha256(p) for p in
                   C2_SCORES + [C2_FIT_EX, C2_MEDIA, CN_CLUSTERS] + CN_POSTS},
    }


def x_feed() -> dict:
    scored = [json.loads(line) for line in TL_SCORES.open()]
    rows = scored                      # zero-doc rule retired (Daniel 2026-09-08)
    n_scored = len(scored)
    n_zero = sum(1 for r in scored if not n_flags(r))
    print(f"  timeline: {n_scored} scored claims, {n_zero} zero-doc KEPT "
          "(padded to 10 silences)")

    scr = pl.read_parquet(TL_SCREEN)
    bad = set(scr.filter(pl.col("verdict") != "ok")["claim_id"].to_list())
    kept = [r for r in rows if r["review_url"] not in bad]
    n_screen = len(rows) - len(kept)

    def text(r: dict) -> str:
        return r.get("claim_resolved") or r.get("claim_text") or ""

    after_media = [r for r in kept if not MEDIA_RE.search(text(r))]
    n_media = len(kept) - len(after_media)

    final, sigs, n_dup = [], [], 0
    for r in after_media:
        t = _tokens(text(r))
        if any(_jaccard(t, s) >= DUP_JACCARD for s in sigs):
            n_dup += 1
            continue
        final.append(r)
        sigs.append(t)
    print(f"  timeline removals: residue screen {n_screen} (of {len(bad)} non-ok in the "
          f"screen file), media locus {n_media}, duplicate claim {n_dup}")

    n = write(pl.DataFrame([{
        "claim_id": r["review_url"], "post_id": r.get("post_id"),
        "handle": r.get("handle"), "frame": r.get("frame"), "side": "true",
    } for r in final]), "x_feed.parquet",
        ["claim_id", "post_id", "handle", "frame", "side"])
    print("  timeline frames: " + ", ".join(
        f"{k} {v}" for k, v in sorted(collections.Counter(
            r.get("frame") for r in final).items())))
    return {
        "rows": n,
        "rules": [
            f"urn_runs/true_timeline/scores.jsonl, the parity set ({n_scored} scored claims)",
            f"zero-doc claims KEPT ({n_zero}); the urn loader pads every claim "
            "to 10 slots with silent documents — the zero-doc rule is retired "
            "(Daniel 2026-09-08)",
            f"minus tl_screen_nocontext.parquet verdict != ok ({n_screen}) "
            "— NEW 2026-09-08, was a sensitivity cut only",
            f"minus media-locus claims, cn_false_stratum.MEDIA_RE ({n_media}) "
            "— NEW 2026-09-08, CN-side parity screen",
            f"minus near-duplicate claims, token Jaccard >= {DUP_JACCARD} first-kept "
            f"({n_dup}) — NEW 2026-09-08, CN-side parity screen",
        ],
        "inputs": {str(p): sha256(p) for p in [TL_SCORES, TL_SCREEN]},
    }


def main():
    argparse.ArgumentParser(description="Freeze the Truth Odds populations as id files.").parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("fc-gold:")
    m_pin = fc_gold()
    print("CN false urn:")
    m_cn = cn_false()
    print("timeline urn:")
    m_tl = x_feed()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    manifest = {
        "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_rev": rev,
        "pad_to": 10,
        "populations": {
            "fc_gold.parquet": m_pin,
            "cn_false.parquet": m_cn,
            "x_feed.parquet": m_tl,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest -> {OUT / 'manifest.json'}")
    for name in manifest["populations"]:
        print(f"  {name} sha256 {sha256(OUT / name)}")


if __name__ == "__main__":
    main()
