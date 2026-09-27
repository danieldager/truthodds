"""Substitute a reader_lab re-read's flags into a slim copy of a run's results jsonl.

The fitting scripts (fit_urn, graded_urn, fit_two_urn) read flags out of a run file
at `results[].read.direction`. reader_lab writes its re-read as one record per claim
with `docs[] = {rank, old, new}`. This script joins the two so a refit can run on a
different reader with no change to the fitting code.

`claims` writes the claim list reader_lab wants for one frozen population;
`substitute` writes the slim run file (only the fields the loaders read).

    uv run python -m eval.scripts.build_eval.substitute_reader_flags claims \
        --population fc_gold --out <claims.json>
    uv run python -m eval.scripts.build_eval.substitute_reader_flags substitute \
        --population fc_gold --reads <reader_lab jsonl> --out-dir <dir>

Documents the re-read never saw (no stored region: empty-doc or a failed fetch)
keep whatever flag the source file has. On all three frozen populations that is
"I" for every one of them (817 of 66,282), so the substituted file is single-reader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import polars as pl  # noqa: E402

POP = Path("eval/data/populations")
# population -> (parquet, side filter, source run files) -- manifest.json, 2026-09-10
SOURCES = {
    "fc_gold": (POP / "fc_gold.parquet", None,
                [Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")]),
    "cn_false": (POP / "cn_false.parquet", "false",
                 [Path("eval/data/urn_runs/c2_false/scores.jsonl"),
                  Path("eval/data/urn_runs/c2_false/scores_ext.jsonl")]),
    "x_feed": (POP / "x_feed.parquet", "true",
               [Path("eval/data/urn_runs/true_timeline/scores.jsonl")]),
}
# everything fit_urn.load / fit_two_urn.load_urn read off a record
KEEP = ("review_url", "veracity", "publisher_site", "ceiling_src", "screen_verdict",
        "claim_resolved", "claim_text", "post_id", "frame", "topic")


def population_ids(name: str) -> set[str]:
    path, side, _ = SOURCES[name]
    d = pl.read_parquet(path)
    if side is not None and "side" in d.columns:
        d = d.filter(pl.col("side") == side)
    return set(d["claim_id"].to_list())


def cmd_claims(a) -> None:
    want = population_ids(a.population)
    _, _, paths = SOURCES[a.population]
    out = []
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            if r.get("review_url") in want:
                out.append({"claim_id": r["review_url"], "group": a.population,
                            "post_id": r.get("post_id") or r["review_url"]})
    Path(a.out).write_text(json.dumps(out))
    print(f"{a.population}: {len(out)} of {len(want)} population claims -> {a.out}")


def cmd_substitute(a) -> None:
    want = population_ids(a.population)
    _, _, paths = SOURCES[a.population]
    new = {}
    for rp in a.reads:
        for line in Path(rp).open():
            r = json.loads(line)
            new[r["claim_id"]] = {d["rank"]: d["new"] for d in r["docs"]}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_claims = n_sub = n_kept = 0
    written = []
    for p in paths:
        outf = out_dir / f"{a.population}__{p.stem}.jsonl"
        written.append(outf)
        with outf.open("w") as f:
            for line in p.open():
                r = json.loads(line)
                if r.get("review_url") not in want:
                    continue
                sub = new.get(r["review_url"], {})
                docs = []
                for d in r.get("results") or []:
                    direction = (d.get("read") or {}).get("direction")
                    if direction is None:
                        continue
                    if d["rank"] in sub:
                        direction = sub[d["rank"]]
                        n_sub += 1
                    else:
                        n_kept += 1
                    docs.append({"rank": d["rank"], "domain": d.get("domain"),
                                 "read": {"direction": direction}})
                n_claims += 1
                f.write(json.dumps({**{k: r.get(k) for k in KEEP}, "results": docs},
                                   ensure_ascii=False) + "\n")
    print(f"{a.population}: {n_claims} claims, {n_sub} flags substituted, "
          f"{n_kept} kept from the source (no stored region)")
    for w in written:
        print(f"  wrote {w}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("claims"); c.set_defaults(fn=cmd_claims)
    c.add_argument("--population", required=True, choices=sorted(SOURCES))
    c.add_argument("--out", required=True)
    s = sub.add_parser("substitute"); s.set_defaults(fn=cmd_substitute)
    s.add_argument("--population", required=True, choices=sorted(SOURCES))
    s.add_argument("--reads", required=True, nargs="+", help="reader_lab output jsonl(s)")
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
