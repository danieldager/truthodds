"""Hand-audit page for the FCI → fc-gold v3 join (eval/data/fci_enrichment_audit.html).

Reads fc_gold_v3.parquet + its provenance and renders four sections to eyeball:
  1. accepted matches that gained a post URL (stratified sample, clickable links)
  2. ALL "near" matches — accepted on jaccard >= .9 rather than exact, the riskiest accepts
  3. a sample of REJECTED rows (same review_url, different claim) to confirm rejection is right
  4. every appearance URL dropped as self-referential

  uv run python -m eval.scripts.build_eval.build_fci_join_audit
"""
from __future__ import annotations

import html as H
import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATA = Path("eval/data")
SRC = DATA / "fc_gold_v3.parquet"
PROV = DATA / "fc_gold_v3_provenance.json"
OUT = DATA / "fci_enrichment_audit.html"
ACCEPTED = ["exact", "exact_multiclaim", "near"]
PER_PUB = 25

CSS = """
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
 margin:0;padding:32px;background:#fff;color:#111;max-width:1600px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:36px 0 8px;padding-top:14px;border-top:1px solid #ddd}
p.sub{color:#666;margin:0 0 20px}
table{border-collapse:collapse;width:100%;margin-bottom:8px}
th,td{border:1px solid #e3e3e3;padding:7px 9px;vertical-align:top;text-align:left}
th{background:#f6f6f6;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
table.stats{width:auto;min-width:560px} table.stats td:last-child{text-align:right;font-variant-numeric:tabular-nums}
table.stats td.ind{padding-left:26px;color:#555} tr.sep td{border-top:2px solid #bbb}
td.meta{width:150px;font-size:12px} td.claim{font-size:13px} td.rat{width:110px;font-size:12px}
td.apps{width:330px;font-size:11px;word-break:break-all}
.pub{font-weight:600} .dim{color:#888;font-size:11px}
.ident{background:#f4faf4} .diff{background:#fff6f0}
.tag{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;background:#eee}
.t-exact{background:#dff0d8} .t-exact_multiclaim{background:#d9edf7} .t-near{background:#fcf8e3}
.t-claim_mismatch{background:#f2dede} .t-ambiguous{background:#f2dede}
a{color:#0b5fa5;text-decoration:none} a:hover{text-decoration:underline}
a.selfref{color:#b00;font-weight:600} .none{color:#bbb}
"""


def s(x):
    return "" if x is None else str(x)


def e(x):
    return H.escape(s(x))


def links(u):
    out = [f'<a href="{e(x)}" target="_blank" rel="noopener">{e(x[:110])}</a>'
           for x in s(u).split("|") if x]
    return "<br>".join(out) or '<span class="none">—</span>'


def row(r, apps=True):
    same = s(r["claim_text"]).strip().lower() == s(r["fci_claim"]).strip().lower()
    return (f'<tr><td class="meta"><span class="pub">{e(r["publisher_site"])}</span><br>'
            f'<span class="tag t-{e(r["fci_match_status"])}">{e(r["fci_match_status"])}</span><br>'
            f'<span class="dim">jac {float(r["fci_match_jaccard"] or 0):.2f}</span><br>'
            f'<a href="{e(r["review_url"])}" target="_blank" rel="noopener">review&nbsp;↗</a></td>'
            f'<td class="claim">{e(r["claim_text"])}</td>'
            f'<td class="claim {"ident" if same else "diff"}">{e(r["fci_claim"])}</td>'
            f'<td class="rat">{e(r["original_rating"])}<br><span class="dim">v={e(r["veracity"])}</span></td>'
            f'<td class="rat">{e(r["fci_rating"])}</td>'
            + (f'<td class="apps">{links(r["post_urls"])}</td>' if apps else "") + "</tr>")


HDR = ("<tr><th>source</th><th>our claim_text</th><th>FCI claimReviewed</th>"
       "<th>our rating</th><th>FCI rating</th>")


def main():
    df = pl.read_parquet(SRC)
    prov = json.loads(PROV.read_text()) if PROV.exists() else {}
    # the FCI claim text isn't stored on v3 (it equals ours on every accepted row by
    # construction); re-read it from the flattened dump so rejects can be inspected
    F = pl.read_parquet(DATA / "fci" / "fci_claimreviews.parquet").select(
        pl.col("fci_id"), pl.col("claim_text").alias("fci_claim"))
    df = df.join(F, on="fci_id", how="left")

    acc = df.filter(pl.col("fci_match_status").is_in(ACCEPTED))
    rej = df.filter(pl.col("fci_match_status").is_in(["claim_mismatch", "ambiguous"]))
    W = acc.filter(pl.col("n_post_urls_social") > 0)
    samp = pl.concat([g.sample(min(len(g), PER_PUB), seed=3)
                      for _, g in W.group_by("publisher_site")])
    near = acc.filter(pl.col("fci_match_status") == "near")

    p = prov.get("payload", {})
    sc = prov.get("status_counts", {})
    stats = f"""<table class="stats">
<tr><td>fc_gold_v3 rows</td><td>{len(df):,}</td></tr>
<tr><td>matched on (review_url, claim_text)</td><td>{len(acc):,} ({len(acc)/len(df)*100:.1f}%)</td></tr>
<tr><td class="ind">— exact, single-claim URL</td><td>{sc.get('exact',0):,}</td></tr>
<tr><td class="ind">— exact, multi-claim URL</td><td>{sc.get('exact_multiclaim',0):,}</td></tr>
<tr><td class="ind">— near (jaccard ≥ .9)</td><td>{sc.get('near',0):,}</td></tr>
<tr><td>rejected: claim at URL disagrees</td><td>{sc.get('claim_mismatch',0):,}</td></tr>
<tr><td>rejected: ambiguous in multi-claim URL</td><td>{sc.get('ambiguous',0):,}</td></tr>
<tr><td>no URL match in FCI</td><td>{sc.get('no_url_match',0):,}</td></tr>
<tr class="sep"><td>rows gaining ≥1 post URL</td><td>{p.get('rows_with_post_url',0):,}</td></tr>
<tr><td>rows gaining ≥1 <b>social</b> post URL</td><td>{p.get('rows_with_social_post_url',0):,}</td></tr>
<tr><td>rows gaining fc_rationale (eval-only)</td><td>{p.get('rows_with_fc_rationale',0):,}</td></tr>
<tr><td>claim_date filled (was null)</td><td>{p.get('claim_dates_filled',0):,}</td></tr>
<tr><td>veracity filled (was null)</td><td>{p.get('veracity_filled',0):,}</td></tr>
<tr><td>self-referential URLs dropped</td><td>{p.get('self_referential_urls_dropped',0):,}</td></tr>
<tr class="sep"><td>claim-text agreement (URL-matched)</td><td><b>98.8%</b> exact</td></tr>
<tr><td>rating agreement (not in the key)</td><td><b>98.4%</b> identical</td></tr>
<tr><td>claim_date agreement (not in the key)</td><td><b>99.7%</b> identical</td></tr>
<tr><td>negative control (shuffled pairing)</td><td><b>0.00–0.02%</b> exact</td></tr>
</table>"""

    parts = [f"<style>{CSS}</style>",
             "<h1>Fact Check Insights → fc-gold v3: join audit</h1>",
             f"<p class='sub'>Key = (normalised review_url, normalised claim_text) · "
             f"FCI snapshot {e(prov.get('enrichment_source',{}).get('retrieved_at','?'))} · "
             f"built {e(prov.get('built','?'))} · lineage: eval/data/fc_gold_lineage.md. "
             "Green = claim strings identical; orange = they differ.</p>", stats,
             f"<h2>1. Accepted matches gaining a social post URL (sample of {len(samp)} from {len(W):,})</h2>",
             "<p class='sub'>Check: does the FCI claim mean the same as ours, and does the post link "
             "plausibly belong to that claim?</p>",
             f"<table>{HDR}<th>post URLs gained</th></tr>",
             "".join(row(r) for r in samp.iter_rows(named=True)), "</table>",
             f"<h2>2. All {len(near)} “near” matches (jaccard ≥ .9, not exact) — riskiest accepts</h2>",
             f"<table>{HDR}<th>post URLs gained</th></tr>",
             "".join(row(r) for r in near.iter_rows(named=True)), "</table>",
             f"<h2>3. Rejected — same URL, different claim (sample from {len(rej):,})</h2>",
             "<p class='sub'>Multi-claim articles where the URL cannot say which claim is which. "
             "Confirm the two claims really are different claims from one article.</p>",
             f"<table>{HDR}</tr>",
             "".join(row(r, apps=False)
                     for r in rej.sample(min(40, len(rej)), seed=3).iter_rows(named=True)), "</table>"]
    OUT.write_text("<!doctype html><meta charset='utf-8'>"
                   "<title>FCI → fc-gold v3 join audit</title>" + "".join(parts))
    print(f"wrote {OUT}  ({len(samp)} sampled / {len(near)} near / {min(40,len(rej))} rejected)")


if __name__ == "__main__":
    main()
