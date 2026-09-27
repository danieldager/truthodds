"""fc-gold v2 audit page — everything needed to rule on (1) publisher admission and
(2) harmonization-mapping correctness, one publisher per panel.

Per publisher: volume in the 2020+ gold window (claim_date>=2020 OR review_date>=2020),
date span, veracity mix bar, rule coverage, the COMPLETE rule-mapping table (the rule
layer is a pure function of the rating string, so this is exhaustive — every distinct
rating it mapped and to what), the top now-skipped tail ratings (LLM tail skipped per
Daniel 2026-07-23 — these rows stay out of gold), and stratified example claims.

  uv run python -m eval.scripts.build_eval.build_fc_gold_audit_html
Output: eval/data/fc_gold_v2_audit.html
"""
from __future__ import annotations

import html as H
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATA = Path("eval/data")
OUT = DATA / "fc_gold_v2_audit.html"
AUDITS = DATA / "publisher_audits.md"


def load_audits() -> dict:
    """publisher_site -> HTML block, parsed from publisher_audits.md
    (<!-- publisher: X --> markers; minimal md: **bold**, [t](url), - bullets)."""
    if not AUDITS.exists():
        return {}
    import re as _re
    out = {}
    parts = _re.split(r"<!-- publisher: ([\w.\-+]+) -->", AUDITS.read_text())
    for site, body in zip(parts[1::2], parts[2::2]):
        vm = _re.search(r"VERDICT: ?(.+?)$", body, _re.M)
        verdict = vm.group(1).strip() if vm else "pending"
        body = H.escape(body.strip())
        body = _re.sub(r"^## .*$", "", body, count=1, flags=_re.M)          # panel has its own title
        body = _re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                       r'<a href="\2" target="_blank">\1</a>', body)
        body = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", body)
        lines, html_ls, in_ul = body.splitlines(), [], False
        for l in lines:
            if l.strip().startswith("- "):
                if not in_ul:
                    html_ls.append("<ul>")
                    in_ul = True
                html_ls.append(f"<li>{l.strip()[2:]}</li>")
            elif l.strip():
                if in_ul and not l.startswith("  "):
                    html_ls.append("</ul>")
                    in_ul = False
                (html_ls.append(f"<p>{l.strip()}</p>") if not in_ul
                 else html_ls.append(l.strip()))
            elif in_ul:
                html_ls.append("</ul>")
                in_ul = False
        if in_ul:
            html_ls.append("</ul>")
        out[site] = {"verdict": verdict, "html": "\n".join(html_ls)}
    return out

VNAMES = {1: "false", 2: "mostly-false", 3: "contested/NEE", 4: "mostly-true", 5: "true"}
VCOL = {1: "#b23b3b", 2: "#cf8b4e", 3: "#8a8a8a", 4: "#7da05e", 5: "#3e7d46"}


def bar(counts: dict, total: int) -> str:
    segs = ""
    for v in (1, 2, 3, 4, 5):
        n = counts.get(v, 0)
        if n:
            segs += (f'<span title="{VNAMES[v]}: {n}" style="display:inline-block;'
                     f'width:{max(n/total*100, .4):.1f}%;background:{VCOL[v]};height:14px"></span>')
    none = total - sum(counts.values())
    if none:
        segs += (f'<span title="unresolved (skipped tail): {none}" style="display:inline-block;'
                 f'width:{max(none/total*100, .4):.1f}%;background:#ddd;height:14px"></span>')
    return f'<div style="width:100%;line-height:0">{segs}</div>'


def main():
    df = pl.read_parquet(DATA / "fc_gold_v2.parquet")
    gold = df.filter((pl.col("review_date") >= "2020") | (pl.col("claim_date") >= "2020"))
    audits = load_audits()
    # AFP audit covers both verticals under one section
    if "factcheck.afp.com" in audits:
        audits.setdefault("factuel.afp.com", audits["factcheck.afp.com"])

    pubs = (gold.group_by("publisher_site").len().sort("len", descending=True))
    panels = ""
    for site, n in pubs.iter_rows():
        g = gold.filter(pl.col("publisher_site") == site)
        vc = dict(g.group_by("veracity").len().iter_rows())
        vc = {k: v for k, v in vc.items() if k is not None}
        rule_pct = (g["harmonisation_source"] == "rule").mean() * 100
        dates = g["review_date"].drop_nulls()
        span = f"{dates.min()[:10]} → {dates.max()[:10]}" if len(dates) else "claim dates only"
        nee, cont = int(g["nee"].sum()), int(g["contested"].sum())
        ts = vc.get(4, 0) + vc.get(5, 0)

        # complete rule-mapping table for this publisher (full pull = max coverage)
        m = (df.filter((pl.col("publisher_site") == site) & (pl.col("harmonisation_source") == "rule"))
               .group_by("original_rating", "veracity", "rating_subtype").len()
               .sort("len", descending=True))
        rows = ""
        for r in m.iter_rows(named=True):
            v = r["veracity"]
            rows += (f'<tr><td>{H.escape(str(r["original_rating"]))[:80]}</td>'
                     f'<td style="color:{VCOL.get(v, "#888")}"><b>{v}</b> {H.escape(r["rating_subtype"])}</td>'
                     f'<td>{r["len"]}</td></tr>')

        t = (df.filter((pl.col("publisher_site") == site) & (pl.col("harmonisation_source") == "")
                       & (pl.col("rating_subtype") != "unrated"))
               .group_by("original_rating").len().sort("len", descending=True).head(10))
        tail_rows = "".join(f'<tr><td>{H.escape(str(r[0]))[:90]}</td><td>{r[1]}</td></tr>'
                            for r in t.iter_rows())

        resolved = g.filter(pl.col("veracity").is_not_null())
        ex = resolved.sample(min(6, len(resolved)), seed=11)
        ex_rows = "".join(
            f'<li><b>{r["veracity"]} {H.escape(r["rating_subtype"])}</b> — rating '
            f'<i>{H.escape(str(r["original_rating"]))[:60]}</i><br>'
            f'<span class="c">{H.escape(str(r["claim_text"]))[:160]}</span> '
            f'<a href="{H.escape(str(r["review_url"]))}" target="_blank">review</a></li>'
            for r in ex.iter_rows(named=True))

        a = audits.get(site)
        vtag = (a or {}).get("verdict", "no audit (<100 rows)")
        vcls = ("ok" if vtag.startswith("admit") and "caveat" not in vtag.lower() else
                "cav" if vtag.startswith("admit") else "warn")
        audit_html = (f'<div class="audit"><h4>Reliability audit — '
                      f'<span class="{vcls}">{H.escape(vtag)}</span></h4>{a["html"]}</div>'
                      if a else "")
        panels += f"""
<details class="pub"><summary><b>{H.escape(site)}</b> — <span class="{vcls}">{H.escape(vtag)}</span>
· {n} rows in window · true-side {ts} · contested {cont} · NEE {nee} · rules {rule_pct:.0f}% ·
{span}</summary>
{bar(vc, n)}
{audit_html}
<div class="cols">
<div><h4>Complete rule-mapping table ({len(m)} distinct ratings)</h4>
<table><tr><th>rating</th><th>→ veracity</th><th>rows</th></tr>{rows}</table></div>
<div><h4>Top skipped tail ratings (stay OUT of gold)</h4>
<table><tr><th>rating</th><th>rows</th></tr>{tail_rows or '<tr><td colspan=2>none</td></tr>'}</table>
<h4>Example claims (random, seeded)</h4><ul>{ex_rows}</ul></div>
</div></details>"""

    tot = len(gold)
    vc_all = {k: v for k, v in gold.group_by("veracity").len().iter_rows() if k is not None}
    doc = f"""<!doctype html><meta charset="utf-8">
<title>fc-gold v2 — harmonization & publisher-admission audit</title>
<style>
body{{font:14px/1.5 -apple-system,sans-serif;max-width:1150px;margin:2em auto;padding:0 1em;color:#222}}
table{{border-collapse:collapse;margin:.4em 0;width:100%}} td,th{{border:1px solid #ccc;padding:2px 8px;
text-align:left;font-size:13px}} th{{background:#f4f4f4}}
details.pub{{border:1px solid #ccc;margin:.7em 0;padding:.5em .8em;border-radius:4px}}
summary{{cursor:pointer}} .cols{{display:flex;gap:2em}} .cols>div{{flex:1;min-width:0}}
.c{{color:#555}} li{{margin:.5em 0;font-size:13px}} h4{{margin:.9em 0 .2em}}
.note{{color:#555;max-width:80ch}}
.ok{{color:#2c6e33;font-weight:600}} .cav{{color:#a06a00;font-weight:600}} .warn{{color:#b23b3b;font-weight:600}}
.audit{{background:#f8f8f6;border:1px solid #ddd;border-radius:4px;padding:.2em 1em;margin:.6em 0;font-size:13px}}
</style>
<h1>fc-gold v2 — audit for publisher admission &amp; mapping correctness</h1>
<p class="note">Window: claim_date ≥ 2020 OR review_date ≥ 2020 → <b>{tot} rows</b>.
Veracity: {" · ".join(f"{VNAMES[v]} {vc_all.get(v, 0)}" for v in (5, 4, 3, 2, 1))}
(+{tot - sum(vc_all.values())} unresolved tail rows, OUT of gold — LLM tail skipped).
The mapping tables are <b>exhaustive</b>: the rule layer is a pure function of the rating
string, so every distinct rating a publisher ever used is listed with its mapping — if the
table reads correctly, all its rows are correct. Admission is per-publisher: a publisher you
don't approve simply stays out of the fc-gold build (pull ⊃ gold).</p>
<p class="note"><b>RULING (Daniel 2026-07-23):</b> admitted = the 8 clean admits + politifact
+ snopes → <code>fc_gold_v2_admitted.parquet</code> (47,111 rows / 11 publisher sites).
Excluded: leadstories, factly, vishvasnews, altnews, rumorscanner, defacto-observatoire
(aggregator), and all unaudited sub-100-row publishers.</p>
{bar(vc_all, tot)}
<p class="note" style="font-size:12px">bar: <span style="color:{VCOL[5]}">true</span> ·
<span style="color:{VCOL[4]}">mostly-true</span> · <span style="color:{VCOL[3]}">contested/NEE</span> ·
<span style="color:{VCOL[2]}">mostly-false</span> · <span style="color:{VCOL[1]}">false</span> ·
grey = skipped tail</p>
{panels}
"""
    OUT.write_text(doc)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
