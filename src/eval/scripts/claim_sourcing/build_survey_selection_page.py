"""Build the survey claim selection page for the collaborators (Daniel 2026-09-22).

Joins the 500-post September pool (post-filter-v2 -> extract-key-v8 -> claim-check-v5 ->
political-gate-v1) with the survey-labels-v1 pass (claim lean, author lean, country, recycled
news) and the frozen instrument's dossiers (read-v5 @ Flash, run tag pool500), scores every
claim with the frozen seven-flag weights (fit_urn convention: pad to 10 slots with I, score =
sum(count x weight), FLAG when score <= boundary), and writes

  eval/data/community_notes/survey_pool500_scored_2026-09-22.parquet   one row per claim
  <out>/survey_selection.html                                           the selection page
  <out>/evidence.json                 per claim, the ten sources with the reader's verdict, cited
                                      sentences and reason (published next to the page, loaded on demand)
  <out>/media.json                    avatars and images as data URIs, loaded once by the page

The page places each POST in a cell by its first claim: claim lean (left / right / neither) x the
instrument's verdict in three bands: false (flagged, score <= the frozen boundary -4.0115), true
(score >= the display-only true threshold, true_threshold_2026-09-25.json) and unclear in between.
The true threshold only changes what the board shows; the frozen flag is untouched. Reviewers select / reject / abstain and comment;
votes go to the artifact db (collection reviews, one doc per reviewer) and can be copied as JSON.

  uv run python -m eval.scripts.claim_sourcing.build_survey_selection_page --out <dir>
$0, no API calls.
"""
from __future__ import annotations
import argparse
import collections
import html
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import base64
import io

import pandas as pd
from PIL import Image

SRC = Path(__file__).resolve().parents[3]
DATA = SRC / "eval/data/community_notes"
POOL = DATA / "survey_pool500_2026-09-21.parquet"
WEIGHTS = SRC / "eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json"
DOSS = SRC / "eval/data/urn_runs/cn_survey/results_pool500.jsonl"
SCORED = DATA / "survey_pool500_scored_2026-09-22.parquet"
TRUE_THRESHOLD = DATA / "true_threshold_2026-09-25.json"
TEMPLATE = Path(__file__).with_name("survey_selection_template.html")
FLAGS7 = ("5", "4", "3", "X", "I", "2", "1")
PAD_TO = 10
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def clean(s) -> str:
    """Text from the syndication feed and Community Notes arrives HTML-escaped (&amp; &quot; &gt;) and
    scraped pages can carry U+FFFD. Decode once here so the page escapes exactly once."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return html.unescape(str(s)).replace("\ufffd", "")


def _doc_date(s: str | None) -> str | None:
    """Serper dates look like 'Sep 14, 2026' or '3 days ago'. Return YYYY-MM or None."""
    if not s:
        return None
    m = re.match(r"([A-Z][a-z]{2}) \d{1,2}, (\d{4})", s)
    if m and m.group(1) in _MONTHS:
        return f"{m.group(2)}-{_MONTHS[m.group(1)]:02d}"
    return None


MAX_CITED, CITE_CHARS = 4, 220


def _evidence_rows(docs: list[dict]) -> list[dict]:
    """One row per retrieved source, in rank order, with the reader's verdict, the sentences it
    cited and its stated reason. Sources that were never read keep their status so all ten show."""
    rows = []
    for d in docs:
        rd = d.get("read") or {}
        cited = []
        if d.get("read_status") == "prepped" and rd.get("evidence"):
            pos = {sid: i for i, sid in enumerate(d.get("sent_ids") or [])}
            sents = d.get("sents") or []
            for sid in rd["evidence"][:MAX_CITED]:
                i = pos.get(sid)
                if i is not None and i < len(sents):
                    cited.append(clean(sents[i])[:CITE_CHARS])
        rows.append({"domain": d.get("domain") or "", "url": d.get("url") or "", "date": d.get("date") or "",
                     "status": d.get("read_status") or "", "dir": rd.get("direction") or "",
                     "cited": cited, "reason": clean(rd.get("reason"))[:400]})
    return rows


def score_dossiers(doss: Path, weights: Path) -> tuple[dict[str, dict], float, dict[str, list]]:
    wj = json.loads(weights.read_text())
    W = {k: float(wj["weights"][k]) for k in FLAGS7}
    thr = float(wj["threshold"])
    out, evidence = {}, {}
    for line in doss.open():
        r = json.loads(line)
        f = collections.Counter()
        dates, doms = [], []
        evidence[r["claim_id"]] = _evidence_rows(r.get("docs") or [])
        for d in r.get("docs") or []:
            rd = (d.get("read") or {}).get("direction")
            if d.get("read_status") == "prepped" and rd in FLAGS7:
                f[rd] += 1
                if rd != "I":
                    doms.append(d.get("domain") or "")
                    ym = _doc_date(d.get("date"))
                    if ym:
                        dates.append(ym)
        n_reads = sum(f.values())
        f["I"] += max(0, PAD_TO - n_reads)
        score = sum(W[k] * f[k] for k in FLAGS7)
        out[r["claim_id"]] = {
            "score": round(score, 3), "flag": score <= thr, "n_reads": n_reads,
            **{f"n_{k}": int(f[k]) for k in FLAGS7},
            "evidence_month": statistics.median_low(sorted(dates)) if dates else None,
            "top_domains": ", ".join(dict.fromkeys(doms).keys()) if doms else "",
            "query": r.get("query") or "",
            "error": r.get("error") or r.get("serper_error") or ""}
    return out, thr, evidence


def build(pool: Path, labels: Path, doss: Path, weights: Path, scored_out: Path, second: Path | None = None) -> tuple[pd.DataFrame, float]:
    p = pd.read_parquet(pool).rename(columns={"score": "rank_score"})   # the recency x likes rank
    p["post_id"] = p.post_id.astype(str)
    p["claim_id"] = p.post_id + "_" + p.claim_idx_in_post.astype(str)
    lab = {}
    head = {}
    for s in json.load(open(labels))["survivors"]:
        lab[(str(s["post_id"]), s["claim"])] = s
        head[str(s["post_id"])] = s
    for col in ("claim_lean", "claim_lean_confidence", "country", "recycled_news"):
        p[col] = [lab.get((a, b), {}).get(col) for a, b in zip(p.post_id, p.claim)]
    for col in ("author_lean", "author_lean_confidence"):
        p[col] = [head.get(a, {}).get(col) for a in p.post_id]
    sc, thr, evidence = score_dossiers(doss, weights)
    scdf = pd.DataFrame.from_dict(sc, orient="index")
    p = p.join(scdf, on="claim_id")
    p["boundary"] = thr
    so = json.loads(second.read_text())["claims"] if second else {}
    p["so_verdict"] = [so.get(c, {}).get("verdict") for c in p.claim_id]
    p["so_confidence"] = [so.get(c, {}).get("confidence") for c in p.claim_id]
    p["so_justification"] = [so.get(c, {}).get("justification") for c in p.claim_id]
    p["so_agrees"] = [so.get(c, {}).get("agrees") for c in p.claim_id]
    p = p.sort_values(["post_rank", "claim_idx_in_post"])
    p.to_parquet(scored_out)
    return p, thr, {c: evidence[c] for c in p.claim_id if c in evidence}


def shrink(uri: str | None, max_w: int, q: int) -> str | None:
    """Re-encode a data-URI JPEG smaller so the companion media file stays reviewable."""
    if not uri or not uri.startswith("data:image"):
        return uri
    try:
        im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def verdict_of(r, true_thr: float) -> str:
    if bool(r.flag):
        return "false"
    return "true" if r.score >= true_thr else "unclear"


def page_data(p: pd.DataFrame, media: dict, true_thr: float) -> list[dict]:
    posts = []
    for pid, g in p.groupby("post_id", sort=False):
        m = g.iloc[0]
        md = media.get(pid) or {}
        first = g[g.claim_idx_in_post == g.claim_idx_in_post.min()].iloc[0]
        lean = first.claim_lean if first.claim_lean in ("left", "right") else "neither"
        verdict = verdict_of(first, true_thr)
        posts.append({
            "id": pid, "rank": int(m.post_rank), "handle": m.handle or "",
            "name": clean(m.author_name), "date": str(m.created_at)[:16],
            "verified": bool(md.get("verified", m.verified)),
            "likes": int(m.likes or 0), "replies": int(m.replies or 0) if pd.notna(m.replies) else None,
            "text": clean(m.text), "n_media": len(md.get("media") or []), "has_media": bool(m.has_media),
            "quoted": {**md["quoted"], "text": clean(md["quoted"].get("text")), "name": clean(md["quoted"].get("name"))} if md.get("quoted") else None,
            "subtopic": m.subtopic, "country": m.country or "", "recycled": bool(m.recycled_news),
            "author_lean": m.author_lean or "neutral",
            "note": clean(m.best_note_summary), "note_class": m.best_note_classification or "",
            "note_status": m.best_note_status or "", "quasi": m.quasi_label if isinstance(m.quasi_label, str) else "",
            "cell": f"{lean}_{verdict}",
            "claims": [{
                "id": r.claim_id, "text": clean(r.claim), "score": None if pd.isna(r.score) else float(r.score),
                "flag": bool(r.flag) if pd.notna(r.flag) else None, "verdict": verdict_of(r, true_thr), "lean": r.claim_lean or "neither",
                "lean_conf": r.claim_lean_confidence or "", "n_reads": int(r.n_reads or 0),
                "counts": {k: int(getattr(r, f"n_{k}") or 0) for k in FLAGS7},
                "evidence_month": r.evidence_month if isinstance(r.evidence_month, str) else "",
                "domains": r.top_domains or "", "query": clean(r.query),
                "so": None if not isinstance(r.so_verdict, str) else
                {"verdict": r.so_verdict, "confidence": r.so_confidence, "agrees": None if pd.isna(r.so_agrees) else bool(r.so_agrees),
                 "justification": clean(r.so_justification)}}
                for r in g.itertuples()]})
    return posts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--labels", type=Path, required=True, help="survey-labels-v1 output json")
    ap.add_argument("--dossiers", type=Path, default=DOSS)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--scored-out", type=Path, default=SCORED)
    ap.add_argument("--media", type=Path, default=None, help="fetch_pool_media output json (avatars, images, quoted posts)")
    ap.add_argument("--second", type=Path, default=None, help="survey_second_opinion output json")
    ap.add_argument("--standalone", action="store_true",
                    help="also write survey_selection_standalone.html with media and evidence inlined (one file to email)")
    ap.add_argument("--media-width", type=int, default=240)
    ap.add_argument("--media-quality", type=int, default=50)
    ap.add_argument("--media-chunk", type=int, default=25, help="posts per media chunk file")
    ap.add_argument("--true-threshold", type=float, default=None,
                    help=f"display-only true cut (default: true_threshold in {TRUE_THRESHOLD.name})")
    ap.add_argument("--out", type=Path, required=True, help="directory for survey_selection.html")
    a = ap.parse_args()
    p, thr, evidence = build(a.pool, a.labels, a.dossiers, a.weights, a.scored_out, a.second)
    media = json.loads(a.media.read_text()) if a.media else {}
    true_thr = a.true_threshold if a.true_threshold is not None else float(json.loads(TRUE_THRESHOLD.read_text())["true_threshold"])
    posts = page_data(p, media, true_thr)
    cells = collections.Counter(x["cell"] for x in posts)
    cells2 = collections.Counter(x["cell"].replace("unclear", "true") for x in posts)
    wj = json.loads(a.weights.read_text())["weights"]
    meta = {"built": datetime.now().strftime("%Y-%m-%d %H:%M"), "boundary": thr, "true_threshold": true_thr,
            "score_min": round(PAD_TO * min(map(float, wj.values())), 2), "score_max": round(PAD_TO * max(map(float, wj.values())), 2),
            "n_posts": len(posts), "n_claims": int(len(p)), "cells": dict(cells),
            "instrument": "read-v5 @ DeepSeek-V4-Flash, Serper top ten, seven-flag weights bal3000"}
    # media goes out in small chunks (about 25 posts each) fetched on demand: one big companion file
    # is refused for viewers who are not signed in, small ones load
    order = [x["id"] for x in posts]
    chunks = [order[i:i + a.media_chunk] for i in range(0, len(order), a.media_chunk)]
    for ci, pids in enumerate(chunks):
        (a.out / f"media_{ci:02d}.json").write_text(json.dumps(
            {pid: {"avatar": shrink(media.get(pid, {}).get("avatar"), 64, 60),
                   "media": [{"type": m["type"], "src": shrink(m["src"], a.media_width, a.media_quality)}
                             for m in (media.get(pid, {}).get("media") or [])[:4]]} for pid in pids}))
    chunk_of = {pid: ci for ci, pids in enumerate(chunks) for pid in pids}
    for x in posts:
        x["mc"] = chunk_of[x["id"]]
    meta["n_media_chunks"] = len(chunks)
    html = TEMPLATE.read_text()
    html = html.replace("/*__DATA__*/", json.dumps({"meta": meta, "posts": posts}, ensure_ascii=False))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "survey_selection.html").write_text(html)
    (a.out / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False))
    if a.standalone:
        med = {}
        for ci in range(len(chunks)):
            med.update(json.loads((a.out / f"media_{ci:02d}.json").read_text()))
        ev = (a.out / "evidence.json").read_text()
        one = html.replace("function loadMedia(ci) {", f"const ALLMEDIA = {json.dumps(med)};\nfunction loadMedia(ci) {{ MEDIA = ALLMEDIA; return Promise.resolve(); }}\nfunction _unused(ci) {{")
        one = one.replace('fetch("evidence.json").then(r => r.json())', f"Promise.resolve({ev})")
        one = "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head><body>" + one + "</body></html>"
        (a.out / "survey_selection_standalone.html").write_text(one)
    print(f"posts {len(posts)} claims {len(p)} boundary {thr} true threshold {true_thr}")
    print(f"cells two bands {dict(sorted(cells2.items()))}\ncells three bands {dict(sorted(cells.items()))}")
    print(f"scored -> {a.scored_out}\npage -> {a.out / 'survey_selection.html'}")


if __name__ == "__main__":
    main()
