"""Build the vote review page: every post that got at least one vote on the survey selection board,
with each reviewer's vote and comment side by side, for the three of us to walk through together.

Reads the reviewers' "Copy my JSON" exports (reviewer, exported, votes[{post_id, vote, comment, cell,
claims}]) and resolves every voted post from the current US pool (scored parquet, 2026-09-25) or, when
the post was dropped from it, from the earlier 656-post pool (2026-09-22), marked "not in current pool".
The cell is recomputed from the pool the post resolves to with the board's three-band rule (first
claim's lean x verdict: false when flagged, score <= the frozen boundary; true at or above the
display-only true threshold, true_threshold_2026-09-25.json; unclear in between), so votes recorded
under the earlier two-band cells land in their current cell. A reviewer may pass several exports:
they merge per post_id with the later `exported` winning. Nothing is re-scored: the scored parquets already carry the instrument
score, flag and second opinion.

Writes to --out
  vote_review.html        the page (post data inline)
  chunk_NN.json           per 25 posts, their media (avatars, images) and evidence rows, loaded on demand
  vote_review_standalone.html   with --standalone, one file with every chunk inlined

  uv run python -m eval.scripts.claim_sourcing.build_vote_review_page \
      --votes r1.json r2.json r3.json --media pool_media_lf.json pool_media.json --out <dir> --standalone
$0, no API calls.
"""
from __future__ import annotations
import argparse
import base64
import collections
import html
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

SRC = Path(__file__).resolve().parents[3]
DATA = SRC / "eval/data/community_notes"
DOSS_DIR = SRC / "eval/data/urn_runs/cn_survey"
POOLS = [  # (label, scored parquet, dossiers), current pool first
    ("us_0925", DATA / "survey_pool_us_scored_2026-09-25.parquet", DOSS_DIR / "results_pool_us.jsonl"),
    ("pool656_0922", DATA / "survey_pool656_scored_2026-09-22.parquet", DOSS_DIR / "results_pool500.jsonl"),
]
TRUE_THRESHOLD = DATA / "true_threshold_2026-09-25.json"
WEIGHTS = SRC / "eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json"
TEMPLATE = Path(__file__).with_name("vote_review_template.html")
FLAGS7 = ("5", "4", "3", "X", "I", "2", "1")
PAD_TO = 10
REVIEWERS = ("r1", "r2", "r3")
MAX_CITED, CITE_CHARS = 4, 220


def clean(s) -> str:
    """Scraped text arrives HTML-escaped and can carry U+FFFD. Decode once so the page escapes once."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return html.unescape(str(s)).replace("�", "")


def evidence_rows(docs: list[dict]) -> list[dict]:
    """Copied from build_survey_selection_page: one row per retrieved source with the reader's verdict."""
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


def shrink(uri: str | None, max_w: int, q: int) -> str | None:
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


def load_votes(paths: list[Path]) -> tuple[dict[str, dict], dict[str, str]]:
    """{post_id: {reviewer: {v, c, cell, claims}}} and {reviewer: latest exported}. Several exports per
    reviewer merge per post_id, the later `exported` winning."""
    votes, exported = collections.defaultdict(dict), {}
    exports = [json.loads(p.read_text()) | {"_stem": p.stem} for p in paths]
    for j in sorted(exports, key=lambda j: j.get("exported") or ""):
        who = (j.get("reviewer") or j["_stem"]).strip().lower()
        exported[who] = j.get("exported") or ""
        for v in j["votes"]:
            if v.get("vote") or v.get("comment"):
                votes[str(v["post_id"])][who] = {"v": v.get("vote") or "", "c": (v.get("comment") or "").replace("�", ""),
                                                 "cell": v.get("cell") or "", "claims": v.get("claims") or []}
    return dict(votes), exported


def verdict_of(r, true_thr: float) -> str:
    """The board's three bands (copied from build_survey_selection_page)."""
    if bool(r.flag):
        return "false"
    return "true" if r.score >= true_thr else "unclear"


def post_dict(pid: str, g: pd.DataFrame, md: dict, in_current: bool, true_thr: float) -> dict:
    """The board's page_data row for one post (copied), plus in_current."""
    g = g.sort_values("claim_idx_in_post")
    m = g.iloc[0]
    lean = m.claim_lean if m.claim_lean in ("left", "right") else "neither"
    verdict = verdict_of(m, true_thr)
    q = md.get("quoted")
    return {
        "id": pid, "handle": m.handle or "", "name": clean(m.author_name), "date": str(m.created_at)[:16],
        "verified": bool(md.get("verified", m.verified)),
        "likes": int(m.likes or 0), "replies": int(m.replies) if pd.notna(m.replies) else None,
        "text": clean(m.text), "n_media": len(md.get("media") or []), "has_media": bool(m.has_media),
        "quoted": {**q, "text": clean(q.get("text")), "name": clean(q.get("name"))} if q else None,
        "subtopic": m.subtopic, "country": m.country or "", "recycled": bool(m.recycled_news),
        "author_lean": m.author_lean or "neutral",
        "note": clean(m.best_note_summary), "note_class": m.best_note_classification or "",
        "note_status": m.best_note_status or "", "in_current": in_current,
        "cell": f"{lean}_{verdict}",
        "claims": [{
            "id": r.claim_id, "text": clean(r.claim), "score": None if pd.isna(r.score) else float(r.score),
            "flag": bool(r.flag) if pd.notna(r.flag) else None, "verdict": verdict_of(r, true_thr), "lean": r.claim_lean or "neither",
            "lean_conf": r.claim_lean_confidence or "", "query": clean(r.query),
            "evidence_month": r.evidence_month if isinstance(r.evidence_month, str) else "",
            "so": None if not isinstance(r.so_verdict, str) else
            {"verdict": r.so_verdict, "confidence": r.so_confidence, "agrees": None if pd.isna(r.so_agrees) else bool(r.so_agrees),
             "justification": clean(r.so_justification)}}
            for r in g.itertuples()]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=Path, nargs="+", required=True, help="reviewers' Copy-my-JSON exports")
    ap.add_argument("--media", type=Path, nargs="*", default=[], help="fetch_pool_media outputs, earlier files win")
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--chunk", type=int, default=25, help="posts per media/evidence chunk")
    ap.add_argument("--media-width", type=int, default=240)
    ap.add_argument("--media-quality", type=int, default=50)
    ap.add_argument("--true-threshold", type=float, default=None,
                    help=f"display-only true cut (default: true_threshold in {TRUE_THRESHOLD.name})")
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    votes, exported = load_votes(a.votes)
    true_thr = a.true_threshold if a.true_threshold is not None else float(json.loads(TRUE_THRESHOLD.read_text())["true_threshold"])
    frames = {lab: pd.read_parquet(pq).assign(post_id=lambda d: d.post_id.astype(str)) for lab, pq, _ in POOLS}
    media = {}
    for mp in a.media:
        for k, v in json.loads(mp.read_text()).items():
            media.setdefault(str(k), v)

    posts, unresolved, source_of = [], [], {}
    for pid in votes:
        for lab, _, _ in POOLS:
            g = frames[lab][frames[lab].post_id == pid]
            if len(g):
                posts.append(post_dict(pid, g, media.get(pid) or {}, lab == POOLS[0][0], true_thr))
                source_of[pid] = lab
                break
        else:
            unresolved.append(pid)
    # inconsistencies between what the reviewer saw (vote file) and what the post resolves to
    issues, moved = [], collections.Counter()
    for p in posts:
        ids = [c["id"] for c in p["claims"]]
        for who, v in votes[p["id"]].items():
            if v["cell"] and v["cell"].replace("unclear", "true") == p["cell"].replace("unclear", "true") and v["cell"] != p["cell"]:
                moved[f"{v['cell']} -> {p['cell']}"] += 1   # two-band vote now in its three-band cell (expected)
            elif v["cell"] and v["cell"] != p["cell"]:
                issues.append(f"{p['id']} {who} voted in {v['cell']}, resolves to {p['cell']} ({source_of[p['id']]})")
            if v["claims"] and v["claims"] != ids:
                issues.append(f"{p['id']} {who} claims {v['claims']} vs pool {ids}")
        cells = {v["cell"].replace("unclear", "true") for v in votes[p["id"]].values() if v["cell"]}
        if len(cells) > 1:
            issues.append(f"{p['id']} voted in different cells by different reviewers: {sorted(cells)}")
    for p in posts:
        p["votes"] = {w: {"v": v["v"], "c": v["c"]} for w, v in votes[p["id"]].items()}
    # most selects first, then newest (stable two-pass sort); the page keeps this order
    posts.sort(key=lambda p: p["date"], reverse=True)
    posts.sort(key=lambda p: -sum(v["v"] == "select" for v in p["votes"].values()))

    # evidence rows for the voted claims only
    want = {c["id"]: source_of[p["id"]] for p in posts for c in p["claims"]}
    evidence = {}
    for lab, _, doss in POOLS:
        todo = {c for c, s in want.items() if s == lab}
        with doss.open() as fh:
            for line in fh:
                if not todo:
                    break
                if '"claim_id"' not in line:
                    continue
                r = json.loads(line)
                if r.get("claim_id") in todo:
                    evidence[r["claim_id"]] = evidence_rows(r.get("docs") or [])
                    todo.discard(r["claim_id"])

    a.out.mkdir(parents=True, exist_ok=True)
    order = [p["id"] for p in posts]
    chunks = [order[i:i + a.chunk] for i in range(0, len(order), a.chunk)]
    by_id = {p["id"]: p for p in posts}
    for ci, pids in enumerate(chunks):
        (a.out / f"chunk_{ci:02d}.json").write_text(json.dumps({
            "media": {pid: {"avatar": shrink((media.get(pid) or {}).get("avatar"), 64, 60),
                            "media": [{"type": m["type"], "src": shrink(m["src"], a.media_width, a.media_quality)}
                                      for m in ((media.get(pid) or {}).get("media") or [])[:4]]} for pid in pids},
            "evidence": {c["id"]: evidence.get(c["id"], []) for pid in pids for c in by_id[pid]["claims"]}},
            ensure_ascii=False))
        for pid in pids:
            by_id[pid]["mc"] = ci

    wj = json.loads(a.weights.read_text())
    wv = list(map(float, wj["weights"].values()))
    sel = lambda p, w: p["votes"].get(w, {}).get("v") == "select"  # noqa: E731
    nsel = [sum(sel(p, w) for w in REVIEWERS) for p in posts]
    meta = {"built": datetime.now().strftime("%Y-%m-%d %H:%M"), "boundary": float(wj["threshold"]), "true_threshold": true_thr,
            "score_min": round(PAD_TO * min(wv), 2), "score_max": round(PAD_TO * max(wv), 2),
            "reviewers": list(REVIEWERS), "exported": exported, "n_chunks": len(chunks),
            "n_posts": len(posts), "n_not_current": sum(not p["in_current"] for p in posts),
            "selects": {w: sum(sel(p, w) for p in posts) for w in REVIEWERS},
            "sel2": sum(n >= 2 for n in nsel), "sel3": sum(n == 3 for n in nsel)}
    page = TEMPLATE.read_text().replace(
        "/*__DATA__*/", json.dumps({"meta": meta, "posts": posts}, ensure_ascii=False).replace("</", "<\\/"))
    (a.out / "vote_review.html").write_text(page)
    if a.standalone:
        allc = [json.loads((a.out / f"chunk_{ci:02d}.json").read_text()) for ci in range(len(chunks))]
        merged = {"media": {k: v for c in allc for k, v in c["media"].items()},
                  "evidence": {k: v for c in allc for k, v in c["evidence"].items()}}
        one = page.replace("/*__CHUNKS__*/null", json.dumps(merged, ensure_ascii=False).replace("</", "<\\/"))
        (a.out / "vote_review_standalone.html").write_text(one)

    print(f"voted posts {len(posts)} (not in current pool {meta['n_not_current']}), chunks {len(chunks)}")
    print(f"unresolved post_ids: {unresolved or 'none'}")
    print(f"votes moved from a two-band to a three-band cell: {dict(moved)}")
    print(f"vote-file inconsistencies: {len(issues)}")
    for s in issues:
        print("  " + s)
    print(f"header: selects {meta['selects']}, 2+ {meta['sel2']}, 3/3 {meta['sel3']}")
    print(f"page -> {a.out / 'vote_review.html'}")


if __name__ == "__main__":
    main()
