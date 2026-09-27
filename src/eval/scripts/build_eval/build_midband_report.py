"""Build a survey-claim pool report (markdown + HTML) from a scored pool.

Defaults are the mid-band pool; `--scored/--selection/--run/--candidates/--out-*` plus
`--title/--intro/--status-file/--issues-file` point it at another band (low band, 2026-09-11).

Everything in the summary tables is recomputed from `midband_scored.json`; the
example cards pull their evidence out of the run's `results-00.jsonl`. One
command regenerates both outputs after a re-run:

    uv run python eval/scripts/build_eval/build_midband_report.py
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import html
import json
import pathlib
import re
import statistics

FLAG_ORDER = ["5", "4", "3", "X", "I", "2", "1"]
FLAG_MEANING = {
    "5": "states / establishes the claim",
    "4": "points toward it",
    "3": "contested / mixed",
    "X": "on-claim context, no direction",
    "I": "irrelevant (or an empty slot)",
    "2": "points against it",
    "1": "contradicts / disproves it",
}
FLAG_PHRASE = {
    "5": "establish the claim",
    "4": "point toward it",
    "3": "contested",
    "X": "on-claim context only",
    "I": "irrelevant or empty",
    "2": "point against it",
    "1": "contradict it",
}
CLASS_ORDER = ["1", "2", "supported"]
CLASS_TITLE = {"1": "Class 1", "2": "Class 2", "supported": "Supported"}
LEAN_ORDER = ["Left", "Right"]
CELL_TARGET = 30

# Overridable per band: --title / --intro / --status-file / --issues-file.
TITLE = "Mid-Band Claim Pool"
INTRO = ("Scored pool for the survey's refutable posts, from the mid-band outlet capture.")

# Hand-read findings from the 2026-09-11 pass over 51 example posts. Narrative,
# not derived from the data: re-check it after any instrument change.
ISSUES_MD = """\
**This hand read is from the 2026-09-11 pre-fix pass** — it is what the two fixes above
were built from, and it has NOT been repeated on the re-run. Re-check it after the audit.

51 example posts were read by hand against the sentences the reader itself cited.
**28 of 51 are suspect**, and the errors are overwhelmingly in the low classes:
class 1 **14 / 16**, class 2 **12 / 16**, supported **1 / 16**, lowest-five **1 / 3**.

Caveat on the sample: within each class the examples are taken worst-score-first, so
this is the extreme tail rather than a random draw. It is the right sample for finding
failure modes and the wrong one for estimating a precision. The direction is still
unambiguous, and much worse than the 0.94 P(false | flag) from the timeline band audit.

In rough order of damage:

1. **Time-window blindness — about half of all errors.** The instrument has no notion of
   "as of the post date". Documents that *predate* the event contradict a breaking claim
   (eight February build-up stories used to refute a March war); documents that *postdate*
   it score a later outcome against an accurate contemporaneous report (a 28 July "he
   missed the deadline" story refuting an 8 July "he vowed to file"). A date relation
   between document and post, or an explicit as-of instruction in the read prompt, removes
   most of this bucket.
2. **"X said Y" is checked as "is Y true"** — roughly nine of the 28. The damage starts
   upstream: the query generator strips the attribution before searching, so the retrieved
   set is about Y and the reader never had a chance.
3. **Thin retrieval correlates with error.** Every claim in the sample that drew only two
   or three documents is an error, and 7.4% of claims retrieve nothing at all. A
   minimum-document gate (or an on-topic check) is cheap.
4. **Fact-check restatement read as assertion — Left only.** factcheck.org, politifact.com,
   AP "FACT FOCUS" and snopes are the deciding contradicting document in three Left posts,
   always because the article restates the claim in order to debunk it. It did not fire on
   the Right sample, where the correctly-scored class-1 posts are long-standing partisan
   claims with dedicated fact-checks in the index.
5. **Scope and numeric mismatch treated as contradiction.** A World Cup carve-out for
   *athletes* used to refute a claim about *fans*; a three-month disclosure of 3,700 trades
   used to refute a six-month claim of 7,000+; a full-year layoff total used to refute a
   December hiring claim.
6. **Claim decomposition is not audited.** A post can be rated supported on a trivially
   true background clause while its newsworthy assertion is never extracted (a WashTimes
   post scores +29.4 on "the Abraham Accords were signed in 2020"), and a metonym can be
   literalised ("Trump knocks off Massie" into "Trump defeated Massie"), which manufactures
   a false claim. The min-over-claims rule hides the first and amplifies the second.
7. **Duplicate posts survive into the pool.** Two MotherJones posts (11 and 23 June) are
   near-identical text and get near-identical wrong scores. Dedupe before sampling.

**What holds up.** The supported class is convincing: 15 of 16 read cleanly, 9-10 documents
apiece, near-unanimous flag 5, and the primary source usually in the set (whitehouse.gov,
af.mil, dhs.gov, vatican.va, UN Digital Library, Ballotpedia). The class-1 posts that *are*
right are exactly what a survey wants: long-standing partisan false claims with dedicated
fact-checks.

### Next steps

1. **Decide the band before hand-picking.** Pooling classes 1 and 2 into one "dubious" band
   clears every content-lean cell with no new data. Keeping them separate leaves three short.
2. **Triage is not optional.** Budget for reading all ~180 low-class posts, not for sampling
   from them.
3. **Fix the two cheap upstream things first**, then re-run (one instrument pass, ~$4): a date
   relation between document and post date, and keeping the attribution in the query for
   "X said Y" claims. Between them they account for most of the 28 suspect posts.
4. **Add a minimum-document gate** (or an on-topic check) and re-rate.
5. **Audit claim decomposition separately** — the trivial-background and literalised-metonym
   shapes are both invisible to the current funnel.
6. **Dedupe near-identical posts** before sampling.
7. **Only then consider widening to NG 30-50**, and only if the triaged counts are still short.
"""


# ---------------------------------------------------------------- loading


def load_inputs(args):
    scored = json.loads(pathlib.Path(args.scored).read_text())
    selection = json.loads(pathlib.Path(args.selection).read_text())
    cand = json.loads(pathlib.Path(args.candidates).read_text())
    cand_by_id = {c["claim_id"]: c for c in cand["claims"]}
    return scored, selection, cand_by_id, cand


def stream_run(run_path, wanted):
    """One pass over the run file: keep wanted claims, total the cost."""
    kept, cost, n = {}, 0.0, 0
    with open(run_path) as fh:
        for line in fh:
            rec = json.loads(line)
            n += 1
            cost += rec.get("cost") or 0.0
            cid = rec.get("review_url")
            if cid in wanted:
                kept[cid] = rec
    return kept, cost, n


# ---------------------------------------------------------------- selection


def pick_examples(posts_by_id, selection, per_group):
    """Worst score first, spread across handles, up to `per_group` per cell."""
    groups = {}
    for cls in CLASS_ORDER:
        for lean in LEAN_ORDER:
            ids = selection["by_lean_class"].get(lean, {}).get(cls, [])
            posts = [posts_by_id[p] for p in ids if p in posts_by_id]
            posts.sort(key=lambda p: p["min_score"], reverse=(cls == "supported"))
            by_handle = collections.OrderedDict()
            for p in posts:
                by_handle.setdefault(p["handle"], []).append(p)
            # the hand audit covers every class-1/2 post, so show all of them
            cap = per_group if cls == "supported" else max(per_group, len(posts))
            picked, queues = [], list(by_handle.values())
            while queues and len(picked) < cap:
                for q in list(queues):
                    if len(picked) >= cap:
                        break
                    picked.append(q.pop(0))
                    if not q:
                        queues.remove(q)
            groups[(cls, lean)] = picked
    return groups


# ---------------------------------------------------------------- evidence


def parse_doc_date(raw):
    if not raw:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def post_date(iso):
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).date()


def doc_quote(doc, limit=220):
    """What the reader cited: its reason code, else the sentences it pointed at."""
    read = doc.get("read") or {}
    reason = (read.get("reason") or "").strip()
    if reason:
        return reason
    sents = dict(zip(doc.get("sent_ids") or [], doc.get("sents") or []))
    cited = [sents[i] for i in (read.get("evidence") or []) if i in sents]
    if not cited:
        return ""
    return trim(" ".join(cited), limit)


def region_text(doc, limit=400):
    sents = dict(zip(doc.get("sent_ids") or [], doc.get("sents") or []))
    spans = ((doc.get("prep") or {}).get("read_regions")) or []
    out = []
    for span in spans:
        lo, hi = span.get("span", [None, None])
        if lo is None:
            continue
        out.extend(sents[i] for i in sorted(sents) if lo <= i <= hi)
    if not out:
        out = [sents[i] for i in sorted(sents)]
    return trim(" ".join(out), limit)


def trim(text, limit):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def evidence_summary(claim, docs, p_date):
    """1-3 plain sentences, built from the flags and the deciding documents."""
    if not docs:
        return [
            "No documents were retrieved for this claim, so it scores as ten silences."
        ]
    counts = collections.Counter((d.get("read") or {}).get("direction") for d in docs)
    parts = [
        f"{counts[f]} {FLAG_PHRASE[f]}" for f in FLAG_ORDER if counts.get(f)
    ]
    sents = [f"{len(docs)} documents were read: " + ", ".join(parts) + "."]

    rating = claim["rating"]
    if rating in ("1", "2"):
        wanted, direction = ["1", "2"], "against"
    elif rating == "supported":
        wanted, direction = ["5", "4"], "for"
    else:
        wanted, direction = ["1", "2", "5", "4"], "either way"
    deciding = [
        d for f in wanted for d in docs if (d.get("read") or {}).get("direction") == f
    ]
    if not deciding:
        sents.append(
            f"No document carries a flag {direction}; the score comes from the padding."
        )
        return sents

    shown = deciding[:3]
    bits = []
    for d in shown:
        flag = (d.get("read") or {}).get("direction")
        quote = doc_quote(d, 180)
        date = d.get("date") or "no date"
        bits.append(
            f"{d.get('domain')} ({date}, flag {flag})"
            + (f' "{quote}"' if quote else " (no sentences cited)")
        )
    more = f" and {len(deciding) - len(shown)} more" if len(deciding) > len(shown) else ""
    sents.append(
        f"The rating rests on {len(deciding)} document(s) flagged {direction}: "
        + "; ".join(bits)
        + more
        + "."
    )

    dated = [(d, parse_doc_date(d.get("date"))) for d in deciding]
    gaps = [(dd - p_date).days for _, dd in dated if dd]
    undated = sum(1 for _, dd in dated if not dd)
    if gaps:
        before = sum(1 for g in gaps if g < 0)
        after = sum(1 for g in gaps if g > 0)
        lo, hi = min(gaps), max(gaps)
        span = (
            f"from {abs(lo)} days before to {hi} days after the post"
            if lo < 0 < hi
            else (
                f"{abs(hi)} to {abs(lo)} days before the post"
                if hi <= 0
                else f"{lo} to {hi} days after the post"
            )
        )
        sents.append(
            f"Dates: {before} predate the post ({p_date}), {after} postdate it"
            + (f", {undated} undated" if undated else "")
            + f" — {span}."
        )
    elif undated:
        sents.append(
            f"Dates: none of the {undated} deciding documents carries a date, so the "
            f"post date ({p_date}) cannot be checked against them."
        )
    return sents


# ---------------------------------------------------------------- summaries


def summarise(scored):
    posts, claims = scored["posts"], scored["claims"]
    s = {}
    s["post_lean_class"] = collections.Counter((p["lean"], p["rating"]) for p in posts)
    s["claim_lean_class"] = collections.Counter(
        (c["lean"], c["rating"]) for c in claims
    )
    s["handle"] = collections.Counter((p["handle"], p["rating"]) for p in posts)
    s["handle_meta"] = {}
    for p in posts:
        s["handle_meta"][p["handle"]] = (p["lean"], p["ng_score"])
    s["claim_content"] = collections.Counter(
        (c["rating"], c["content_lean"]) for c in claims
    )
    driving = {c["claim_id"]: c for c in claims}
    s["cells"] = collections.Counter()
    s["post_content"] = collections.Counter()
    for p in posts:
        d = driving.get(p["driving_claim_id"])
        cl = d["content_lean"] if d else "neutral"
        s["cells"][(p["lean"], cl, p["rating"])] += 1
        s["post_content"][(p["rating"], cl)] += 1
    s["quantiles"] = {}
    for lean in LEAN_ORDER:
        vals = sorted(p["min_score"] for p in posts if p["lean"] == lean)
        s["quantiles"][lean] = (len(vals), [pct(vals, q) for q in (1, 5, 10, 25, 50, 75, 90)])
    flags = collections.Counter()
    ndocs = []
    for c in claims:
        flags.update(c["flags"])
        ndocs.append(c["n_docs"])
    s["flags"] = flags
    s["n_slots"] = sum(flags.values())
    s["docs_mean"] = sum(ndocs) / len(ndocs)
    s["docs_median"] = statistics.median(ndocs)
    s["docs_zero"] = sum(1 for x in ndocs if x == 0)
    s["n_posts"] = len(posts)
    s["n_claims"] = len(claims)
    return s


def pct(sorted_vals, q):
    """Linear-interpolated percentile (numpy's default), so the numbers match the clog."""
    if not sorted_vals:
        return 0.0
    pos = q / 100 * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def fmt_score(x):
    return f"{x:+.2f}"


# ---------------------------------------------------------------- rendering


def build_rows(s):
    """Table rows shared by both renderers: (caption, header, rows)."""
    tables = []

    hdr = ["lean", "1", "2", "middle", "supported", "total"]
    rows = []
    for lean in LEAN_ORDER:
        cells = [s["post_lean_class"][(lean, c)] for c in ("1", "2", "middle", "supported")]
        rows.append([lean] + [str(v) for v in cells] + [str(sum(cells))])
    tot = [
        sum(s["post_lean_class"][(l, c)] for l in LEAN_ORDER)
        for c in ("1", "2", "middle", "supported")
    ]
    rows.append(["all"] + [str(v) for v in tot] + [str(sum(tot))])
    tables.append(("Posts by outlet lean x class", hdr, rows))

    rows = []
    for lean in LEAN_ORDER:
        cells = [s["claim_lean_class"][(lean, c)] for c in ("1", "2", "middle", "supported")]
        rows.append([lean] + [str(v) for v in cells] + [str(sum(cells))])
    tot = [
        sum(s["claim_lean_class"][(l, c)] for l in LEAN_ORDER)
        for c in ("1", "2", "middle", "supported")
    ]
    rows.append(["all"] + [str(v) for v in tot] + [str(sum(tot))])
    tables.append(("Claims by outlet lean x class", hdr, rows))

    hdr = ["handle", "lean", "NG", "posts", "1", "2", "middle", "supported", "flag rate"]
    rows = []
    handles = sorted(
        s["handle_meta"], key=lambda h: (s["handle_meta"][h][0], -sum(
            s["handle"][(h, c)] for c in ("1", "2", "middle", "supported")))
    )
    for h in handles:
        lean, ng = s["handle_meta"][h]
        cells = [s["handle"][(h, c)] for c in ("1", "2", "middle", "supported")]
        n = sum(cells)
        rate = (cells[0] + cells[1]) / n * 100 if n else 0
        rows.append([h, lean, f"{ng:g}", str(n)] + [str(v) for v in cells] + [f"{rate:.1f}%"])
    tables.append(("Posts per handle x class", hdr, rows))

    hdr = ["class", "pro_dem", "pro_rep", "neutral", "total"]
    rows = []
    for cls in ["1", "2", "supported", "middle"]:
        cells = [s["claim_content"][(cls, cl)] for cl in ("pro_dem", "pro_rep", "neutral")]
        rows.append([cls] + [str(v) for v in cells] + [str(sum(cells))])
    tables.append(
        ("Content lean of the CLAIM, within each class (who the claim helps if true)", hdr, rows)
    )

    hdr = ["outlet lean", "content lean", "1", "2", "supported"]
    rows = []
    for lean in LEAN_ORDER:
        for cl in ("pro_dem", "pro_rep", "neutral"):
            rows.append(
                [lean, cl] + [str(s["cells"][(lean, cl, c)]) for c in ("1", "2", "supported")]
            )
    tables.append(("Posts by outlet lean x content lean x class (the survey's cells)", hdr, rows))

    hdr = ["content lean", "class 1", "class 2", "1+2 pooled", "supported"]
    rows = []
    for cl in ("pro_dem", "pro_rep", "neutral"):
        c1 = s["post_content"][("1", cl)]
        c2 = s["post_content"][("2", cl)]
        sup = s["post_content"][("supported", cl)]
        def mark(v):
            return f"**{v}** (short {CELL_TARGET - v})" if v < CELL_TARGET else str(v)
        rows.append([cl, mark(c1), mark(c2), mark(c1 + c2), mark(sup)])
    tables.append(
        (f"Distance from a usable cell (target {CELL_TARGET} candidates per cell, posts)", hdr, rows)
    )

    hdr = ["lean", "n", "p1", "p5", "p10", "p25", "p50", "p75", "p90"]
    rows = []
    for lean in LEAN_ORDER:
        n, qs = s["quantiles"][lean]
        rows.append([lean, str(n)] + [fmt_score(q) for q in qs])
    tables.append(("Post min-score distribution per lean", hdr, rows))

    return tables


def md_table(hdr, rows):
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def html_table(hdr, rows):
    out = ["<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in hdr) + "</tr></thead><tbody>"]
    for r in rows:
        cells = "".join(f"<td>{md_bold_to_html(c)}</td>" for c in r)
        out.append(f"<tr>{cells}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def md_bold_to_html(text):
    esc = html.escape(str(text))
    while "**" in esc:
        esc = esc.replace("**", "<strong>", 1)
        if "**" in esc:
            esc = esc.replace("**", "</strong>", 1)
        else:
            esc += "</strong>"
    return esc


def header_facts(scored, manifest, run_cost, run_path, s):
    run_dir = pathlib.Path(run_path).parent
    try:
        start = dt.datetime.fromtimestamp((run_dir / "manifest.json").stat().st_mtime)
        end = dt.datetime.fromtimestamp(pathlib.Path(run_path).stat().st_mtime)
        window = f"{start:%Y-%m-%d %H:%M}-{end:%H:%M} ({(end - start).total_seconds() / 60:.0f} min)"
    except OSError:
        window = "unknown"
    pr = manifest.get("prompts", {})
    return {
        "run": run_path,
        "window": window,
        "cost": run_cost,
        "n_claims_run": manifest.get("n_claims", s["n_claims"]),
        "reader": f"read-{pr.get('read', '?')} on {pr.get('read_model', '?')}",
        "query": f"{pr.get('query', '?')} on {pr.get('query_model', '?')}",
        "weights": scored["weights"],
        "cuts": scored["cuts"],
    }


def render_examples(groups, claims_by_id, cand_by_id, run_by_claim, fmt):
    """fmt is 'md' or 'html'. Returns the section body."""
    out = []
    for cls in CLASS_ORDER:
        for lean in LEAN_ORDER:
            posts = groups[(cls, lean)]
            title = f"{CLASS_TITLE[cls]} — {lean} ({len(posts)} posts)"
            out.append(f"<h3>{html.escape(title)}</h3>" if fmt == "html" else f"### {title}\n")
            for p in posts:
                out.append(render_post(p, claims_by_id, cand_by_id, run_by_claim, fmt))
    return "\n".join(out)


def render_post(p, claims_by_id, cand_by_id, run_by_claim, fmt):
    p_date = post_date(p["created_at"])
    head = f"@{p['handle']} · {p_date} · class {p['rating']} · score {fmt_score(p['min_score'])}"
    blocks = []
    if fmt == "html":
        blocks.append(f'<div class="card"><div class="cardhead">{html.escape(head)}</div>')
        blocks.append(
            '<div class="blk"><div class="lab">POST</div>'
            f'<div class="meta">@{html.escape(p["handle"])} · {p_date} · '
            f'<a href="{html.escape(p["url"])}">{html.escape(p["url"])}</a> · NG {p["ng_score"]:g}</div>'
            f'<div class="posttext">{html.escape(p["post_text"])}</div></div>'
        )
        if p.get("audit"):
            blocks.append(
                '<div class="blk"><div class="lab">HAND AUDIT</div>'
                f'<div class="summary">{html.escape(p["audit"])}</div></div>'
            )
    else:
        blocks.append(f"#### {head}\n")
        blocks.append(f"**POST** — @{p['handle']} · {p_date} · [{p['url']}]({p['url']}) · NG {p['ng_score']:g}\n")
        blocks.append("> " + p["post_text"].replace("\n", "\n> ") + "\n")
        if p.get("audit"):
            blocks.append(f"**HAND AUDIT** — {p['audit']}\n")

    for cid in p["claim_ids"]:
        c = claims_by_id.get(cid)
        if not c:
            continue
        cand = cand_by_id.get(cid, {})
        ctype = cand.get("type", "unknown")
        rec = run_by_claim.get(cid) or {}
        docs = rec.get("results") or []
        summary = evidence_summary(c, docs, p_date)
        if fmt == "html":
            blocks.append(
                '<div class="blk claim"><div class="lab">CLAIM</div>'
                f'<div class="claimtext">{html.escape(c["claim"])}</div>'
                f'<div class="meta">type <b>{html.escape(ctype)}</b> · score <b>{fmt_score(c["score"])}</b>'
                f' · rating <b>{html.escape(str(c["rating"]))}</b> · content lean {html.escape(c["content_lean"])}'
                f' · {len(docs)} documents retrieved · query: {html.escape(rec.get("query") or "n/a")}</div>'
                f'<div class="lab">EVIDENCE SUMMARY</div>'
                f'<div class="summary">{html.escape(" ".join(summary))}</div>'
                + evidence_details(docs, p_date, "html")
                + "</div>"
            )
        else:
            blocks.append(
                f"**CLAIM** ({ctype}, score {fmt_score(c['score'])}, rating {c['rating']}, "
                f"content lean {c['content_lean']}, {len(docs)} documents retrieved)\n\n"
                f"> {c['claim']}\n\n"
                f"*Query:* `{rec.get('query') or 'n/a'}`\n\n"
                f"**Evidence summary.** {' '.join(summary)}\n\n"
                + evidence_details(docs, p_date, "md")
            )

    if fmt == "html":
        blocks.append("</div>")
    return "\n".join(blocks)


def evidence_details(docs, p_date, fmt):
    if not docs:
        return ""
    hdr = ["#", "domain", "date", "rel", "flag", "reader's quote / reason", "region read"]
    rows = []
    for d in docs:
        read = d.get("read") or {}
        flag = read.get("direction") or "-"
        url = d.get("url") or ""
        dom = d.get("domain") or ""
        dd = parse_doc_date(d.get("date"))
        datestr = d.get("date") or "—"
        if dd:
            gap = (dd - p_date).days
            datestr += f" ({gap:+d}d)"
        if fmt == "html":
            rows.append([
                str(d.get("rank", "")),
                f'<a href="{html.escape(url)}">{html.escape(dom)}</a>',
                html.escape(datestr),
                html.escape(str(d.get("rel") or "")),
                f'<span class="flag f{flag}">{html.escape(flag)}</span>',
                html.escape(doc_quote(d, 300) or "—"),
                html.escape(region_text(d) or "—"),
            ])
        else:
            rows.append([
                str(d.get("rank", "")),
                f"[{dom}]({url})",
                datestr,
                str(d.get("rel") or ""),
                flag,
                (doc_quote(d, 300) or "—").replace("|", "\\|"),
                (region_text(d) or "—").replace("|", "\\|"),
            ])
    if fmt == "html":
        body = "<table class='ev'><thead><tr>" + "".join(f"<th>{h}</th>" for h in hdr) + "</tr></thead><tbody>"
        body += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        body += "</tbody></table>"
        return f"<details><summary>Full evidence table ({len(docs)} documents)</summary>{body}</details>"
    return (
        f"<details>\n<summary>Full evidence table ({len(docs)} documents)</summary>\n\n"
        + md_table(hdr, rows)
        + "\n\n</details>\n"
    )


STATUS = (
    "Status: re-run under the attribution fix (the query keeps the speaker for \"X said Y\" "
    "claims, leaked post-ceiling documents are dropped, and the reader is given each "
    "document's publication date), then filtered by a 7-day selection floor — a document "
    "published more than a week before the post can no longer contradict it. **All 53 class-1/2 "
    "posts have now been hand-read** (2026-09-11): 13 of 53 carry a claim that is actually false "
    "or distorted and **9 are survey-usable** (5 pro_dem from Left outlets, 4 pro_rep from Right "
    "outlets, 0 neutral). Each low-class card below carries its verdict under HAND AUDIT, in the "
    "form `false-judgement | driver | survey suitability: why`. The counts in the tables are the "
    "pool as scored, not the post-triage yield."
)


def render_md(facts, tables, examples, s):
    w = facts["weights"]
    c = facts["cuts"]
    parts = [
        f"# {TITLE}",
        "",
        f"{INTRO} Generated {dt.date.today()}.",
        "",
        "## What was run",
        "",
        f"- Run `{facts['run']}` — {facts['n_claims_run']} claims over {s['n_posts']} posts, "
        f"{facts['window']}, **${facts['cost']:.3f}**.",
        f"- Reader **{facts['reader']}**; query generator {facts['query']}; Serper top-10 with the "
        f"originating outlet excluded and a date ceiling at the post's own date.",
        "- Each retrieved document gets one of seven flags: "
        + " · ".join(f"**{f}** {FLAG_MEANING[f]}" for f in FLAG_ORDER)
        + ".",
        f"- Each claim is scored over ten slots, padded with silence, as a log-odds sum with the "
        f"pinned weights "
        + " · ".join(f"{f} {w[f]:+.3f}" for f in FLAG_ORDER)
        + ".",
        f"- Cuts pinned at **{c['cut_1']:+.3f} / {c['cut_2']:+.3f} / {c['cut_4']:+.3f} / "
        f"{c['cut_5']:+.3f}**: class 1 at or below {c['cut_1']:+.3f}, class 2 between "
        f"{c['cut_1']:+.3f} and {c['cut_2']:+.3f}, supported above {c['cut_5']:+.3f}, everything "
        f"else middle and discarded. A post's rating is the minimum over its kept claims.",
        f"- Retrieval yield: {s['docs_mean']:.2f} documents per claim (median "
        f"{s['docs_median']:g}); {s['docs_zero']} claims ({s['docs_zero'] / s['n_claims'] * 100:.1f}%) "
        f"retrieved nothing and score as ten silences.",
        "",
        STATUS,
        "",
        "Jump to: [Examples](#examples) · [Summary](#summary) · [Issues](#issues)",
        "",
        "## Examples",
        "",
        "Up to eight posts per class x lean, worst score first, spread across handles. Each post "
        "card carries the post text, its extracted claims, a plain-language evidence summary and "
        "the full evidence table behind a dropdown.",
        "",
        examples,
        "",
        "## Summary",
        "",
    ]
    for caption, hdr, rows in tables:
        parts += [f"### {caption}", "", md_table(hdr, rows), ""]
    flags = s["flags"]
    parts += [
        "### Flag mix and document yield",
        "",
        f"- claims scored {s['n_claims']}; posts {s['n_posts']}; mean claims/post "
        f"{s['n_claims'] / s['n_posts']:.2f}",
        f"- documents read per claim: mean {s['docs_mean']:.2f}, median {s['docs_median']:g}, "
        f"zero-document claims {s['docs_zero']} ({s['docs_zero'] / s['n_claims'] * 100:.1f}%)",
        "- flag mix over all "
        + f"{s['n_slots']} padded slots: "
        + " · ".join(
            f"{f} {flags[f]} ({flags[f] / s['n_slots'] * 100:.1f}%)" for f in FLAG_ORDER
        ),
        "",
        "## Issues",
        "",
        ISSUES_MD,
        "",
        "## Provenance",
        "",
        f"Run `{facts['run']}`. Weights and cuts from `eval/data/populations/refit_results.json` "
        f"cell `graded_urn_7flag`. Scoring `eval/scripts/build_eval/select_midband.py`. This page "
        f"`eval/scripts/build_eval/build_midband_report.py`. Session log `src/clog/110926.md`.",
        "",
    ]
    return "\n".join(parts)


CSS = """
:root{--bg:#fff;--fg:#16181c;--mut:#5b6169;--line:#d8dce1;--soft:#f5f6f8;--card:#fbfbfc;
--accent:#1f3d7a;--claim:#eef1f6;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14161a;--fg:#e7e9ec;
--mut:#9aa1ab;--line:#343a42;--soft:#1b1e23;--card:#1a1d22;--accent:#9ec1ff;--claim:#20252d;}}
:root[data-theme="dark"]{--bg:#14161a;--fg:#e7e9ec;--mut:#9aa1ab;--line:#343a42;--soft:#1b1e23;
--card:#1a1d22;--accent:#9ec1ff;--claim:#20252d;}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,
"Segoe UI",Helvetica,Arial,sans-serif;margin:0;}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px;}
h1{font-size:26px;margin:0 0 6px;}
h2{font-size:21px;margin:38px 0 10px;padding-top:10px;border-top:1px solid var(--line);}
h3{font-size:17px;margin:26px 0 8px;}
p,li{max-width:none;}
a{color:var(--accent);}
code{background:var(--soft);padding:1px 4px;border-radius:3px;font-size:13px;}
.nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:8px 0;margin-bottom:8px;z-index:5;}
.nav a{margin-right:16px;text-decoration:none;font-weight:600;}
.status{background:var(--soft);border-left:3px solid var(--accent);padding:10px 12px;margin:14px 0;}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13.5px;}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top;}
th{background:var(--soft);font-weight:600;}
.card{border:1px solid var(--line);border-radius:6px;margin:16px 0;background:var(--card);
overflow:hidden;}
.cardhead{background:var(--soft);border-bottom:1px solid var(--line);padding:8px 12px;
font-weight:600;font-size:14px;}
.blk{padding:10px 12px;border-bottom:1px solid var(--line);}
.blk:last-child{border-bottom:none;}
.claim{background:var(--bg);}
.lab{font-size:11px;letter-spacing:.08em;color:var(--mut);font-weight:700;margin-bottom:4px;}
.meta{font-size:12.5px;color:var(--mut);margin:4px 0;word-break:break-word;}
.posttext{white-space:pre-wrap;background:var(--soft);border:1px solid var(--line);
border-radius:4px;padding:8px 10px;}
.claimtext{background:var(--claim);border-left:3px solid var(--accent);padding:8px 10px;
border-radius:0 4px 4px 0;font-weight:500;}
.summary{margin-top:2px;}
details{margin-top:8px;}
summary{cursor:pointer;font-size:13px;color:var(--mut);}
table.ev{font-size:12.5px;table-layout:fixed;}
table.ev th:nth-child(1),table.ev td:nth-child(1){width:28px;}
table.ev th:nth-child(2),table.ev td:nth-child(2){width:120px;}
table.ev th:nth-child(3),table.ev td:nth-child(3){width:110px;}
table.ev th:nth-child(4),table.ev td:nth-child(4){width:70px;}
table.ev th:nth-child(5),table.ev td:nth-child(5){width:44px;}
table.ev td{word-break:break-word;}
.flag{display:inline-block;min-width:18px;text-align:center;border:1px solid var(--line);
border-radius:3px;padding:0 4px;font-weight:700;}
.f5,.f4{background:rgba(40,120,60,.16);}
.f1,.f2{background:rgba(170,50,50,.18);}
.f3{background:rgba(180,130,20,.18);}
ul{padding-left:20px;}
"""


def render_html(facts, tables, examples, s):
    w, c = facts["weights"], facts["cuts"]
    flags = s["flags"]
    body = []
    body.append('<div class="wrap">')
    body.append(f"<h1>{html.escape(TITLE)}</h1>")
    body.append(
        f'<p class="meta">{html.escape(INTRO)} Generated {dt.date.today()}.</p>'
    )
    body.append(
        '<div class="nav"><a href="#examples">Examples</a><a href="#summary">Summary</a>'
        '<a href="#issues">Issues</a></div>'
    )
    body.append("<h2>What was run</h2><ul>")
    body.append(
        f"<li>Run <code>{html.escape(facts['run'])}</code> — {facts['n_claims_run']} claims over "
        f"{s['n_posts']} posts, {html.escape(facts['window'])}, <b>${facts['cost']:.3f}</b>.</li>"
    )
    body.append(
        f"<li>Reader <b>{html.escape(facts['reader'])}</b>; query generator "
        f"{html.escape(facts['query'])}; Serper top-10 with the originating outlet excluded and a "
        f"date ceiling at the post's own date.</li>"
    )
    body.append(
        "<li>Seven flags: "
        + " · ".join(f"<b>{f}</b> {FLAG_MEANING[f]}" for f in FLAG_ORDER)
        + ".</li>"
    )
    body.append(
        "<li>Ten slots per claim, padded with silence, log-odds sum with pinned weights "
        + " · ".join(f"{f} {w[f]:+.3f}" for f in FLAG_ORDER)
        + ".</li>"
    )
    body.append(
        f"<li>Cuts pinned at <b>{c['cut_1']:+.3f} / {c['cut_2']:+.3f} / {c['cut_4']:+.3f} / "
        f"{c['cut_5']:+.3f}</b>: class 1 at or below {c['cut_1']:+.3f}, class 2 between "
        f"{c['cut_1']:+.3f} and {c['cut_2']:+.3f}, supported above {c['cut_5']:+.3f}, everything "
        f"else middle. Post rating = minimum over its kept claims.</li>"
    )
    body.append(
        f"<li>Retrieval yield {s['docs_mean']:.2f} documents per claim (median "
        f"{s['docs_median']:g}); {s['docs_zero']} claims "
        f"({s['docs_zero'] / s['n_claims'] * 100:.1f}%) retrieved nothing.</li></ul>"
    )
    body.append(f'<div class="status">{html.escape(STATUS)}</div>')

    body.append('<h2 id="examples">Examples</h2>')
    body.append(
        "<p>Up to eight posts per class x lean, worst score first, spread across handles. Each "
        "card carries the post text, its extracted claims, a plain-language evidence summary and "
        "the full evidence table behind a dropdown.</p>"
    )
    body.append(examples)

    body.append('<h2 id="summary">Summary</h2>')
    for caption, hdr, rows in tables:
        body.append(f"<h3>{html.escape(caption)}</h3>")
        body.append(html_table(hdr, rows))
    body.append("<h3>Flag mix and document yield</h3><ul>")
    body.append(
        f"<li>claims scored {s['n_claims']}; posts {s['n_posts']}; mean claims/post "
        f"{s['n_claims'] / s['n_posts']:.2f}</li>"
    )
    body.append(
        f"<li>documents read per claim: mean {s['docs_mean']:.2f}, median {s['docs_median']:g}, "
        f"zero-document claims {s['docs_zero']} "
        f"({s['docs_zero'] / s['n_claims'] * 100:.1f}%)</li>"
    )
    body.append(
        f"<li>flag mix over all {s['n_slots']} padded slots: "
        + " · ".join(f"{f} {flags[f]} ({flags[f] / s['n_slots'] * 100:.1f}%)" for f in FLAG_ORDER)
        + "</li></ul>"
    )

    body.append('<h2 id="issues">Issues</h2>')
    body.append(issues_html())
    body.append("<h2>Provenance</h2>")
    body.append(
        f"<p>Run <code>{html.escape(facts['run'])}</code>. Weights and cuts from "
        "<code>eval/data/populations/refit_results.json</code> cell <code>graded_urn_7flag</code>. "
        "Scoring <code>eval/scripts/build_eval/select_midband.py</code>. This page "
        "<code>eval/scripts/build_eval/build_midband_report.py</code>. Session log "
        "<code>src/clog/110926.md</code>.</p>"
    )
    body.append("</div>")
    return (
        f"<title>{html.escape(TITLE)}</title>\n<style>"
        + CSS
        + "</style>\n"
        + "\n".join(body)
    )


def issues_html():
    """Minimal markdown -> HTML for the hand-written issues block."""
    out, in_list = [], False
    para = []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    for raw in ISSUES_MD.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            if in_list:
                out.append("</ol>")
                in_list = False
            continue
        if line.startswith("### "):
            flush()
            if in_list:
                out.append("</ol>")
                in_list = False
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
            continue
        stripped = line.strip()
        m = re.match(r"(\d+)\.\s+(.*)", stripped)
        if m:
            flush()
            if not in_list:
                out.append("<ol>")
                in_list = True
            out.append("<li>" + inline(m.group(2)) + "</li>")
            continue
        if in_list and line.startswith("   "):
            if out and out[-1].endswith("</li>"):
                out[-1] = out[-1][: -len("</li>")] + " " + inline(stripped) + "</li>"
            continue
        para.append(stripped)
    flush()
    if in_list:
        out.append("</ol>")
    return "\n".join(out)


def inline(text):
    esc = html.escape(text)
    for src, a, b in (("**", "<strong>", "</strong>"), ("*", "<em>", "</em>")):
        parts = esc.split(src)
        if len(parts) > 1:
            rebuilt = parts[0]
            for i, seg in enumerate(parts[1:]):
                rebuilt += (a if i % 2 == 0 else b) + seg
            if len(parts) % 2 == 0:
                rebuilt += b
            esc = rebuilt
    esc = esc.replace("`", "")
    return esc


# ---------------------------------------------------------------- main


def main():
    global TITLE, INTRO, STATUS, ISSUES_MD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default="eval/data/survey_claims/midband_scored.json")
    ap.add_argument("--selection", default="eval/data/survey_claims/midband_selection.json")
    ap.add_argument("--run", default="eval/data/urn_runs/e4_midband_fix/results-00.jsonl")
    ap.add_argument("--candidates", default="eval/data/survey_claims/candidate_claims_midband.json")
    ap.add_argument("--out-md", default="eval/data/survey_claims/midband_selection_report.md")
    ap.add_argument("--out-html", default="eval/data/survey_claims/midband_selection_report.html")
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--intro", default=INTRO)
    ap.add_argument("--status-file", default=None, help="text file replacing the STATUS paragraph")
    ap.add_argument("--issues-file", default=None, help="markdown file replacing the Issues block")
    args = ap.parse_args()
    TITLE, INTRO = args.title, args.intro
    if args.status_file:
        STATUS = pathlib.Path(args.status_file).read_text().strip()
    if args.issues_file:
        ISSUES_MD = pathlib.Path(args.issues_file).read_text().strip()

    scored, selection, cand_by_id, _ = load_inputs(args)
    posts_by_id = {p["post_id"]: p for p in scored["posts"]}
    claims_by_id = {c["claim_id"]: c for c in scored["claims"]}

    groups = pick_examples(posts_by_id, selection, args.per_group)
    wanted = {cid for posts in groups.values() for p in posts for cid in p["claim_ids"]}
    print(f"examples: {sum(len(v) for v in groups.values())} posts, {len(wanted)} claims")

    run_by_claim, run_cost, n_recs = stream_run(args.run, wanted)
    print(f"run: {n_recs} records, ${run_cost:.3f}, {len(run_by_claim)} example claims matched")

    manifest_path = pathlib.Path(args.run).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    s = summarise(scored)
    facts = header_facts(scored, manifest, run_cost, args.run, s)
    tables = build_rows(s)

    md = render_md(
        facts, tables, render_examples(groups, claims_by_id, cand_by_id, run_by_claim, "md"), s
    )
    page = render_html(
        facts, tables, render_examples(groups, claims_by_id, cand_by_id, run_by_claim, "html"), s
    )
    pathlib.Path(args.out_md).write_text(md)
    pathlib.Path(args.out_html).write_text(page)
    print(f"wrote {args.out_md} ({len(md)} chars) and {args.out_html} ({len(page)} chars)")


if __name__ == "__main__":
    main()
