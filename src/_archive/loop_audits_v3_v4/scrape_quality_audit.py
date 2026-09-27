"""Read-only audit of scrape/evidence quality from existing verify traces.

No live APIs, no re-scraping. Reads:
  - eval/data/survey_claims/runs/{dev50_lowng_v3,dev50_main_v3,dev50_main_v3_rerun}/results-*.jsonl
  - pipeline/.cache/scrape/*.json  (READ-ONLY, keyed sha256(url))
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.verify_tweet_claims import _domain_of, clean_text, sentences, numbered_block, _JUNK, CFG
from pipeline.search import newsguard_score_map

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["dev50_lowng_v3", "dev50_main_v3", "dev50_main_v3_rerun"]
CACHE = ROOT / "pipeline" / ".cache" / "scrape"
SCORES = newsguard_score_map()

REASONS = ["scrape-failed", "too-short", "junk-page", "republication",
           "near-dup", "too-thin", "dup-url", "dup-domain", "origin"]


def tier_of(url: str) -> tuple[int, str]:
    """(rank_hits tier, finer bucket label)."""
    dom = _domain_of(url)
    if dom.endswith((".gov", ".mil", ".edu")) or ".gov/" in (url or ""):
        return 0, "primary(.gov/.mil/.edu)"
    s = SCORES.get(dom)
    if s is None:
        parts = dom.split(".")
        if len(parts) > 2:
            s = SCORES.get(".".join(parts[-2:]))
    if s is None:
        return 3, "unrated"
    if s >= 90:
        return 1, "NG>=90"
    if s >= 75:
        return 1, "NG 75-90"
    if s >= 60:
        return 2, "NG 60-75"
    return 4, "NG<60"


def cached_text(url: str) -> str | None:
    p = CACHE / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("text")
    except Exception:
        return None


def load():
    recs = []
    for r in RUNS:
        for f in sorted((ROOT / "eval/data/survey_claims/runs" / r).glob("results-*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    if not o.get("ok") or not isinstance(o.get("result"), dict):
                        continue
                    o["_run"] = r
                    recs.append(o)
    return recs


def pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "  n/a"


def main():
    recs = load()
    rounds = [(rec, rd) for rec in recs for rd in rec["result"].get("rounds", [])]
    print(f"posts(records)={len(recs)}  rounds={len(rounds)}  runs={RUNS}\n")

    # ---------- (A) drop-rate by reason ----------
    drops = Counter()
    n_docs = n_dropped = 0
    zero_doc_rounds = 0
    per_round_drops = defaultdict(list)
    for rec, rd in rounds:
        d = rd.get("dropped") or []
        docs = rd.get("docs") or []
        n_docs += len(docs)
        n_dropped += len(d)
        per_round_drops[rd["round"]].append((len(d), len(docs)))
        if not docs:
            zero_doc_rounds += 1
        for x in d:
            drops[x["reason"]] += 1

    cand = n_docs + n_dropped
    print("=== (A) DROP-RATE BY REASON ===")
    print(f"{'reason':16} {'count':>6} {'% of drops':>11} {'% of candidates':>16}")
    for r in REASONS:
        c = drops.get(r, 0)
        print(f"{r:16} {c:>6} {pct(c, n_dropped):>11} {pct(c, cand):>16}")
    for r in drops:
        if r not in REASONS:
            print(f"{r+' (other)':16} {drops[r]:>6}")
    print(f"{'TOTAL DROPPED':16} {n_dropped:>6} {'100.0%':>11} {pct(n_dropped, cand):>16}")
    print(f"{'DOCS READ':16} {n_docs:>6} {'':>11} {pct(n_docs, cand):>16}")
    print(f"\ncandidates touched = {cand}   docs read = {n_docs}   dropped = {n_dropped}")
    print(f"wasted rounds (0 docs read) = {zero_doc_rounds}/{len(rounds)} "
          f"({pct(zero_doc_rounds, len(rounds)).strip()})")
    print("\ndrops+docs per round index:")
    for k in sorted(per_round_drops):
        v = per_round_drops[k]
        dd = sum(a for a, _ in v)
        do = sum(b for _, b in v)
        z = sum(1 for _, b in v if b == 0)
        print(f"  round {k}: n={len(v):3}  dropped={dd:4}  docs={do:4}  zero-doc rounds={z}")

    # zero-doc rounds: split by cause (triage read=[] vs everything filtered out)
    empty_triage = sum(1 for _, rd in rounds
                       if not (rd.get("docs") or []) and (rd.get("triage") or {}).get("read") == [])
    print(f"  of the {zero_doc_rounds} zero-doc rounds: {empty_triage} had triage read=[] "
          f"(deliberate skip), {zero_doc_rounds - empty_triage} had picks that all died in filters/search")

    # ---------- (B) drop by source tier ----------
    print("\n=== (B) DROP-RATE BY SOURCE TIER ===")
    buckets = ["primary(.gov/.mil/.edu)", "NG>=90", "NG 75-90", "NG 60-75", "unrated", "NG<60"]
    grid = defaultdict(Counter)   # bucket -> reason -> n
    seen_docs_b = Counter()
    for rec, rd in rounds:
        for x in (rd.get("dropped") or []):
            _, b = tier_of(x["url"])
            grid[b][x["reason"]] += 1
        for d in (rd.get("docs") or []):
            _, b = tier_of(d["url"])
            seen_docs_b[b] += 1

    hdr = f"{'bucket':24}" + "".join(f"{r[:9]:>10}" for r in REASONS) + f"{'DROPPED':>9}{'READ':>7}{'droprate':>10}"
    print(hdr)
    for b in buckets:
        row = grid[b]
        tot = sum(row.values())
        read = seen_docs_b[b]
        print(f"{b:24}" + "".join(f"{row.get(r,0):>10}" for r in REASONS)
              + f"{tot:>9}{read:>7}{pct(tot, tot+read):>10}")
    allrow = Counter()
    for b in buckets:
        allrow.update(grid[b])
    print(f"{'ALL':24}" + "".join(f"{allrow.get(r,0):>10}" for r in REASONS)
          + f"{sum(allrow.values()):>9}{sum(seen_docs_b.values()):>7}")

    # gate-relevant: PRIMARY or NG>=60 lost to a *quality* filter (scrape-failed/junk/too-thin/too-short)
    QUAL = {"scrape-failed", "too-short", "junk-page", "too-thin"}
    gate_buckets = ["primary(.gov/.mil/.edu)", "NG>=90", "NG 75-90", "NG 60-75"]
    lost = sum(grid[b][r] for b in gate_buckets for r in QUAL)
    lost_reads = sum(seen_docs_b[b] for b in gate_buckets)
    lost_all = sum(sum(grid[b].values()) for b in gate_buckets)
    print(f"\nGATE-RELEVANT: gate-eligible sources (primary or NG>=60) lost to a QUALITY filter "
          f"({'/'.join(sorted(QUAL))}): {lost}")
    print(f"  gate-eligible candidates touched = {lost + lost_reads} "
          f"(read {lost_reads}, lost-to-quality {lost}, lost-to-any-reason {lost_all})")
    print(f"  -> {pct(lost, lost + lost_reads).strip()} of every gate-eligible source we touched "
          f"was destroyed by a scrape/junk/thin filter")
    for r in sorted(QUAL):
        print(f"     {r:14} {sum(grid[b][r] for b in gate_buckets):>4}")
    # per-post: how many posts lost >=1 gate-eligible source
    posts_lost = set()
    for rec, rd in rounds:
        for x in (rd.get("dropped") or []):
            _, b = tier_of(x["url"])
            if b in gate_buckets and x["reason"] in QUAL:
                posts_lost.add((rec["_run"], rec["post_id"]))
    npost = len({(r["_run"], r["post_id"]) for r in recs})
    print(f"  posts losing >=1 gate-eligible source to a quality filter: "
          f"{len(posts_lost)}/{npost} ({pct(len(posts_lost), npost).strip()})")

    # ---------- (C) backfill / substitution ----------
    print("\n=== (C) BACKFILL / SUBSTITUTION ===")
    sub_rounds = 0
    dl, rl = [], []          # per-round mean tier of dropped vs read
    pairs = []
    for rec, rd in rounds:
        dr = [x for x in (rd.get("dropped") or [])
              if x["reason"] in {"scrape-failed", "too-short", "junk-page", "too-thin",
                                 "republication", "near-dup"}]
        docs = rd.get("docs") or []
        picks = (rd.get("triage") or {}).get("read") or []
        # backfill fired iff more docs were read than would remain if drops shrank the round:
        # i.e. drops occurred AND we still filled the triage target
        if dr and docs and len(docs) >= len(picks) and picks:
            sub_rounds += 1
        if dr and docs:
            dt = [tier_of(x["url"])[0] for x in dr]
            rt = [tier_of(d["url"])[0] for d in docs]
            dl.append(sum(dt)/len(dt))
            rl.append(sum(rt)/len(rt))
            pairs.append((rec["_run"], rec["post_id"], rd["round"],
                          sum(dt)/len(dt), sum(rt)/len(rt)))
    # under-fill: triage asked for N reads, the walk (picks + backfill) delivered fewer
    under = shortfall = 0
    for _, rd in rounds:
        picks = (rd.get("triage") or {}).get("read") or []
        if not picks:
            continue
        target = len(picks)
        got = len(rd.get("docs") or [])
        if got < target:
            under += 1
            shortfall += target - got
    print(f"rounds where the walk UNDER-FILLED the triage target (backfill ran out): "
          f"{under}  (total slots lost: {shortfall})")
    rounds_with_drops = sum(1 for _, rd in rounds if rd.get("dropped"))
    print(f"rounds with >=1 drop: {rounds_with_drops}/{len(rounds)}")
    print(f"rounds where drops occurred AND the triage read-target was still filled "
          f"(=> untriaged backfill hits filled the slots): {sub_rounds}")
    if dl:
        import statistics as st
        print(f"\nper-round mean TIER (0=primary .. 4=NG<60), rounds with both drops and docs (n={len(dl)}):")
        print(f"  dropped: mean {st.mean(dl):.2f}  median {st.median(dl):.2f}")
        print(f"  read   : mean {st.mean(rl):.2f}  median {st.median(rl):.2f}")
        worse = sum(1 for _, _, _, d, r in pairs if r > d)
        better = sum(1 for _, _, _, d, r in pairs if r < d)
        same = len(pairs) - worse - better
        print(f"  rounds where READ docs are on average WORSE tier than the DROPPED ones: {worse}")
        print(f"  rounds where READ docs are BETTER tier: {better}   equal: {same}")
    # tier distribution: dropped-for-quality vs read
    print("\ntier distribution, quality-dropped vs read:")
    qd = Counter()
    for _, rd in rounds:
        for x in (rd.get("dropped") or []):
            if x["reason"] in {"scrape-failed", "too-short", "junk-page", "too-thin"}:
                qd[tier_of(x["url"])[1]] += 1
    tq, tr = sum(qd.values()), sum(seen_docs_b.values())
    print(f"{'bucket':24}{'quality-dropped':>17}{'read':>10}")
    for b in buckets:
        print(f"{b:24}{qd[b]:>7} {pct(qd[b],tq):>9}{seen_docs_b[b]:>4} {pct(seen_docs_b[b],tr):>5}")

    # ---------- (D) failing domains ----------
    print("\n=== (D) TOP FAILING DOMAINS ===")
    for reason in ("scrape-failed", "junk-page", "too-thin", "too-short"):
        c = Counter()
        for _, rd in rounds:
            for x in (rd.get("dropped") or []):
                if x["reason"] == reason:
                    c[_domain_of(x["url"])] += 1
        print(f"\n-- {reason} (total {sum(c.values())}, {len(c)} distinct domains) --")
        for dom, n in c.most_common(18):
            print(f"  {n:>3}  {dom:38} tier={tier_of('https://'+dom+'/')[1]}")

    # per-domain touch rate (fail vs read) for notable domains
    print("\n-- per-domain: touched vs killed by scrape-failed/junk-page/too-thin --")
    touched, killed = Counter(), Counter()
    for _, rd in rounds:
        for x in (rd.get("dropped") or []):
            d = _domain_of(x["url"])
            touched[d] += 1
            if x["reason"] in {"scrape-failed", "junk-page", "too-thin", "too-short"}:
                killed[d] += 1
        for d0 in (rd.get("docs") or []):
            touched[d0["domain"]] += 1
    rows = [(d, killed[d], touched[d], killed[d]/touched[d]) for d in touched if touched[d] >= 3]
    rows.sort(key=lambda r: (-r[1], -r[3]))
    print(f"{'domain':38}{'killed':>7}{'touched':>9}{'killrate':>10}  tier")
    for d, k, t, r in rows[:25]:
        print(f"{d:38}{k:>7}{t:>9}{r*100:>9.0f}%  {tier_of('https://'+d+'/')[1]}")

    # ---------- (E) truncation + hypotheses (scrape cache) ----------
    print("\n=== (E) TRUNCATION + CLEAN_TEXT HYPOTHESES (scrape cache) ===")
    cap_chars = CFG.cap_tok * 4
    hit = miss = trunc = 0
    frac_cut = []
    dropped_lines = Counter()
    numeric_dropped = 0
    total_dropped_lines = 0
    title_examples = []
    for _, rd in rounds:
        for d in (rd.get("docs") or []):
            raw = cached_text(d["url"])
            if raw is None:
                miss += 1
                continue
            hit += 1
            cl = clean_text(raw)
            ss = sentences(cl)
            _, last, _rs = numbered_block(ss, CFG.cap_tok)
            if last < len(ss):
                trunc += 1
                kept = sum(len(s) for s in ss[:last])
                total = sum(len(s) for s in ss)
                frac_cut.append(1 - kept / total if total else 0)
            # hypothesis (a): lines killed by the <25-char & no-terminal-punct rule
            for ln in raw.splitlines():
                ln2 = re.sub(r"[ \t]+", " ", ln).strip()
                if ln2 and len(ln2) < 25 and not re.search(r"[.!?]$", ln2):
                    total_dropped_lines += 1
                    if re.search(r"\d", ln2):
                        numeric_dropped += 1
                        dropped_lines[ln2] += 1
    print(f"docs read with a cache hit: {hit}   cache miss: {miss}")
    print(f"docs whose cleaned sentence list EXCEEDED the {cap_chars}-char READ cap "
          f"(truncation bit): {trunc}/{hit} ({pct(trunc, hit).strip()})")
    if frac_cut:
        import statistics as st
        frac_cut.sort()
        print(f"  fraction of the doc CUT, over truncated docs: mean {st.mean(frac_cut)*100:.1f}% "
              f"median {st.median(frac_cut)*100:.1f}%  p90 {frac_cut[int(.9*len(frac_cut))-1]*100:.1f}% "
              f"max {max(frac_cut)*100:.1f}%")
    print(f"\nhypothesis (a): lines killed by '<25 chars & no terminal punctuation'")
    print(f"  total such lines across cached read docs: {total_dropped_lines}")
    print(f"  ... of which contain a DIGIT (figure/date/count): {numeric_dropped} "
          f"({pct(numeric_dropped, total_dropped_lines).strip()})")
    print(f"  most common numeric dropped lines:")
    for ln, n in dropped_lines.most_common(20):
        print(f"    {n:>4}  {ln!r}")

    # hypothesis (b): does the page TITLE survive into cleaned text?
    # substantive figures among the killed short lines
    SUBST = re.compile(r"(\$[\d,.]+|\d+(\.\d+)?\s*(%|percent|million|billion|months|years|"
                       r"killed|dead|people|votes|jobs|cases|deaths|arrests|counts))", re.I)
    subst = Counter()
    for _, rd in rounds:
        for d in (rd.get("docs") or []):
            raw = cached_text(d["url"])
            if not raw:
                continue
            for ln in raw.splitlines():
                ln2 = re.sub(r"[ \t]+", " ", ln).strip()
                if ln2 and len(ln2) < 25 and not re.search(r"[.!?]$", ln2) and SUBST.search(ln2):
                    subst[ln2] += 1
    print(f"\n  killed short lines carrying a SUBSTANTIVE figure ($ / % / million / months / "
          f"killed / votes ...): {sum(subst.values())} ({len(subst)} distinct)")
    for ln, n in subst.most_common(15):
        print(f"    {n:>3}  {ln!r}")

    # ---- hypothesis (b): does the page TITLE survive into the text READ sees? ----
    # Serper cache stores each hit's title -> ground truth for the headline of a scraped URL.
    print("\nhypothesis (b): does the page TITLE/HEADLINE reach READ?")
    print("  NOTE: the serper cache stores only {url, snippet, date, content} — no title; and the")
    print("  scrape cache stores trafilatura's BODY text (extract() without with_metadata).")
    print("  So there is no ground-truth headline on disk. PROXY: the URL slug (news URLs are")
    print("  slugified headlines). Test = do the slug's content words appear in the text READ saw?")
    titles: dict[str, str] = {}
    for _, rd in rounds:
        for d in (rd.get("docs") or []):
            slug = re.sub(r"[-_/]+", " ", re.sub(r"\.(html?|php|aspx?)$", "",
                                                 (d["url"].split("?")[0].rstrip("/").split("/")[-1])))
            if len([w for w in slug.split() if len(w) > 3]) >= 4:   # a real slug, not an id
                titles[d["url"]] = slug
    print(f"  read docs with a usable headline-slug: {len(titles)}")

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", " ", s.lower())

    def head_core(t):
        # strip the trailing " - Outlet" / " | Outlet" site suffix
        return re.split(r"\s[|–—-]\s", t)[0]

    tot = in_clean = in_read = 0
    misses = []
    for _, rd in rounds:
        for d in (rd.get("docs") or []):
            t = titles.get(d["url"])
            raw = cached_text(d["url"])
            if not t or not raw:
                continue
            tot += 1
            core = norm(head_core(t))
            toks = [w for w in core.split() if len(w) > 3]
            if not toks:
                continue
            cl = clean_text(raw)
            ss = sentences(cl)
            _, last, _rs = numbered_block(ss, CFG.cap_tok)
            body_clean = " ".join(norm(x) for x in ss)
            body_read = " ".join(norm(x) for x in ss[:last])
            # "title present" = >=70% of its content words appear in the text
            hit_c = sum(1 for w in toks if w in body_clean) / len(toks)
            hit_r = sum(1 for w in toks if w in body_read) / len(toks)
            if hit_c >= 0.7:
                in_clean += 1
            if hit_r >= 0.7:
                in_read += 1
            elif len(misses) < 12:
                misses.append((d["domain"], head_core(t)[:80], round(hit_r, 2)))
    print(f"  read docs with a known title: {tot}")
    print(f"    title content-words present in the CLEANED text: {in_clean} ({pct(in_clean, tot).strip()})")
    print(f"    title present in the text READ ACTUALLY SAW (post-cap): {in_read} ({pct(in_read, tot).strip()})")
    print(f"    -> headline NEVER reaches READ for {tot - in_read} docs ({pct(tot-in_read, tot).strip()})")
    print("    examples where the headline is absent from what READ saw:")
    for dom, t, h in misses:
        print(f"      [{h:.2f}] {dom:26} {t!r}")

    # ---------- per-run breakdown (main_v3 and its rerun are the SAME 50 posts) ----------
    print("\n=== PER-RUN BREAKDOWN ===")
    for run in RUNS:
        rr = [(rec, rd) for rec, rd in rounds if rec["_run"] == run]
        nd = sum(len(rd.get("dropped") or []) for _, rd in rr)
        ndc = sum(len(rd.get("docs") or []) for _, rd in rr)
        q = sum(1 for _, rd in rr for x in (rd.get("dropped") or [])
                if x["reason"] in {"scrape-failed", "too-short", "junk-page", "too-thin"}
                and tier_of(x["url"])[1] in gate_buckets)
        posts = len({rec["post_id"] for rec, _ in rr})
        pl = len({rec["post_id"] for rec, rd in rr for x in (rd.get("dropped") or [])
                  if x["reason"] in {"scrape-failed", "too-short", "junk-page", "too-thin"}
                  and tier_of(x["url"])[1] in gate_buckets})
        print(f"  {run:20} posts={posts:3} rounds={len(rr):3} docs={ndc:3} dropped={nd:3} "
              f"({pct(nd, nd+ndc).strip()})  gate-eligible lost to quality filter={q:3}  "
              f"posts affected={pl}/{posts}")

    # post-40 DOJ case
    print("\n-- post-40 / '500 months' hunt (all cached docs + drops) --")
    for rec, rd in rounds:
        for x in (rd.get("dropped") or []) + [dict(url=d["url"], reason="READ") for d in (rd.get("docs") or [])]:
            if True:
                raw = cached_text(x["url"])
                if raw and re.search(r"\b\d{3}\s*months\b", raw, re.I):
                    cl = clean_text(raw)
                    inclean = bool(re.search(r"\b\d{3}\s*months\b", cl, re.I))
                    ss = sentences(cl)
                    _, last, _rs = numbered_block(ss, CFG.cap_tok)
                    seen_by_read = any(re.search(r"\b\d{3}\s*months\b", s, re.I) for s in ss[:last])
                    print(f"  {rec['_run']} {rec['post_id']} r{rd['round']} {x['reason']:14} {x['url']}")
                    print(f"     raw has 'NNN months': YES | survives clean_text: {inclean} | "
                          f"inside READ's {cap_chars}-char window: {seen_by_read} "
                          f"(doc has {len(ss)} sents, READ saw {last})")
                    for ln in raw.splitlines():
                        if re.search(r"\b\d{3}\s*months\b", ln, re.I):
                            ln2 = re.sub(r"[ \t]+", " ", ln).strip()
                            kill = len(ln2) < 25 and not re.search(r"[.!?]$", ln2)
                            print(f"     raw line ({len(ln2)}c, killed_by_short_rule={kill}): {ln2!r}")


if __name__ == "__main__":
    main()
