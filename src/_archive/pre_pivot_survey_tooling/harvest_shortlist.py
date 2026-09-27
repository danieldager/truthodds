"""Systematic per-outlet harvester for the survey claim pool (GitHub #13/#14).

Reads the curated 60-outlet shortlist (survey_claims/source_shortlist.csv: cell x domain, chosen by
NewsGuard reliability x orientation, ranked by Tranco traffic). For each outlet, pulls recent articles
(RSS feed -> homepage scrape -> Jina for bot-blocked), keeps only federal-politics articles (tweet-doc
keyword scope), and runs the neutral body-only ACE extractor. Per-outlet claim cap + per-cell target so
the four cells stay balanced toward a 1000-claim pool (250/cell). Resumable: skips URLs already in
ace_attempts, writes claims incrementally. High-visibility progress: per-outlet line + running cell
totals + throughput + ETA.

Run:  cd src && uv run --with feedparser python eval/scripts/claim_sourcing/harvest_shortlist.py
Args: [per_cell_target=250] [claim_cap_per_outlet=18] [article_budget_per_outlet=60] [only=cell|outlet,...]
"""
import os, sys, re, io, json, time, gzip, datetime
import pandas as pd
import requests, feedparser, trafilatura
from dotenv import load_dotenv

sys.path.insert(0, "src")
from eval import ace

load_dotenv()
REPO = "src"
OUTDIR = REPO + "/eval/data/survey_claims"
SHORTLIST = OUTDIR + "/source_shortlist.csv"
CLAIMS_OUT = OUTDIR + "/shortlist_claims.parquet"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"
JINA_KEY = os.environ.get("JINA_API_KEY", "")
BR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
TIMEOUT = 12
MIN_CHARS = 400

PER_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 250
CLAIM_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 18
ART_BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 60
ONLY = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
CELL_STOP = PER_CELL + 25  # small headroom for trimming

# --- federal-politics scope (tweet-selection procedure doc keyword lists) ---
KW = [r"\bbiden\b", r"\btrump\b", r"\bpresident", r"\belection", r"\badministration\b", r"\bwhite house\b",
      r"\bharris\b", r"\bvance\b", r"\bobama\b", r"\bcongress", r"\bsenate\b", r"\bgop\b", r"\brepublican",
      r"\bdemocrat", r"\bgovernor\b", r"\bsupreme court\b", r"\bvoter?s?\b", r"\bcampaign\b", r"\bpolitic",
      r"\babortion\b", r"\broe v", r"\bdobbs\b", r"\bimmigration\b", r"\bborder\b", r"\bmigrant",
      r"\basylum\b", r"\bobamacare\b", r"\bmedicare\b", r"\bmedicaid\b", r"\binflation\b", r"\btariff",
      r"\bforeign policy\b", r"\bnato\b", r"\bisrael\b", r"\bgaza\b", r"\bukraine\b", r"\brussia\b",
      r"\bchina\b", r"\btaiwan\b", r"\bhamas\b", r"\blgbtq", r"\btransgender\b", r"\bwoke\b",
      r"\bjanuary 6\b", r"\bcapitol\b", r"\belectoral fraud\b", r"\bclimate change\b", r"\bpentagon\b",
      r"\bwhite house\b", r"\bimpeach", r"\bdeportation\b", r"\bfederal\b", r"\bcongressional\b"]
KW_RE = re.compile("|".join(KW), re.I)
SECTION = re.compile(r"/(account|login|log-in|section|sitemap|tag|topic|topics|author|authors|video|videos|live|newsletter|newsletters|subscribe|hub|category|photos|gallery|about|contact|privacy|terms)(/|$)", re.I)


def dom(u):
    m = re.match(r"https?://([^/]+)", u or "")
    return m.group(1).lower().replace("www.", "") if m else ""


def is_article(u, domain):
    if domain.replace("www.", "") not in dom(u):
        return False
    if SECTION.search(u) or u.rstrip("/").lower().endswith((".xml", ".gz", ".jpg", ".png", ".css", ".js")):
        return False
    path = re.sub(r"https?://[^/]+", "", u).split("?")[0]
    segs = [s for s in path.split("/") if s]
    return bool(re.search(r"/\d{4}/\d{2}/", path)) or bool(segs and (segs[-1].count("-") >= 3 or re.search(r"\d{6,}", segs[-1])))


def dedupe(pairs):
    seen, out = set(), []
    for u, t in pairs:
        if not u:
            continue
        b = u.split("#")[0].split("?")[0]
        if b in seen:
            continue
        seen.add(b)
        out.append((b, t))
    return out


def jina_get(url):
    if not JINA_KEY:
        return None
    try:
        r = requests.get("https://r.jina.ai/" + url, headers={**BR, "Authorization": "Bearer " + JINA_KEY}, timeout=30)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def discover(outlet, access, note, domain, sess):
    """Return list of (article_url, title) recent-first."""
    if access == "FEED" and str(note).startswith("http"):
        try:
            r = sess.get(note, timeout=TIMEOUT)
            ents = feedparser.parse(r.content).entries if r.status_code == 200 else []
            return dedupe([(getattr(e, "link", ""), getattr(e, "title", "")) for e in ents])
        except Exception:
            return []
    if access == "HTML":
        try:
            html = sess.get("https://" + domain + "/", timeout=TIMEOUT).text
        except Exception:
            return []
        links = re.findall(r'href=["\']([^"\']+)["\']', html or "")
        links = [l if l.startswith("http") else "https://" + domain + l for l in links if l.startswith(("/", "http"))]
        return dedupe([(l, None) for l in links if is_article(l, domain)])
    # BLOCKED / DEAD -> Jina homepage, pull markdown links
    md = jina_get("https://" + domain + "/")
    if not md:
        return []
    links = re.findall(r"\]\((https?://[^)]+)\)", md)
    return dedupe([(l, None) for l in links if is_article(l, domain)])


def fetch_body(url, access, sess):
    if access in ("BLOCKED", "DEAD"):
        txt = jina_get(url)
        return (txt, None) if txt and len(txt) >= MIN_CHARS else (None, None)
    try:
        html = sess.get(url, timeout=TIMEOUT, allow_redirects=True).text
    except Exception:
        html = None
    if not html:
        txt = jina_get(url)  # fallback to Jina on transport failure
        return (txt, None) if txt and len(txt) >= MIN_CHARS else (None, None)
    body = trafilatura.extract(html, include_comments=False, include_tables=True)
    title = None
    try:
        md = trafilatura.extract_metadata(html)
        title = getattr(md, "title", None) if md else None
    except Exception:
        pass
    return body, title


def main():
    sl = pd.read_csv(SHORTLIST)
    if ONLY:
        sl = sl[sl.cell.isin(ONLY) | sl.domain.isin(ONLY)]
    seen = set(pd.read_parquet(ATTEMPTS)["url"]) if os.path.exists(ATTEMPTS) else set()
    have = pd.read_parquet(CLAIMS_OUT) if os.path.exists(CLAIMS_OUT) else pd.DataFrame()
    cell_have = have.groupby("cell").size().to_dict() if len(have) else {}
    sess = requests.Session(); sess.headers.update(BR)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    attempts, new_claims = [], []
    n_out = len(sl)

    for oi, row in enumerate(sl.itertuples(), 1):
        cell, domain, access, note = row.cell, row.domain, row.access, row.note
        rel, ori = cell.split("-")
        if cell_have.get(cell, 0) >= CELL_STOP:
            print(f"[{oi}/{n_out}] {domain:28} ({cell}) SKIP - cell full ({cell_have[cell]})", flush=True)
            continue
        pairs = [(u, t) for u, t in discover(domain, access, note, domain, sess) if u not in seen]
        kept, art = 0, 0
        for aurl, rtitle in pairs:
            if kept >= CLAIM_CAP or art >= ART_BUDGET or cell_have.get(cell, 0) >= CELL_STOP:
                break
            seen.add(aurl); art += 1
            body, mtitle = fetch_body(aurl, access, sess)
            headline = rtitle or mtitle or ""
            rec = {"source": domain, "url": aurl, "headline": headline, "attempted_at": now,
                   "body_chars": len(body) if body else 0}
            if not body or len(body) < MIN_CHARS:
                rec.update({"status": "fetch_failed", "n_claims": 0, "claims_json": "[]"})
                attempts.append(rec); time.sleep(0.25); continue
            # federal-politics scope gate (title + head of body)
            if not KW_RE.search((headline or "") + " " + body[:2500]):
                rec.update({"status": "off_topic", "n_claims": 0, "claims_json": "[]"})
                attempts.append(rec); time.sleep(0.2); continue
            try:
                cl = ace.extract(body, headline=headline).get("claims", [])
            except Exception:
                cl = []
            rec.update({"status": "claims" if cl else "none", "n_claims": len(cl),
                        "claims_json": json.dumps(cl, ensure_ascii=False)})
            attempts.append(rec)
            for c in cl:
                new_claims.append({"cell": cell, "reliability": rel, "orientation": ori, "outlet": domain,
                                   "domain": domain, "ng_score": row.ng_score, "url": aurl, "headline": headline,
                                   "claim": c.get("claim"), "basis": c.get("basis"), "quote": c.get("quote")})
                kept += 1
            cell_have[cell] = cell_have.get(cell, 0) + len(cl)
            time.sleep(0.25)
        rate = (oi) / max(1e-9, (time.time() - t0) / 60)
        eta = (n_out - oi) / max(1e-9, rate)
        tot = sum(cell_have.get(c, 0) for c in ["Reliable-Left", "Reliable-Right", "Unreliable-Left", "Unreliable-Right"])
        print(f"[{oi}/{n_out}] {domain:28} ({cell}) {access}: {art} arts -> {kept} claims | "
              f"cell={cell_have.get(cell,0)} total={tot} | {rate:.1f} outl/min ETA {eta:.0f}m", flush=True)

        # incremental persist after each outlet
        if new_claims:
            allc = pd.concat([have, pd.DataFrame(new_claims)], ignore_index=True) if len(have) else pd.DataFrame(new_claims)
            allc.to_parquet(CLAIMS_OUT, index=False)
        if attempts:
            at = pd.DataFrame(attempts)
            if os.path.exists(ATTEMPTS):
                at = pd.concat([pd.read_parquet(ATTEMPTS), at]).drop_duplicates("url", keep="last").reset_index(drop=True)
            at.to_parquet(ATTEMPTS, index=False)
            attempts = []

    final = pd.read_parquet(CLAIMS_OUT) if os.path.exists(CLAIMS_OUT) else pd.DataFrame()
    print("\n=== harvest complete ===", flush=True)
    if len(final):
        print("claims by cell:\n" + final.groupby("cell").size().to_string(), flush=True)
        print(f"total raw claims: {len(final)} (pre dedup/filter)", flush=True)


if __name__ == "__main__":
    main()
