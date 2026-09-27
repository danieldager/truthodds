# Fact-check harvest sources — trusted-publisher feasibility

Goal: **1k–5k recent, highest-reliability fact-checks**, primarily to test that our
pipeline's **binary flag matches `harmonize(`the fact-checker's verdict`)`**.

## Field tiers (decided 2026-06-23, w/ Daniel)

- **REQUIRED (verification eval):** a **claim** (normalized is enough) + a **harmonizable
  verdict** → binary gold. A row is usable the moment it has these two. **Harmonizability is
  the one hard gate** — a publisher whose verdict we can't map (even via LLM) is the only kind
  we drop.
- **BONUS (parallel claim-extraction dataset), best-effort:** wherever the fact-check exposes
  the **claim's source URL** (original tweet/post/article), fetch it → **raw claim + context**,
  paired with the normalized claim = a `(raw → normalized)` extraction pair. Chase when present
  and the link is alive; accept misses.
- **OPTIONAL:** verdict's evidence sources, context without a source URL.

The API and DataCommons feed give claim + verdict cheaply (no article scrape needed for the
core). The **source URL** is stripped by the API but carried by ClaimReview `itemReviewed.url`/
`appearance[]` (the feed, and some on-page JSON-LD) — that's what earns the feed/JLD path.

## Verified matrix (live fetches, 2026-06-23)

ENUM = discrete rating we map by table; LLM = free-text we LLM-map (already do for Full Fact).
✓/p/✗ = clean / partial / absent.

| Publisher | Discovery channel | Verdict (harmonize) | Norm. claim | Src-URL (extraction bonus) | ~Vol/mo | Lang | Core-usable? |
|---|---|---|---|---|---|---|---|
| **Lead Stories** | Atom `/hoax-alert/atom.xml` → per-article ClaimReview JSON-LD | ENUM (1–5) ✓ | ✓ `claimReviewed` | **✓ `itemReviewed.sameAs`** (only clean one) | 300–500 | EN | ✓✓ |
| **Snopes** | inline ClaimReview JSON-LD + `/feed/` (browser-UA) | ENUM ✓ | ✓ | p (embedded tweets in body) | ~100s | EN | ✓ |
| **PolitiFact** *(built)* | GFC API + RSS/archive → page scrape | ENUM (Truth-O-Meter) ✓ | ✓ | p ("Our Sources" body) | ~150–300/6mo | EN/ES | ✓ |
| **AFP** | GFC API (JSON-LD) + Wayback for body | ENUM ✓ | ✓ | p | very high | multi (EN+FR) | ✓ |
| **Full Fact** | own API `api.fullfact.org/content/claim-reviews/{uuid}` + JSON-LD | LLM (sentence, 1st token) | ✓ | p (prose) | ~100 | EN | ✓ |
| **20 Minutes "Fake Off"** | GFC API + `rss-societe.xml` → JSON-LD | ENUM ("Faux" 1–5) ✓ | ✓ | p | ~15–30 | FR | ✓ |
| **Science/Health Feedback** | **WP REST** `/wp-json/wp/v2/sf_review` + JSON-LD | ENUM ("Inaccurate" 1–5) ✓ | ✓ | weak (`url`="TBD") | ~10–20 | EN | ✓ (low vol) |
| **Les Surligneurs** | `/feed/` RSS + `sitemap_index.xml` | LLM (free-text) | ✓ headline | p (prose) | ~30–45 | FR | ✓ (LLM verdict) |
| **Africa Check** | `sitemap.xml` → HTML scrape | LLM (badge image) | ✓ headline | p (body; **strongest evidence links**) | ~15–30 | EN | ✓ (LLM verdict) |
| **FactCheck.org** | sitemap + `/feed/` → HTML | **LLM from prose conclusion** | **LLM from article** | p | ~30–60 | EN | ✓ (LLM claim+verdict) |
| **CheckNews (Libé)** | Wayback CDX (403 live) → HTML | **LLM from "Réponse"** | **LLM (invert question headline)** | p | ~20–30 | FR | ✓ (LLM claim+verdict) |

**Result under the relaxed bar: all 11 usable** — 7 with native enum verdicts (rule-map), 2 with
native free-text verdicts (LLM-map), and **2 (FactCheck.org, CheckNews) where the verdict — and
for FactCheck.org the claim too — is LLM-extracted from the prose conclusion** (lower-confidence
gold; decided fine 2026-06-23, w/ Daniel).

## Access tricks worth reusing

- **On-page ClaimReview JSON-LD** — the cleanest path where present (Lead Stories, Snopes,
  20 Minutes, Science Feedback): claim + verdict (+ sometimes source URL) with no HTML parsing.
- **Publisher's own API** — **Full Fact** (`api.fullfact.org/content/claim-reviews/{uuid}`) and
  **Science Feedback** (WordPress REST `sf_review`, paginated) are cleaner than scraping.
- **Atom/RSS + sitemaps** for enumeration; `news-sitemap.xml` is the complete recent list.
- **Wayback for hard-blocked origins** — AFP (Akamai 403) and CheckNews: Wayback **preserves
  the original JSON-LD**, whereas `r.jina.ai` returns text but **strips** JSON-LD. Prefer Wayback
  when markup matters; jina only for plain body.
- **Browser-UA** defeats soft bot-walls (Snopes AI-bot robots block, 20 Minutes, Newschecker).
- PolitiFact & FactCheck.org emit **no** on-page ClaimReview → claim+verdict come from the GFC API.

## Shared record schema (one dataset, many sources)

```
publisher · review_url · review_date · claim_date · language
normalized_claim · verdict_raw · verdict_harmonized · binary_label · judged_axis   # REQUIRED
claim_source_url · raw_claim · context                                            # BONUS (extraction pair)
verdict_sources[]                                                                  # OPTIONAL
```
Strict superset of `politifact_harvest.parquet`. Every publisher harvester emits this; rows flow
through `eval/harmonize.py` (+ LLM fallback) and `eval/enrich.py` (judged_axis).

## Build order (to clear 1k–5k recent with required fields + max extraction pairs)

1. **Lead Stories** — Atom → ClaimReview JSON-LD. Volume king (300–500/mo → thousands/yr) AND the
   only source with a clean structured **source URL** (`sameAs`) → best extraction-pair yield.
   Easiest scraper (JSON-LD, not visual). **Start here.**
2. **Snopes** (browser-UA JSON-LD) + **AFP** (GFC API + Wayback body) — high volume; AFP adds FR.
   NB: Snopes/AFP/Full Fact claim+verdict are already in hand via the API — scrape only to add the
   extraction bonus (source URLs in body).
3. **PolitiFact** — done (`eval/politifact.py`).
4. **French + science rounders:** 20 Minutes "Fake Off", Les Surligneurs, Science Feedback.
5. **LLM-verdict tier:** FactCheck.org, CheckNews — discover via sitemap/Wayback, scrape the
   article body, LLM-extract `{normalized_claim, verdict}` from the prose conclusion, then
   harmonize. Tag these rows (`verdict_source = "llm_from_prose"`) so the lower-confidence gold
   is sliceable.

## Lead-Stories-like social-debunk sources (verified live 2026-06-23)

Sites that debunk viral SOCIAL-MEDIA posts AND emit ClaimReview **with the original post URL** — the
Lead Stories profile (high source-URL yield → extraction pairs). **Our `claimreview._source_url`
already checks all the field variants below** (`author.sameAs` / `itemReviewed.url` / `appearance[].url`).
Reliability is Lead Stories: **IFCN-verified, MBFC Least Biased / Very High** (caveats: right-leaning
story-selection criticism — not accuracy; ByteDance-funding optics; one *dismissed* defamation suit).

| Rank | Publisher | source-URL field | verdict | IFCN/MBFC | lang | note |
|---|---|---|---|---|---|---|
| 1 | **Vishvas News** (IN) | `author.sameAs` (X) | numeric 1–5 | ✓ / listed | Hindi/EN +12 | **closest LS twin, multilingual goldmine** |
| 2 | **AFP Factuel** (FR) | `itemReviewed.url` (FB) | numeric | ✓ / LC-High | FR (+~26-lang AFP net) | **best FR** — DataDome-blocked → Wayback/headless |
| 3 | **India Today FC** (IN) | `author.sameAs` (FB) | numeric | ✓ | Hindi/EN | clean ld+json |
| 4 | **AAP FactCheck** (AU) | `appearance[].url` (IG) | text only | ✓ / **Least Biased-High** | EN | cleanest EN reputation |
| 5 | **Newschecker** (IN) | `appearance[].url` (X) | empty | ✓ | EN/Hindi+regional | ClaimReview in escaped Next.js `__next_f` (un-escape) |
| — | VERA Files (PH) · Misbar (AR) · Demagog (PL, Wayback) | `appearance[].url` | text | ✓/mixed | FIL/AR/PL | good, secondary |
| — | BOOM · Alt News · The Quint · Maldita · Newtral · dpa | weak/empty source field | mixed | mixed | IN/ES/DE | **partial** — ClaimReview but no usable source URL |
| — | Check Your Fact · USA Today · Factly · Africa Check · Facta | — | — | mixed | EN/IT | **no ClaimReview** (NewsArticle only post-Google-deprecation ~Apr 2025) |

**Two small harvester implications:** (a) source-URL extraction must check all three fields (already
does); (b) accept **text-only verdicts** (`alternateName`, harmonized via rule/LLM) for sites that omit
numeric `ratingValue` — our extractor already falls back to `alternateName`. Build order for these:
Vishvas News (clean, multilingual) → AFP Factuel (FR, via Wayback) → India Today → AAP → Newschecker.
