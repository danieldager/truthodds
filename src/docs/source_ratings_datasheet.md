# Datasheet — US Source-Rating Dataset (reliability × bias)

*Status: **LIVING** (update on every change). Last updated: 2026-07-01. Maintainer: Daniel.
Follows the "Datasheets for Datasets" (Gebru et al. 2021) structure, condensed.*

## Motivation
An **open, publishable** way to judge the **reliability** and **political bias** of **popular US news
outlets**, to transparently **select a subset** spanning the reliability×bias space for the survey stimulus
(GitHub issue #13) and later **validate** pipeline veracity against source reliability (#15). Hard constraint:
**must be publishable** → every source redistributable for research (rules out NewsGuard, MBFC, Ad Fontes,
and Lin `pc1`).

## Method of record (simple, publishable)
Over a **curated inventory of popular US outlets**, the selection pool is the **intersection of the two
sources that both rate an outlet**:
- **Reliability** = **Wikipedia RSP** category — a human editorial-consensus label (NOT a computed metric),
  collapsed to 3 tiers: **Reliable** (generally-reliable) / **Mixed** (no-consensus) / **Unreliable**
  (generally-unreliable, deprecated, blacklisted). RSP is the only open, full-range, MBFC-independent source.
- **Bias (editorial)** = **AllSides** signed Left→Right (left / lean-left / center / lean-right / right).
- **Bias (audience, supplementary)** = **Robertson** partisan-audience (−1 Dem … +1 Rep) — *who shares*, a
  different construct; carried for cross-comparison, not the primary bias axis.

**Selection pool = outlets present in BOTH AllSides and RSP.** (Outlets missing either are excluded.)

### Why Iffy/CRED-1 were dropped from scoring
They are **flagged-population** scores (built only from domains a low-cred list already flagged), living in a
compressed ~0.1–0.3 band — NOT full-range reliability raters. Averaging them with RSP structurally penalized
any outlet that merely *appeared* in them (e.g. The Intercept: RSP generally-reliable but CRED-1 0.205 via a
stale 2017 OpenSources "bias" tag → a meaningless 0.6 mean). Fix: reliability = RSP category alone; Iffy/CRED-1
are not used. This also resolves the "bucketed vs continuous" concern by owning the ordinal instead of faking decimals.

## Artifacts (`src/eval/data/`)
| File | What | Status |
|---|---|---|
| **`us_source_ratings.csv`** | **PRIMARY** — 56-outlet selection pool (AllSides ∩ RSP): reliability tier + editorial bias + audience bias | **method of record** |
| `source_candidates_eu.csv` | EU (80 outlets): `pc1` reliability + Wikidata P1387 bias (10% coverage) | parked (bias gap) |

## Sources & licenses
| Source | Provides | License | In scoring? |
|---|---|---|---|
| **Wikipedia RSP** (EN) | reliability category (human consensus) | CC BY-SA 4.0 | **yes** (reliability axis) |
| **AllSides** (`favstats/AllSideR`) | editorial bias, signed 5-pt | CC BY-NC 4.0 (research OK) | **yes** (bias axis) |
| **Robertson 2018** (Dataverse `doi:10.7910/DVN/QAN5VX`) | audience bias −1…+1 | "research only" | supplementary column |
| **Tranco** top-1M | traffic rank | open | reference (popularity) |
| ~~Iffy, CRED-1~~ | low-cred flags | CC-BY | **dropped** (incompatible scale) |
| ~~Lin pc1, NewsGuard, MBFC, Ad Fontes~~ | — | proprietary/unclear | excluded (not publishable) |

## Columns (`us_source_ratings.csv`)
`outlet`, `domain`, `rsp_status` (raw 5-level Wikipedia label), `reliability_tier` (Reliable/Mixed/Unreliable),
`bias_editorial` (left…right, AllSides), `bias_audience` (−1…+1, Robertson, supplementary), `tranco_rank`.

## Coverage & the selection grid (2026-07-01)
56 outlets (of 74 curated; 18 dropped — mainstream outlets RSP omits + low-cred outlets AllSides omits).

```
              left  lean-left  center  lean-right  right
Reliable        8       10       11        1         0
Mixed           4        1        3        3         2
Unreliable      2        0        0        1        10
```

## Known limitations & caveats
- **Diagonal-band asymmetry (the key constraint).** Reliable outlets skew left/center; unreliable skew right.
  **Reliable-right = 0, Reliable-lean-right = 1 (Reason); Unreliable-center = 0, Unreliable-lean-left = 0.**
  A *balanced* reliability×bias grid is impossible from real US outlets — reliable-conservative and
  unreliable-liberal barely exist (consistent with the procedure doc's "pro-Democrat false claims are much fewer").
- **RSP is human-consensus, ordinal, and only covers *disputed* sources** — so uncontroversial mainstream
  outlets (MSNBC, CBS, CNBC, PBS, MarketWatch, Slate…) are absent and excluded by the intersection.
- **⚠️ RSP has a documented LEFT lean — reliability is partly entangled with political side (a confound for
  a political-misinformation study).** Evidence: a large-scale analysis found "a moderate yet systematic
  liberal polarization in Wikipedia's news media sources" (arXiv:2210.16065); Wikipedia's deprecated-sources
  list is 16 right-leaning vs 1 left-leaning (Occupy Democrats), and Fox News is "generally unreliable" for
  politics while MSNBC/CNN are "generally reliable" (en.wikipedia.org/wiki/Ideological_bias_on_Wikipedia);
  Rozado 2024 (Manhattan Institute) found more negative sentiment toward right-of-center figures/media;
  Greenstein & Zhu 2012 (AER 102(3):343–48) found an early Democratic slant that diminished over time (so
  the effect is moderate, not extreme). CONSEQUENCE for our grid: reliable right-of-center outlets get
  scattered — WSJ *news* is coded **center** by AllSides, The Economist **lean-left**, and National
  Review / Washington Examiner / Fox get RSP "no-consensus" (→ Mixed) — leaving Reliable-Right ≈ 0 (only
  Reason). Do NOT read "conservative = unreliable" as ground truth; it is partly a rater artifact.
- **AllSides is CC-BY-NC** → combined product is non-commercial (fine for academic release).
- **Robertson is "research only"** and measures *audience* not editorial slant → supplementary; confirm terms
  before publishing that column.
- **Curated 74-outlet inventory** = a documented, extensible sampling frame (intersection = 56), not exhaustive.

## Intended uses
Select the survey source subset (#13); reliability-vs-veracity validation (#15); editorial-vs-audience comparison.

## Changelog
- **2026-07-01 (pm, v2)** — Switched to **AllSides ∩ RSP** intersection as the selection pool (56 outlets);
  reliability = **RSP category tier** (dropped Iffy/CRED-1 — flagged-population scores corrupted the average);
  bias = AllSides; Robertson audience kept as supplementary. Deleted pc1/NewsGuard-based tables.
- **2026-07-01 (am)** — Initial datasheet; US pc1×Robertson tables; fixed `www.` bug; removed audience-suspect flag.
