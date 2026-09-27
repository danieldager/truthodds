# Open Source-Rating Dataset — Design Brief

*Generated 2026-06-30 from a 6-angle research workflow (open reliability data, signed bias,
multilingual/EU, Lin methodology + better aggregation, licensing/legal, LLM/ML gap-fill) +
a licensing-verification pass. Goal: an OPEN, research-licensable, multilingual outlet-level
dataset on two axes — reliability and SIGNED left↔right bias — to (a) select sources spanning
the reliability×bias space and (b) validate our pipeline's veracity vs source reliability,
replacing paid NewsGuard. Context: `src/eval/data/newsguard_source_candidates.csv` already
prototypes the join. Related GitHub issues: #13 (source selection), #15 (veracity-vs-NewsGuard).*

## Bottom line

Build as **two layers**: a redistributable **OPEN CORE** we can publish + a **private LOOK-UP
layer** (MBFC/NewsGuard/Ad Fontes/GDI) used only for validation and selection, never republished.
- **Reliability axis:** Lin's "wisdom of experts" idea, rebuilt from cleanly-licensed inputs
  (Iffy CC-BY, Wikipedia RSP CC-BY-SA, CRED-1 CC-BY, Lasser/misinformation_domains CC-BY-SA, +
  our own ClaimReview debunk counts), aggregated with a **Bayesian ordinal latent-trait (IRT)
  model** instead of PCA-on-mean-imputed-data — giving per-domain reliability **with uncertainty**.
- **Signed bias axis (Lin lacks this entirely):** a separate latent seeded from **Wikidata P1387
  (CC0)** + **AllSides (CC-BY-NC, US anchors)**, extended to EU outlets via **audience-based
  ideal-point estimation** (Barberá/Eady, language-agnostic) crosswalked to **CHES** party
  positions (open, 31 EU countries).
- **Hard constraint:** the only sources with BOTH axes + broad EU coverage (MBFC, NewsGuard, Ad
  Fontes, GDI) are all **non-redistributable** → private validation targets only.
- **Main build risk:** signed bias for non-English EU mainstream outlets — **no open table
  exists**, so that layer requires an original **audience-scaling data-collection step** (X/API
  access, cost/ToS risk), not a download. Publishable reliability core is achievable now.

---

## Recommended stack

| Source | Axis | License | Publishable? |
|---|---|---|---|
| Iffy Index | reliability (low-cred anchor) | CC BY 4.0 | yes (attrib) |
| Wikipedia Perennial Sources (per-lang) | reliability (MBFC-independent) | CC BY-SA 4.0 | yes (share-alike) |
| CRED-1 | reliability + computed signals | CC BY 4.0 | yes |
| JanaLasser/misinformation_domains | reliability (adds IT/DE) | CC BY-SA 4.0 (compilation) | yes (care) |
| Google Fact Check / ClaimReview | reliability (our debunk-count signal, multilingual) | markup reusable | yes (derived) |
| Wikidata P1387 / P1142 | signed bias (best EU seed) | CC0 | yes |
| AllSides (AllSideR mirror) | signed bias (US anchor) | CC BY-NC 4.0 | yes (NC only) |
| Robertson 2018 bundle | signed bias (US calibration crosswalk) | cite / open | yes |
| CHES | signed bias (EU party crosswalk) | free for research | yes |
| Eady mediascores / Barberá tweetscores | signed bias (EU native estimation) | OSS method | outputs ours |
| euro\|topics + Media Cloud + Wikipedia lists | sampling frame (not a rating) | mixed/open | inventory |
| MBFC / NewsGuard / Ad Fontes / GDI | both (broad EU) | proprietary | **LOOK-UP ONLY** |

---

> Scope: an OPEN, research-licensable dataset of **outlet/domain-level** ratings on two axes — (1) reliability/factuality and (2) **signed** political bias (left↔right) — that is **multilingual and covers a broad set of European outlets across many countries**, improves on Lin et al. (2023), and cleanly separates what we can **publish** from what is **look-up-only**. A prior local artifact, `src/eval/data/newsguard_source_candidates.csv`, already prototypes the intended join (AllSides lean + Lin `pc1` + NewsGuard) and confirms the two-layer pattern below.

---

## 1. Inventory of usable sources

**Legend:** R = reliability, B(±) = signed bias, B(u) = unsigned bias, Inv = inventory only. Redist = redistributable in a published dataset.

| Source | Axis | Unit | Coverage | Languages | Access | License | Redist? |
|---|---|---|---|---|---|---|---|
| **Iffy Index** | R (low-cred) | domain | Global, EN-centric | EN meta | download (Sheet/CSV/JSON) | CC BY 4.0 | **Yes** (attrib) |
| **Wikipedia RSP** (per-lang) | R | outlet | Global; EN/FR/RU/SV/PT/ZH editions | multi | scrape / MW API | CC BY-SA 4.0 | **Yes** (share-alike) |
| **CRED-1** | R | domain | US/EN | EN | download (CSV/Zenodo) | CC BY 4.0 | **Yes** |
| **JanaLasser/misinformation_domains** | R | domain | US + IT + some DE | EN/IT | download (CSV) | CC BY-SA 4.0 (compilation) | **Yes*** (components mostly unlicensed factual data) |
| **OpenSources (Zimdars)** | R | domain | US/EN | EN | download | LGPL-3.0 (data!) / CC BY 4.0 claimed downstream | **Restricted** (attrib; muddled) |
| **Google Fact Check / ClaimReview** | R (independent signal) | article→agg | Global incl. EU FCs | multi | free REST API | markup openly reusable | **Yes** (our derived counts) |
| **Wikidata P1387 / P1142** | B(±) categorical | outlet | Global incl. all EU | lang-agnostic | SPARQL/dumps | **CC0** | **Yes** (best seed) |
| **AllSides** (AllSideR mirror) | B(±) 5-pt + meter | outlet | US; intl EN only | EN | download (~547 rows) | CC BY-NC 4.0 | **Yes (NC)** |
| **Robertson 2018 bundle** | B(±) audience + 4 bundled scales | domain | US | EN | download (tarball) | cite-paper / effectively open | **Yes** |
| **Bakshy 2015** | B(±) audience | domain | US (~500) | EN | Science suppl. / in Robertson | article suppl., widely reused | **Yes** |
| **Eady mediascores** | B(±) audience, w/ uncertainty | outlet | US published; method portable | lang-agnostic | Dataverse + R pkg | open academic / OSS | **Yes** (method + US) |
| **Barberá ideal points / tweetscores** | B(±) audience | outlet | US + ES/IT/DE/UK/FR/NL citizens | lang-agnostic | GitHub/Dataverse | open academic / OSS | **Yes** (method) |
| **CHES** | B(±) party | party (31 EU) | 31 EU countries | multi | download (reg.) | free for research + cite | **Yes** (crosswalk) |
| **Gentzkow-Shapiro slant** | B(±) content | US newspapers | US | EN (lang-dependent) | download | open academic | Yes but **US/EN-only, non-portable** |
| **Décodex (Le Monde)** | R 4-class | domain | France/FR | FR | GitHub mirror / JSON webservice | Le Monde IP, no license | **Unknown** (contact Le Monde) |
| **MBFC** | R + B(±) numeric | outlet | **Broadest intl incl. all target EU** | EN meta | scrape / paid API | proprietary, no redistribution | **No — look-up only** |
| **NewsGuard** | R 0-100 | domain | US/CA/UK/AU/NZ + FR/DE/IT/AT | EN/FR/DE/IT | paid feed | proprietary | **No — look-up only** |
| **Ad Fontes** | R + B(±) continuous | outlet | US + few intl | EN | paid CSV; free=image | proprietary | **No — look-up only** |
| **GDI** | R (disinfo-risk) | domain | Multi incl. EU markets | multi | paid list / PDFs | proprietary | **No** (cite PDFs) |
| **Robertson Partisan Audience (Dataverse)** | B(±) audience | domain | US | EN | download | "research only" (not CC0) | **Restricted** |
| **NELA-GT labels** | R | outlet | US + some intl | EN | de-accessioned ~2024 | orig CC0, now restricted; MBFC-derived | **Restricted** |
| **euro\|topics** | Inv | outlet | 32 EU countries | multi | scrape | unclear (bpb-funded) | **Unknown** (inventory use) |
| **Media Cloud** | Inv | outlet | 100+ countries, 25k src | multi | API/CSV | open project | **Yes** (frame) |
| **Lin/hauselin (REFERENCE)** | R `pc1` + B(u) | domain | US-weighted; some EU | EN | download CSV | no LICENSE; README "reuse" | **Restricted** (`pc1` OK; per-provider MBFC/AFM cols not) |

`*` misinformation_domains: CC-BY-SA covers the *compilation*; underlying ratings are factual data from mostly-unlicensed component lists — publishable but attribute + share-alike.

---

## 2. Harmonization design — two latents, multilingual

We estimate **two separate latent variables per domain**, not one collapsed axis. This is the core departure from Lin (who PCA's everything into one reliability `pc1` and reduces bias to unsigned centrism).

### 2a. Reliability latent (ordinal Bayesian factor model)
Inputs (open only): Iffy (binary low-cred), Wikipedia RSP (ordinal 5-level), CRED-1 (0-1), misinformation_domains (accuracy 1-5 / transparency 1-3), Décodex if licensed (4-class), and our **ClaimReview debunk-rate** signal (Poisson count of times the domain is the *subject* of an IFCN/EFCSN fact-check, exposure-normalized by Media Cloud story volume).

Method — replace "mean-impute then PCA" with a **graded-response / ordinal latent-trait model** (Bayesian IRT):
- Treat each source as a rater with its own thresholds and discrimination; each domain has a latent reliability `θ_rel`.
- Missingness is handled by the likelihood (a domain rated by only 2 sources simply has a wider posterior) — **no imputation of fake values**, which is Lin's weakest step.
- Output per domain: posterior mean `θ_rel`, **credible interval**, and n_sources. Publish all three.
- Anchor the scale with a small set of universally-agreed reliable (Reuters/AP/AFP) and unreliable (RT/Sputnik/Epoch/InfoWars) domains so cross-language editions share a metric.

### 2b. Signed bias latent (anchored audience scaling + party crosswalk)
Inputs: Wikidata P1387 (CC0 categorical, all EU), AllSides (US continuous anchors), Robertson bundle (US audience anchors), CHES (EU party positions), and our **own audience ideal-point estimates** for EU outlets.

Method:
1. **Anchor calibration (US/EN):** place AllSides + Robertson/Bakshy signed scores on one continuous axis via the Robertson crosswalk; regress to a shared −1…+1 scale. This calibrates what "one unit of left/right" means.
2. **EU native estimation (the hard part):** run a language-agnostic audience ideal-point model (`mediascores` / `tweetscores` correspondence analysis) on who follows/shares each EU outlet's account, per country. This yields native signed positions for Le Monde, Der Spiegel, Corriere, El País, De Telegraaf, etc. **This is data collection we must perform** (X/API access), not a download.
3. **CHES crosswalk & cross-scale bridging:** because per-country audience axes are only *internally* comparable, bridge them by mapping each country's audience-space to its CHES party left-right (parties appear in the same follow/share graph), and by using cross-national outlets (BBC, RT, Politico EU, Euronews) as bridge items. Result: EU outlet signed scores on a common axis.
4. **Wikidata as prior/backfill:** use P1387 categorical labels as an informative prior and to backfill outlets too small for reliable audience estimation.
- Output per outlet: posterior signed position, **credible interval**, method flag (audience / seed-only / crosswalk).

### 2c. Joining & keys
Key on **effective 2nd-level domain** (registrable domain), with explicit disambiguation for split brands (`20minutes.fr` FR vs `20min.ch` CH; RT/Sputnik language editions as distinct rows). Media Cloud + euro|topics provide canonical domain + country + language metadata.

---

## 3. Explicit improvements over Lin (2023)

1. **Signed bias axis** — Lin has none (only unsigned centrism/extremity). We add a genuine left↔right latent.
2. **No imputation-then-PCA** — Lin mean-imputes missing provider cells then runs PCA, which fabricates data and hides uncertainty. We use an ordinal Bayesian latent-trait model where missingness widens the posterior instead of inventing values.
3. **Per-domain uncertainty** — we publish credible intervals and n_sources for both axes; Lin ships a point `pc1` only.
4. **Multilingual / broad-EU coverage** — Lin is US-weighted and English. We add a per-country EU inventory + native-language reliability (RSP editions, Décodex, national ClaimReview) and native EU signed estimates.
5. **Cleaner licensing posture** — Lin's public CSV re-publishes proprietary MBFC/Ad Fontes per-provider columns (legally gray). Our open core uses only redistributable inputs; proprietary sources are validation-only.
6. **Updatability** — ClaimReview counts, Wikidata, RSP, and audience feeds all refresh; the model is re-runnable on a schedule rather than a one-off 2023 snapshot. Version + date-stamp each release.
7. **Two-axis source selection** — supports the project's stated need to *sample outlets spanning the reliability×bias space* (the `newsguard_source_candidates.csv` prototype generalized to EU).

---

## 4. Licensing / redistribution assessment

### CAN PUBLISH (open core)
- **CC0:** Wikidata P1387/P1142.
- **CC BY:** Iffy, CRED-1, ClaimReview-derived counts (our computation).
- **CC BY-SA (share-alike constrains our license):** Wikipedia RSP, misinformation_domains compilation. → If we include these, the derived dataset likely must be **CC BY-SA 4.0**. Decide early: BY-SA is the safe umbrella but forbids a more permissive release.
- **CC BY-NC (non-commercial only):** AllSides. Adding it makes the *combined* product NC — acceptable for academic release but flag it; consider shipping AllSides-derived columns as a *separately-licensed* file so the core can stay commercial-friendly if desired.
- **Open academic (cite):** Robertson bundle, Bakshy, Eady/Barberá methods+US data, Gentzkow-Shapiro, CHES.
- **Our own estimates** (EU audience ideal points, IRT reliability posteriors): we own these outputs; license them as we choose (subject to the SA/NC constraints inherited from any input actually redistributed).

### LOOK-UP ONLY (never republished as columns)
- **MBFC** — ToU prohibits redistribution; API sold commercially. Scrape privately for validation + selection; publish only agreement statistics.
- **NewsGuard, Ad Fontes, GDI** — proprietary. Validation targets and qualitative benchmarks only.
- **Lin per-provider MBFC/AFM columns** — do not re-redistribute; the derived `pc1` aggregate is defensible but we prefer our own recomputed reliability latent.

### VERIFY BEFORE USE
- Décodex (contact Le Monde), euro|topics ToS (inventory use only mitigates), OpenSources license intent, Robertson-Dataverse "research only" clause, NELA-GT current status (de-accessioned; MBFC-derived — avoid).

**Rule of thumb:** publish *facts and our computations* (domain lists, our latent scores, debunk counts, CC0/CC-BY inputs); keep *proprietary editorial ratings* private and report only how well our open scores agree with them.

---

## 5. Multilingual / EU coverage plan

1. **Inventory (frame):** union of euro|topics (32 countries, 500+ outlets), Media Cloud country collections, and per-country Wikipedia "list of newspapers/most-read media". Target broad breadth — FR, DE, IT, ES, CH, UK, NL, PL, PT, SE, DK, NO, FI, IE, GR, BE, AT — not just the big five. Add each country's top-reach outlets + its known low-reliability/disinfo outlets (seed the low end from Iffy + EUvsDisinfo outlet field + Décodex FR).
2. **Reliability layer (native):** RSP per-language editions (FR/PT/SV/…), Décodex (FR), ClaimReview debunk counts from national fact-checkers (Correctiv DE, Pagella IT, Maldita+Newtral ES, Full Fact UK, Décodeurs FR, AFP multi). Lasser/Lewandowsky adds DE mainstream; misinformation_domains adds IT.
3. **Signed-bias layer (native):** Wikidata P1387 seed for high-profile EU outlets; audience ideal-point estimation for the rest, bridged via CHES parties. Cross-national bridge items (BBC, Euronews, Politico EU, RT editions) tie country axes together.
4. **Validation:** for the four highest-weight markets (FR/DE/IT/UK, where NewsGuard + MBFC are strongest) privately benchmark our open scores against NewsGuard/MBFC to report coverage and agreement — the Lin-style correspondence analysis, now on the *signed* axis and *in Europe*, which no one has established yet.

**Honest gap:** signed bias for non-English EU mainstream is the single scarcest attribute anywhere. There is no open, complete, signed EU outlet table; step 3's audience estimation is the only route and it is real work with cost/access risk.

---

## 6. Phased build plan (with risks)

**Phase 0 — Frame & keys (low risk).** Build the EU+US outlet inventory (euro|topics + Media Cloud + Wikipedia lists), canonicalize registrable domains, disambiguate split brands. *Risk:* euro|topics scraping ToS — mitigate by using it as inventory only.

**Phase 1 — Open reliability latent (low-med risk).** Ingest Iffy, RSP (multi-lang), CRED-1, misinformation_domains, ClaimReview counts. Fit the ordinal Bayesian IRT reliability model with anchors + uncertainty. Deliverable: publishable reliability table with CIs. *Risk:* most open reliability inputs trace to MBFC (correlated errors, not independent) — mitigate by weighting the MBFC-independent signals (RSP consensus, ClaimReview counts) and reporting source provenance per domain.

**Phase 2 — Signed bias, seed + US anchors (low-med risk).** Wikidata P1387 seed for all EU; calibrate US continuous axis from AllSides + Robertson bundle. Deliverable: partial signed table (dense US, sparse EU categorical). *Risk:* NC/SA license propagation — decide the umbrella license now.

**Phase 3 — Signed bias, native EU estimation (HIGH risk / main effort).** Collect EU outlet audience data (X following/sharing), run `mediascores`/`tweetscores`, bridge via CHES + cross-national items. Deliverable: continuous signed EU outlet scores with CIs. *Risks:* (a) X/API access cost & ToS — scope tightly, start with top-reach outlets per country; (b) cross-country scale bridging is methodologically nontrivial — validate with bridge items; (c) small outlets lack enough sharers — fall back to Wikidata prior and flag method.

**Phase 4 — Validation & release (med risk).** Privately benchmark both axes vs MBFC/NewsGuard/Ad Fontes on overlapping outlets; report agreement (correspondence analysis, à la Lin, extended to signed + EU). Publish open core (CC BY-SA or dual-file), a datasheet documenting provenance/licenses/uncertainty per column, version + date stamps. *Risk:* accidental redistribution of proprietary columns — enforce a "no MBFC/NG/AF/GDI values in published files" lint before release.

---

## 7. Open questions

1. **License umbrella:** accept CC BY-SA (share-alike, from RSP/misinformation_domains) and CC BY-NC (from AllSides) on the combined product, or isolate SA/NC-derived columns into separate files to keep a permissive core? This choice constrains everything downstream.
2. **Décodex:** can Le Monde grant redistribution? It is the only true FR domain-reliability list — high value if usable.
3. **How much MBFC independence is enough?** Since Iffy/CRED-1/NELA/Lin all descend from MBFC, is our "open" reliability latent actually independent, or a laundered MBFC? Need to quantify the MBFC-independent signal share (RSP + ClaimReview).
4. **Signed EU estimation feasibility & budget:** is X audience data attainable at acceptable cost, or do we fall back to Wikidata-seed + expert coding + MBFC-numeric-as-private-validation only (weaker, but no data collection)?
5. **Cross-country signed comparability:** are bridge items (BBC/Euronews/RT) sufficient to place French and German outlets on one axis, or should signed scores stay *within-country* (rank/percentile per country) for validity?
6. **Update cadence & hosting:** Zenodo/OSF versioned releases; how often to refresh ClaimReview/audience layers?
7. **Do we need continuous MBFC-style bias at all for our goal?** For source *selection* spanning the space, coarse signed buckets may suffice; continuous signed is mainly for the later veracity-vs-reliability validation — scope Phase 3 accordingly.
