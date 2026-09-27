# Enrichment + Audit — Spec

*Owner: Daniel. Created 2026-06-27 (clog/270626). The single image-aware labeling/audit step that
builds the **Stage-1 (selection), Stage-2 (extraction) and Stage-3 (verdict)** eval datasets from the
core fact-check pool. This determines the quality of everything downstream — it is core methodology.*

> **Status: DESIGN LOCKED, not built.** Field rubric + architecture agreed with Daniel. Draft prompts
> in §9 await ratification of the `in_scope` language + a stratified smoke before any full run.

---

## 0. Principle

One labeled/audited core → three stage datasets, each built by its **own** builder with a documented
inclusion funnel and its **own** provenance ledger. The audit *tags* the data objectively; each builder
*derives* its labels deterministically from those tags. No hand-edited parquets — every label traces to
a model, a human, or a rule (see §6).

---

## 1. Architecture — 2 audits, 3 datasets

```
                          ┌───────────────────────────────────────────────┐
  core pool               │  PASS A · BLIND POST AUDIT (image; gold-blind) │
  dataset_*.parquet  ───► │   labels the POST as the gate/extractor sees   │ ─► Stage-1 builder (gate splits)
  (~11,813 rows)          │   it — NO gold, NO rating                       │ ─► Stage-2 builder (extraction set)
                          │   pop: ~4,524 has-input rows                   │
                          │   fields: topic, in_scope, has_claim,          │
                          │     claim_locus, readable, note                │
                          └───────────────────────────────────────────────┘
                          ┌───────────────────────────────────────────────┐
                     ───► │  PASS B · SIGHTED VALIDATION AUDIT            │ ─► Stage-3 builder (verdict set)
                          │   sees claim+rating+gold AND post+image;        │    + dataset-validity ledger
                          │   validates the ground truth                    │
                          │   pop: all ~11,813 (post/image when present)   │
                          │   fields: judged_axis(img-aware), is_satire,   │
                          │     veracity_agrees, suggested_veracity,        │
                          │     rating_subtype_ok, claim_matches_post,      │
                          │     image_supports_claim, note                 │
                          └───────────────────────────────────────────────┘
  (+ image serialization, WS1: Qwen3-VL over claim_locus∈{image,both} → `image_serialization`, Stage-2/3)
```

**Why blind + sighted (still 2 passes, 3 datasets):** the post-side audit does two genuinely different
jobs. **Pass A is BLIND** (no gold, no rating) so it labels the post *exactly as the production gate/
extractor sees it* — which makes it a valid *independent* check of the gate and a clean source of Stage-1
negatives (feeding the gold would inflate `in_scope`/`has_claim`). **Pass B is SIGHTED** — it sees the
claim, rating, gold AND the post+image, and *validates the ground truth* (does the post match the gold
claim, does the image support it, is the rating→veracity translation right, what axis was judged). Because
Pass B is sighted, `judged_axis` is properly **image-aware**, and `claim_matches_post` supersedes the
weaker fetch-time `raw_related` gate. **Cost accepted:** for has-input *image* rows (~927) the image goes
through the VLM **twice** (blind label + sighted validation) — the price of separating "what the gate
sees" from "is the ground truth valid", which is the whole point of the audit.

**Models (both passes):**
- **Labeler = `Qwen/Qwen3-VL-235B-A22B-Instruct`** (smartest multimodal on DeepInfra; Session-1's audit
  model → cross-session consistency). Sees `claim_text` + `original_rating` + image (when present).
- **Cross-check = `google/gemma-3-27b-it`** — same rubric, per-field agreement measured; disagreements
  drive `borderline` (§3). It is the **uncertainty signal**, replacing the model's useless self-reported
  confidence (Session-1: `confidence` was always "high").

**Population / field-applicability** (one model, but fields apply per row type):

| pass | judges | applies to | feeds |
|---|---|---|---|
| **A · BLIND** (topic, in_scope, has_claim, claim_locus, readable) | the post, gate-faithfully (no gold) | **~4,524 has-input** | Stage-1, Stage-2 |
| **B · SIGHTED** (judged_axis, is_satire, veracity_agrees, suggested_veracity, rating_subtype_ok, claim_matches_post, image_supports_claim) | the claim+rating+gold, validated against the post+image | **all ~11,813** (post/image fields null when no post) | Stage-3 |

For empty-input rows Pass A doesn't run and Pass B's `claim_matches_post`/`image_supports_claim` are
`null` (no post to validate against); `judged_axis` falls back to text-only there.

---

## 2. Field rubric (definition / justification / stage gating)

### ⭐ political_implication (bool) — the most important field; mines negative samples  *(Pass A, blind)*
**Definition — does the claim's veracity drive POLITICAL POLARIZATION?** (global sense; Daniel 280626.)
IN if, believed, it would **sow division** — make one political side/group look unreasonably bad, a favoured
side unreasonably good, or shift sentiment for/against a group or those in power. **Decided by IMPLICATIONS,
NOT topic:** IN even when the topic isn't explicitly political if the claim carries political implications
(a fabricated migrant-crime story; a weaponised health/science claim; a culture-war claim); OUT even when
the topic is political or names political actors if the veracity carries no polarization implication (a
politician's mundane biography; a neutral government-process fact; a true uncontested event; sports; a
celebrity's private life; a consumer product). Any country, any political division (left/right, pro/anti-
government, immigration, religion/ethnicity-as-politics). **Materially narrower than the earlier "societal
consequence" scope** — ties the filter to the polarization mechanism the project tests. (Supersedes the
280626 public-figure rule: a public-figure claim is IN only when it carries a *political-polarization*
implication, not merely any non-personal checkable assertion.)
**Justification:** an out-of-scope fact-check is a **gold-quality negative** for the filter (a real post the
filter must reject). Fuzzy scope ⇒ noisy negatives ⇒ mistuned filter.
**Gating:** Stage-1 (filter) — `political_implication ∧ has_claim ∧ readable ∧ ¬obvious_joke ∧ carries_claim`
→ positive; `readable ∧ carries_claim ∧ (¬political_implication ∨ ¬has_claim ∨ obvious_joke)` → **negative
sample**. Stage-2 — slice only. Stage-3 — informational.
**Daniel hand-reviews the decisive borderlines (235↔Gemma disagree on the label) + a sample.**

### obvious_joke (bool) — blatant satire/parody the filter should NOT flag  *(Pass A, blind)*
Daniel 280626: satire by **deceptiveness**, not a blanket rule. `obvious_joke=true` (reads as an obvious
joke/parody no reasonable reader takes as real) → **negative**; a satirical claim hard to distinguish from
genuine (`obvious_joke=false`) → treated as a real check-worthy claim → eligible **positive** (the deceptive
satire we must catch). Blind judgment from the post alone; imperfect by design. The gold `is_satire` (Pass B,
from the rating) stays as **metadata** (misinfo DB + satire-recall reporting) — not the label.

### topic (short string) — the evidence for political_implication
politics|election|health|crime|war|sports|celebrity|product|business|personal|other. **Gating:** not a
gate; a slice/diagnostic. Note scope is implication-based, so topic alone never decides it.

### has_claim (bool) — does the post assert a SPECIFIC, checkable factual claim (text OR image)?
Distinguishes a claim from rhetoric/opinion/fragment/question. Session-1 over-fired on unfalsifiable
rhetoric ("the system is broken") → prompt must require a *verifiable proposition*. **Gating:** Stage-1
positive requires it; political-but-no-claim is still a **negative**. Stage-2 no-claim → nothing to extract.

### claim_locus ∈ {text, image, both, none} — where the checkable claim lives
**Gating:** Stage-2 slice (image-locus is the hard extraction case); selects WS1 serialization targets
(`{image,both}`); Stage-1 signal for image-only gateability.

### judged_axis ∈ {content, attribution, artifact} — image-aware  *(Pass B, sighted)*
Which proposition the fact-checker adjudicated: content (is the fact true), attribution (did X say/do it),
artifact (is the media authentic). Produced in the **sighted** Pass B so it's genuinely image-aware (the
VLM *sees* a manipulated/miscaptioned image); text-only fallback for empty-input rows. `is_attribution`
is **derived** from `judged_axis=="attribution"` (no separate field) — carries the nudge-safety
attribution open-question (a TRUE attribution of a FALSE claim is not veracity-1 for the post).
**Non-exhaustive → a recall floor.** **Gating:** Stage-2 — `attribution`
→ `needs_attribution` should fire, `artifact` → `needs_authenticity` should fire (recall floor); slice by
axis. Stage-3 — content-axis is the headline verdict eval; feeds `rating_subtype=altered_media` (artifact)
and the attribution open-question. Stage-1 — report gate recall **by axis** (33% of positives are artifact).

### claim_matches_post (bool|null) — does the post actually assert the GOLD claim?  *(Pass B, sighted)*
The sighted, image-aware post↔gold check — **supersedes the fetch-time `raw_related` gate** (which used a
weaker text-only model). `null` when no post. `raw_tier='quote'` (body_quote) is aligned-by-construction
(selected to contain the claim span). **Justification:** protects Stage-2's extraction gold from spurious
misses when recovery grabbed a mismatched/wrong post, and validates the dataset row coheres.
**Gating:** Stage-2 strict-extraction subset = `claim_matches_post`; misaligned → separate slice.

### image_supports_claim (bool|null) — does the post IMAGE depict/support the claim?  *(Pass B, sighted)*
`null` when no image. Catches wrong/mismatched image attachments (a clean subject photo where the claim is
about a fabricated post). **Gating:** Stage-2/3 — image evals exclude/flag `image_supports_claim=false`.

### readable (bool) — legible / interpretable (false = blank, garbled, link-only)
Session-1: rarely false (harvest pre-filters). Real job: derive `drop`. **Gating:** Stage-1/2 `readable=false`
→ drop; Stage-3 irrelevant.

### is_satire (bool) — claim originated as / is labelled satire  *(verdict audit; sees rating + publisher)*
Metadata flag so satire is included/excluded per run, not silently folded into a class. **Gating:** Stage-3
— dropped from the verdict gold by default (revisit specific cases in audit); Stage-1 informational.

### veracity_agrees (bool) + suggested_veracity (1–5) — audits the VERDICT gold  *(Pass B, sighted)*
Checks WS0's `original_rating → gold_veracity(1–5) + rating_subtype` translation (NOT the old 4-class).
**Non-circular:** the audit *checks* a gold the frozen rule table *assigns*; the judge sees publisher +
rating + claim (+ image), never the evidence or our verdict. **Gating:** Stage-3 — disagreement → human
review / use `suggested_veracity`; the clean 1–5 gold is the verdict eval target. Only **veracity** has a
gold; the 3 confidence dims are validated by construct-validity (verdict_eval_plan WS4), not gold-audited.

### rating_subtype_ok (bool) — protects the gold-3 split  *(verdict audit)*
Special check on `rating_subtype=unprovable` rows: genuinely "no evidence either way", not a mislabeled
resolved claim. The `mixed` vs `unprovable` split is what the confidence construct-validity test (H1/H2)
rests on. **Gating:** Stage-3 — dirty split rows → review.

### note (free text, both passes) — one-sentence reason, especially on disagreement/borderline.

---

## 3. Derived fields — NOT asked of the model

Session-1 found the model's *direct* `recommend` (leaks truth judgments) and self-reported `confidence`
(always "high") unreliable. Both are **derived deterministically** in the builders:

**recommend ∈ {positive, negative, drop}** (Stage-1 label):
```
positive = readable ∧ in_scope ∧ has_claim ∧ ¬is_satire
negative = readable ∧ (¬in_scope ∨ ¬has_claim ∨ is_satire)
drop     = ¬readable   (or empty-input / not a gateable post)
```
**Satire is a should-EXCLUDE class (Daniel 280626):** the tool must *successfully exclude satire from
selection* → no nudge. A satire post is a **negative even when it is in-scope and carries a claim** (political
satire usually is). `is_satire` (gold, from the rating; Pass B) defines the label; the **blind** gate must
detect it from the post alone — hard, since stripped-of-context satire has no markers (that's why it fooled
people and got fact-checked) → may need a dedicated satire signal (satire-domain list, parody-account, "not
real" watermark). **Report satire-exclusion recall separately** — don't fold satire into the ordinary
out-of-scope negatives. `is_satire` is also carried as **metadata into the misinfo DB** (no nudge, but worth
recording). Stage-3 still drops satire from the veracity gold (kept as the same flag).

**confidence = borderline** (→ human-review queue) if ANY of:
- 235B ≠ Gemma on {in_scope, has_claim, judged_axis, derived `recommend`, gold_veracity}, OR
- a component-conflict fires (e.g. `in_scope=true` but `topic`∈OOS set; `claim_locus=none` but `has_claim=true`;
  model raw-recommend ≠ derived recommend), OR
- either model self-flags borderline.

Daniel reviews **every borderline + a sample of high-confidence** rows (HTML tool, cf. Session-1).

---

## 4. Drop & scrub rules

- **Drop** (Stage-1/2): `readable=false`; empty-input.
- **Drop** (Stage-3 verdict gold): `is_satire` (default); uncheckable (`has_claim=false`).
- **Leaky-raw scrub:** any recovered raw (`raw_tier='post'`/`'quote'`) the audit flags as carrying verdict/
  fact-check framing → scrub framing+verdict words (generalize `recover_image_captions.py`), paraphrase if
  needed → **keep only if it stays distinct from `claim_text`** (reuse the validated distinctness; raws ran
  token-Jaccard ~0.13). If scrubbing collapses it onto the normalized claim → **drop** (it's leaked gold,
  not a genuine raw→normalized pair). Mostly fires on `page_meta` og:descriptions, not pure body_quotes.

---

## 5. Contamination handling — images

- **Leakage is closed by construction:** `resolve_media_urls` only emits `x_media` (original post media —
  safe) and `og_image` (the *source* page's preview, NOT a fact-check graphic). No `fc_article` kind exists.
- **og_image (232):** keep — a real fraction is the claim's own media (inspection 270626). Add a one-line
  **md5 hash-dedup** to drop recurring platform/site placeholders (~8% of og_images). The post-audit's
  `claim_locus` sorts the rest (placeholders → `claim_locus=none`, fall out). Optional deterministic guard:
  `og_image` whose source domain ∈ fact-check-publishers → exclude (≈0 rows today).
- **No general LLM leak-detector** — unneeded given the above.

---

## 6. Provenance ledgers & reproducibility

One **append-only ledger per builder** (gate / stage2 / verdict; cf. `gate_dataset_provenance`). Each row =
one decision: `key(review_url/uid) | field | old → new | source(rule|audit_235B|audit_gemma|human|derived)
| reason | run_id`. Each builder is a **pure function**: `raw_harvest + ledger → final dataset` — no manual
parquet edits, re-run = byte-identical. Human borderline calls land as `source=human` rows (not lost).
Combined with frozen hash-stable splits (`md5(review_url)%k`), the whole chain harvest→splits is reproducible
and every label traces to its origin. **Reproducibility is the top constraint (Daniel).**

---

## 7. Verdict gold (WS0) — owned by `verdict_eval_plan.md`

Stage-3 gold = `gold_veracity 1–5` + `rating_subtype`, re-harmonized from **raw `original_rating`** in
`harmonize.py` (NOT the 4-class `harmonised_label`, which lost the gradation). This audit (Pass B) is
WS2 of that plan. See `verdict_eval_plan.md` for the veracity map, leakage guardrails, and WS4 eval harness.

---

## 8. Adjacent workstreams (non-blocking — feed rows in, don't gate the audit design)

- **`fc_rationale` harvest (confirmed):** a gold column = the fact-checker's OWN reasoning/summary (distinct
  from `body_quote` = claimant words and `context` = the 110-char dek). Per-source scraper + backfill over
  existing `review_url`s (`backfill_*` pattern); API sources (afp/aap/factcheckorg) get whatever the API
  returns. **Eval-only — never fed to the verifier** (leaks verdict + reasoning).
- **IDEA-014 — justification-vs-justification eval:** compare our Stage-3 `justification` to `fc_rationale`
  (same decisive reason?), distinct from WS4 grounding (which checks our justification vs *retrieved
  evidence*). LLM-judge agreement metric, human-validated. → verdict_eval_plan WS4 third sub-axis.
- **fc-page image recovery (viable, selective — inspection 270626):** for empty-input / no-post-image rows,
  the review page often reproduces the **clean** claim media (no verdict overlay) — high yield on **Snopes /
  Lead Stories**, low on PolitiFact/FullFact/AAP (editorial press-photos), none on AFP/API (scrape-blocked).
  Needs a **VLM classifier** per page image → `{claim-media (keep) | editorial-illustration (drop) |
  verdict-graphic (drop)}`. Separate recovery harvester; sequence after the rubric locks.

---

## 9. Draft prompts — FOR RATIFICATION (do not run until signed off)

### Pass A — BLIND POST AUDIT (Qwen3-VL-235B + Gemma-3-27B; image when present; NO gold/rating)
```
SYSTEM:
You audit a dataset of social-media posts for the evaluation of an automatic fact-checking system's
selection gate. Judge each post OBJECTIVELY and ONLY from what the post itself shows — never decide
whether any claim is true, and you are given no verdict. The checkable claim often lives in the IMAGE
(a screenshot, a fabricated/quoted headline, a miscaptioned or manipulated photo) — read text AND image.

SCOPE = matters of SOCIETAL CONSEQUENCE: politics & elections, government policy/spending, the economy,
corruption & bribery, war & geopolitics, terrorism, immigration, public health & safety, crime,
disasters, science & climate. OUT of scope: sports, entertainment/celebrity/music, consumer products/
brands/business gossip, personal life, jokes, trivia — even when specific or sensational. Topic decides
scope, not specificity (a specific sports score or celebrity marriage is still OUT). PUBLIC FIGURES &
CELEBRITIES: a claim about a public figure or celebrity is IN whenever it makes a CHECKABLE factual
assertion bearing on anything BEYOND their private personal life — politics, law, crime, corruption,
finance/taxes, public conduct, an official action, a verifiable scientific/societal claim, OR a
conspiracy/defamation/hoax imputing wrongdoing or a notable false event (a fabricated crime, a doctored
political image, a false claim they funded/said/did something). It is OUT ONLY when the claim is purely
about their private personal life, relationships, physical appearance, health or death, mundane personal
behavior, personal opinions, or pure entertainment/career trivia (a new film, a concert, an award).
Sports results are OUT.
IN examples: "Comey's charges dismissed after Trump's prosecutor misspelled his name" (legal/political);
"Epstein files reveal Ellen DeGeneres is a cannibal" (conspiracy imputing crime); "doctored photo of
Ghislaine Maxwell with Melania Trump" (crime + political figures); "Bad Bunny burned an American flag"
(political conduct); "James Hetfield pledged to fund Charlie Kirk's kids' education" (false claim a
celebrity funded a political figure). OUT examples: "Melania Trump holding hands with a man" (private
relationship); "Trump with missing, disheveled hair" (appearance); "a new Godfather film by Coppola"
(entertainment trivia); "Paige Bueckers is 0-4 vs Caitlin Clark" (sports result).

A post HAS A CLAIM only if it asserts a SPECIFIC, verifiable factual proposition (in text or image). NOT
a claim: a fragment/bare label with no predicate ("Footage of a protest in Paris"), a QUESTION ("A road
sign banning drivers who wear glasses?"), an opinion/value judgment, or unfalsifiable rhetoric ("the
system is broken").

Output strict JSON only:
{"topic":"politics|election|health|crime|war|sports|celebrity|product|business|personal|other",
 "in_scope":true|false,
 "has_claim":true|false,
 "claim_locus":"text|image|both|none",
 "readable":true|false,               // legible/interpretable (false=blank, garbled, link-only)
 "note":"<one short sentence; esp. if borderline>"}

USER:  POST TEXT: {raw_context}     [+ image attached when present]
```

### Pass B — SIGHTED VALIDATION AUDIT (Qwen3-VL-235B + Gemma-3-27B; gold + post + image)
```
SYSTEM:
You audit a dataset of fact-checks for the evaluation of an automatic fact-checking system. For each
record you are given the publisher, the claim (the ground-truth claim), the claimant, the publisher's
textual rating, and the rule-assigned gold_veracity (1–5) + rating_subtype — and, when available, the
ORIGINAL post (text and/or image) the claim came from. Validate the record is what we expect; do NOT
re-investigate whether the claim is true.

1. judged_axis — which KIND of proposition the fact-checker adjudicated: content = a worldly factual
   claim (is it true?); attribution = whether a person/entity really SAID/POSTED/WROTE it; artifact =
   whether a MEDIA item is authentic / correctly captioned (Miscaptioned/Altered/AI-generated/Fake-photo,
   and "an image authentically shows X" — artifact whether affirmed OR refuted). If both, pick PRIMARY.
   Use the image when present to tell an artifact case from a content case.
2. verdict-gold check — is rating→veracity right? veracity: 5 clearly-true · 4 mostly-true · 3 not-
   clearly-either · 2 mostly-false · 1 clearly-false. rating_subtype: mixed = contested/half-true/
   missing-context (evidence both ways); unprovable = genuinely no evidence either way — NOT satire/fabrication.
3. ground-truth check (only when a post is provided) — does the post actually correspond to this claim,
   and is the claim a faithful rendering of what the post asserts/shows?

Output strict JSON only:
{"judged_axis":"content|attribution|artifact",
 "is_satire":true|false,
 "veracity_agrees":true|false,
 "suggested_veracity":1-5,            // your veracity if you disagree, else echo the gold
 "rating_subtype_ok":true|false,      // esp. confirm `unprovable` is truly no-evidence, not a mislabeled resolved claim
 "claim_matches_post":true|false|null,    // null if no post
 "image_supports_claim":true|false|null,  // null if no image
 "note":"<one short sentence; esp. if anything disagrees>"}

USER:  publisher: {publisher_site}\n claimant: {claimant}\n textual_rating: {original_rating}\n
       gold_veracity: {gold_veracity}\n rating_subtype: {rating_subtype}\n claim: {claim_text}
       [+ POST TEXT: {raw_context} and image, when available]
```

---

## 10. Build order

1. **This spec** — sign-off (esp. the `in_scope` language §2 / §9). ✓ signed off 270626.
2. **Smoke Pass A + Pass B** on a stratified ~30-row sample (235B + Gemma) → per-field agreement + Daniel
   spot-check → iterate prompt language. ← *here.* *(Frugal; checkpoint before any full run.* Pass B's
   `veracity_agrees`/`rating_subtype_ok` use a **provisional** `gold_veracity` (crude map from
   `harmonised_label`/`rating_value`) until WS0 — directional only; the rest of Pass B is real.)*
3. **WS0 veracity map** in `harmonize.py` (gold construction) — prerequisite for the REAL Pass B verdict audit.
4. On sign-off → **full 2-pass audit** (235B + Gemma) over the core → **per-builder ledgers**.
5. **Rebuild the 3 stage datasets** (gate / stage2 / verdict) from core + ledgers → re-run `audit_harvest`
   + the clean Stage-2 2×2.
6. **Non-blocking:** `fc_rationale` backfill + IDEA-014; fc-page image recovery (Snopes/LeadStories-first).
```
