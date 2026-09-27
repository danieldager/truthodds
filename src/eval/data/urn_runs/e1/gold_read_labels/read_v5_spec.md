Read `READ_SYS` (line 71-86), `read_doc` (370), `aggregate_reads` (326), `select_regions` (240), and the record schema written in `run_claim` (479-518).

---

# read-v5 spec

## 0. Baseline numbers I'm optimizing against

184 diagnosed reads, 63 errors = **34.2% read error rate**. Error mass by group:

| group | modes | n | % of errors |
|---|---|---|---|
| **A. directional over-call on off-claim / scope-mismatched docs** | off_claim_but_topical_called_directional 9, scope_mismatch_object 7, scope_mismatch_geo_time 5, scope_mismatch_subset 2, proposal_vs_enacted 2, nuance_exception 1 | **26** | 41% |
| **B. under-call (real evidence discarded)** | missed_evidence_present 12, evidence_dismissed_as_irrelevant 5 | **17** | 27% |
| **C. mirror counted as evidence** | claim_mirror_as_support 7 | **7** | 11% |
| **D. partial/compound claim judged whole** | partial_claim_judged_whole 5 | **5** | 8% |
| **E. prep/extraction artifact** | truncation_or_prep_artifact 3, genuinely_ambiguous 1 | **4** | 6% |
| **F. misc** | opinion 1, pointer_mismatch 1, other 2 | **4** | 6% |

A and B pull in **opposite directions**. The single mechanism that fixes both is to **decouple citation from direction**: a generous relevance floor (cite anything touching a claim slot → `neutral`) plus a strict directional bar (every slot the claim names must match). Everything below is built on that split. Group A is also exactly what the 213-read audit saw as TRUE-refutes 65% wrong / CONT-refutes 33% wrong ("related-claim scope mismatch"); group C is FALSE-supports 63% wrong. TRUE-supports being 100% correct is the constraint: **do not add anything that raises the bar for a doc that matches all slots.**

---

## 1. Ranked fixes

**F1 — Slot test before any direction.** *(kills A: 26 errors; the 65% TRUE-refutes and 33% CONT-refutes wrong)*
Fix the claim's slots (actor/speaker, act/object/scheme/wording, place/jurisdiction, date/period, quantity+population+measure, each conjunct). `supports`/`refutes` require every slot the claim *names* to match; a different scheme, product, country, date, event, population, category or measure → `neutral`. Paired with an explicit **synonym/co-reference clause** (brand = common = scientific name, acronym, definite description, renamed entity) so the bar doesn't fire on paraphrase.
*Expected:* directional-error rate down ~40% relative; overall read error 34% → ~23%.
*Risk:* legitimate support downgraded when the doc matches semantically but not lexically. The synonym clause is the mitigation; measure via the supports→neutral flip count on reads diagnosed `none` (§4 metric 2).

**F2 — Mirror is never support.** *(kills C: 7 errors; the dominant FALSE-supports failure)*
A doc whose only claim-bearing text restates the claim — repost, headline, video title, "X reported that…", "reportedly/promises to", unsourced assertion — is `neutral`, cited in the neutral list. Carve-out for **first-party/primary** docs (the institution's own page about the object it hosts; a report quoting the named speaker's own words; an official/dated record) which *are* evidence.
*Expected:* biggest single reduction in the false-claim supports bias, which is what corrupts the urn's positive LR. FALSE-supports wrong-rate 63% → target <25%.
*Risk:* the carve-out is the failure surface — mislabeling a genuine agency page or on-record quote as a mirror loses real support. Keep the carve-out sentence explicit and short; watch for supports→neutral flips on `.gov`/first-party domains in the A/B.

**F3 — Relevance floor: cite first, then decide direction.** *(kills B: 17 errors)*
If any shown sentence names a claim slot — including a different operationalization of the same measure over the same population, a definite description of the actor, or the claim's outcome variable under a different metric — it **must** be cited and the doc is at least `neutral`. `irrelevant` requires that no slot appears. `neutral` requires ≥1 cited id (else `irrelevant`).
*Expected:* under-call errors down ~60%; usable-evidence yield up; `irrelevant` share falls.
*Risk:* neutral inflation → more docs in the urn denominator carrying no direction, diluting the fitted LRs. This is honest, but **the Truth-Odds LRs must be refit on v5 output** (see §3.7).

**F4 — Compound claims: all conjuncts or neutral.** *(D: 5 errors)*
`supports` only if every asserted component is evidenced; a doc confirming the event but silent on the attributed reason/quote/quantity is `neutral`. `refutes` if it defeats any one conjunct.
*Expected:* −5 errors; also suppresses a chunk of A (the "event confirmed, motive assumed" cases).
*Risk:* neutral inflation on multi-clause claims, which are common in this corpus.

**F5 — Modality ladder.** *(proposal_vs_enacted 2, nuance_exception 1, plus part of A)*
Four one-liners with high leverage: plan/proposal/pre-clinical/"on track to" ≠ enacted outcome; authority ≠ obligation; a degree ("reduced") ≠ an absolute ("no/never/without any"); a point statistic ≠ a trend claim ("more than ever"). Plus: exceptions/waivers **confirm** the rule exists.
*Expected:* −3 to −5 errors, essentially free (5 clauses).
*Risk:* negligible.

**F6 — Document is not the arbiter.** *(opinion 1, plus off-claim cases)*
Don't read direction off a verdict line ("this claim is false"), a Myth/Fact label, or an advocacy stance — use the underlying facts, and check they address *this* claim's wording, not a neighbouring formulation.
*Expected:* small n here but this is the failure that most contaminates the urn (fact-check verdict leakage under a date ceiling that already tags `fc_domain`).
*Risk:* none.

**F7 — Pointer/direction coherence.** *(pointer_mismatch 1)*
Directional label ⇒ matching list non-empty. **Enforce in code** (§3.1); one prompt line only.
*Expected:* contract violations → 0.
*Risk:* none.

**F8 — Junk vs irrelevant discipline.** *(part of E + "other" 2)*
`junk` = unreadable (bot/cookie/paywall wall, nav shell, link farm, table/CSV dump, index page, error page). `irrelevant` = readable prose about a different subject. Keeps retrieval-quality separable from reader-quality in the urn.
*Expected:* −2 to −3 errors; makes the run diagnosable.
*Risk:* none.

**F9 — Truncation guard.** *(E: 3 errors)*
Prompt line: never derive a direction from a sentence cut at the point where the decisive detail would appear. **The real fix is code-side** (§3.4–3.5): mark truncated sentences, strip boilerplate before windowing.
*Expected:* −3 errors, mostly from the code half.
*Risk:* the code half changes the shown text, so it cannot ride in the paired A/B — run it as a separate arm.

**Not adopted:** ~50 of the proposed rules are per-case restatements of F1–F3 (each "check the jurisdiction / the scheme / the population / the speaker" rule is F1). Adding them individually would triple the prompt for no marginal coverage. Also rejected: every rule of the form "state X in the rationale" — **the contract has no rationale field** (see §3.8).

---

## 2. read-v5 system prompt (drop-in)

```python
READ_PROMPT_V = "read-v5"

READ_SYS = """You judge ONE web document against ONE factual claim for an academic misinformation-research measurement. You see the claim, then the document as numbered sentences. Judge STRICTLY from what the document text asserts — never from your own knowledge of the claim.

First fix the claim's SLOTS: who (actor, speaker), what (act, object, scheme, product, exact wording), where (jurisdiction, place), when (date, period), how much (quantity, population, measure), and each separate thing it asserts. A slot MATCHES even when the document names it differently — synonym, common/brand/scientific name, acronym, later name, or a definite description of the same person or event. A slot does NOT match a different scheme, product, place, country, date, event, population, category or measure, however similar.

Output JSON with exactly four fields.

"direction" — the document's overall bearing on the claim. Exactly one of:
- "supports": it asserts, reports or evidences that the claim AS STATED is true, with EVERY slot the claim names matched and EVERY thing the claim asserts covered. Minor caveats still count as supports.
- "refutes": it asserts a fact incompatible with the claim as stated — contradicting it on a slot the claim names, or defeating one of the things it asserts. Not merely making it implausible.
- "neutral": on-claim but not settling it. Use this whenever the evidence is ADJACENT: a different scheme, place, date, event, population, subgroup or measure; only one part of a multi-part claim; a plan, proposal, bill, investigational product or "on track to" where the claim asserts an accomplished fact; an authority to act where the claim asserts an obligation; a degree ("reduced", "less") where the claim asserts an absolute ("no", "never", "without any"); a single point value where the claim asserts a trend or a first-ever; an opinion, advocacy stance or characterisation; or a relay of the claim.
- "irrelevant": readable content in which NO slot of the claim appears — a different subject, however thematically adjacent, and however many claim words it happens to reuse.
- "junk": no readable propositional content — cookie/consent/paywall/bot-check walls, navigation shells, link farms, index or table-of-contents pages, table/CSV dumps with no prose, error pages.

"support" / "neutral" / "against" — lists of sentence NUMBERS, AT MOST 8 per list, most probative first. Cite a sentence if it asserts something bearing on a slot of the claim — including a figure over the claim's population under a different operationalisation, or the claim's outcome under a different metric. Put a sentence in "support"/"against" ONLY if it asserts or denies the claim's own proposition with the named slots matched; every other on-claim sentence goes in "neutral". Background about the subject, biography, definitions and related events belong in NO list; most sentences of most documents belong in no list. If direction is "supports" or "refutes", the matching list must be non-empty. If direction is "neutral", "neutral" must be non-empty — if you cannot cite one sentence, the answer is "irrelevant". If direction is "irrelevant" or "junk", all three lists are empty.

Also:
- MIRROR: a document whose only claim-bearing text restates the claim — a repost, headline, video title, caption, "X reported that...", "reportedly", "promises to" — with no independent named source, official statement, dated record or data of its own is NEVER "supports"; cite it in "neutral". This is not a mirror: an institution's or agency's own page about the thing the claim names, or a report giving the named speaker's actual words — those are first-party evidence.
- The document is not the arbiter. Never take a verdict line ("this is false"), a Myth/Fact label, or an advocacy stance as evidence; use the underlying facts it reports, and check those address THIS claim's wording rather than a neighbouring version of it.
- Exceptions, waivers and exemptions to a rule CONFIRM the rule exists.
- Never take a direction from a sentence truncated ("...") where the decisive detail would be, from a number that merely coincides with one in the claim, or from the URL or page title alone.
- Resolve dates written as "this year", "Saturday", "next" against the document itself; if the resolved date differs from the claim's date, the direction is at most neutral.

You may be shown ONE CONTIGUOUS REGION of a longer document; sentence numbering shows its position, and a leading lede block may be prepended for context (numbering will jump) — its sentences are citable. Judge only WHAT YOU ARE SHOWN; never speculate about unseen parts. A search-result snippet is judged the same way, but a snippet asserts only what it literally says.

Respond JSON only: {"direction": "...", "support": [..], "neutral": [..], "against": [..]}"""
```

Length: ~870 tokens vs v4's ~610 (+43%). It sits in the constant system prefix, so under the existing `prompt_cache_key=review_url` scheme the delta is paid once per claim (first region) and ~18% thereafter — at K=3 regions × 10 docs, the marginal run cost is roughly +2% total, not +43%.

---

## 3. Changes OUTSIDE the prompt

**3.1 `read_doc` — coerce, don't just flag** (replaces lines 382-390):

- `d in {supports, refutes}` and the matching list empty → `d = "neutral"`, `qc_flag="empty-directional"`. (Today this is unchecked.)
- `d == "supports"` and `len(against) > len(support)` → `d = "neutral"`, flag `direction-count-divergence` (today: flag only, direction kept). Same, mirrored, for `refutes`. This implements "never output supports when the against list is larger".
- `d == "neutral"` and all three lists empty → `d = "irrelevant"`, flag `neutral-uncited`.
- `d == "irrelevant"` with citations → `neutral` (existing behaviour, keep) but flag `irrelevant-with-cites`; `d == "junk"` with citations → `neutral`, flag `junk-with-cites` — today both collapse into one silent branch and the distinction is lost.

**3.2 `aggregate_reads`** — add a partial-document flag: if the doc-level direction is `irrelevant`/`junk` and `prep["skipped_tiles"] > 0`, set `qc_flag="partial-doc-irrelevant"`. The urn must treat that as an **abstention** (unread remainder), not as a negative observation. Also: `DIR_RANK` (line 323) is dead code — unused since v4; remove or wire it.

**3.3 Lede label is a lie.** The prompt claims "a labeled LEDE block may precede it", but `read_doc` (line 371) emits only `[i] sentence` — there is no label, just a numbering jump. Either emit `LEDE:` / `--- REGION ---` separators, or (as done in the v5 text above) describe it truthfully as a numbering jump. Pick one; don't leave the mismatch.

**3.4 Boilerplate strip before windowing (`_clean` / `select_regions` → prep-v8).** Drop lines matching cookie/consent/privacy/terms/subscribe/newsletter/share/nav patterns and member/link rosters *before* BM25 scoring, so boilerplate never consumes the shown budget. If fewer than 4 substantive sentences survive → emit `direction="junk", qc_flag="boilerplate-only"` without a read (saves a call). Fixes the `genuinely_ambiguous` and part of `truncation_or_prep_artifact` mass.

**3.5 Truncation visibility.** `MAX_SENT_CHARS=320` truncates silently (line 140). Append `"…"` to truncated sentences and record their ids in `prep["truncated_ids"]` so F9 can actually fire and so the artifact class is countable.

**3.6 Title as sentence.** Keep the page `<title>`/H1 as a citable leading sentence — several diagnosed "returned junk" cases had the claim's content only in the headline that `_clean` discarded.

**3.7 Code-side no-anchor gate (measure first, enable later).** Compute `anchor_hits = _anchor_terms(claim) & _anchor_terms(shown_text)`; when empty, the read is near-certainly `irrelevant`. **Do not act on it yet** — synonym cases (guyabano/soursop) are exactly the F3 failures. Log it as `qc_flag="no-anchor-code"` in the A/B, measure how often the model disagrees, and only turn it into a skip if disagreement is <2%.

**3.8 Truth-Odds consequence (important).** v5 deliberately moves probability mass from directional → `neutral` and from `irrelevant` → `neutral`. **LRs fitted on v4 reads do not transfer.** The weight fit must be re-run on v5 output before any headline urn number. Also add a run-level metric block: `directional_share`, `neutral_share`, `irrelevant_share`, `junk_share`, and the qc_flag histogram, printed at the end of `main()`.

**3.9 Not doing:** a 5th `note`/`why` field. ~15 of the proposed rules say "state it in the rationale" — that would break the contract and add output tokens on every read. If QC rationale is wanted later, add it as an optional field ignored by `aggregate_reads`, gated behind a `--rationale` flag for audit runs only.

---

## 4. Validation plan (offline, paired, no search/scrape)

**Harness:** `eval/scripts/build_eval/read_ab.py`, following the `cap_sweep.py` pattern (`import eval.scripts.build_eval.evidence_urn_run as R`).

**Exact input reconstruction** — the run file already stores everything needed:
- per result: `sent_ids[]`, `sents[]` (aligned, sorted), and `prep["read_regions"] = [{"span":[a,b],...}]`.
- rebuild region *r* as `ids = ([1..min(3,n)] if span[0] > 3 else []) + list(range(span[0], span[1]+1))`, `sents = [map[i] for i in ids]` where `map = dict(zip(sent_ids, sents))`; non-windowed docs (`prep["windowed"] is False`) are a single region = all `sent_ids`.
- rebuild `claim_block` from `claim_text` + `claim_date` using the same f-string (line 454).
- **Assert** `sorted(set(union of all region ids)) == sent_ids` per result before scoring; drop and report any result that fails reconstruction. Then call `R.read_doc` with `READ_SYS_V5`, temperature 0, same `VERIFICATION_MODEL`. Zero search calls, zero scrapes.

**Samples:**
- **S1** = the 184 diagnosed reads (has per-read ground truth).
- **S2** = 400 additional reads sampled from `results-00.jsonl`, stratified by v4 direction (100 each supports/refutes/neutral/irrelevant) — the over-conservatism probe.

**Metrics:**
1. **Diagnosed error rate on S1.** Re-diagnose v5 outputs with the *same* diagnosis prompt and the same taxonomy, blind to arm. v4 = 34.2%. Target ≤ 20%. Report the per-mode delta table (must show A/B/C/D collapsing, and must show no new mode appearing above n=4).
2. **Flip matrix v4→v5** on S1 ∪ S2. The over-conservatism number is: `supports→neutral` and `refutes→neutral` **among reads diagnosed `none`**. Budget: ≤10% of correct directional reads may flip.
3. **Sign accuracy proxy** (needs no re-diagnosis, so it can be computed on all of S2): for CAL claims with veracity ≥4, count `refutes` reads as wrong-sign; for veracity ≤2, count `supports` reads as wrong-sign. v4 baselines: FALSE-supports 63% wrong, TRUE-refutes 65% wrong. Target: wrong-sign directional reads −≥40% relative.
4. **Yield:** directional share must not fall more than 15% relative; `irrelevant` share should fall (F3); `neutral` share will rise — record it for the LR refit (§3.8).
5. **Contract violations:** empty-directional / uncited-neutral / count-divergence rates, pre-coercion. Should approach 0.

**Decision rule — adopt v5 iff all three hold:**
(a) wrong-sign directional rate (metric 3) drops ≥40% relative; (b) right-sign directional rate drops <15% relative; (c) diagnosed error rate (metric 1) drops ≥10 absolute points.

**Significance:** McNemar on the S1 paired correct/incorrect vector. With 63 v4 errors, fixing ≥25 while breaking ≤6 is significant at p<0.001; report the discordant cells, not just the rate.

**Attribution ablation** (only if adopted, ~130 calls): re-run the 63-error subset with v5-minus-mirror-clause and v5-minus-slot-paragraph to confirm F1 and F2 carry the gain and the prompt isn't winning by generic verbosity.

**Cost & process:** ~584 reads for the main A/B + ~130 ablation + the re-diagnosis pass. Per project convention, write the cost + ETA line to `eval/data/run_ledger.md` and walk `docs/run_checklist.md` **before** launching; log actuals after.

**Out of the paired A/B:** §3.4/3.5/3.6 change the *shown text*, so they break the paired design. Run them as a separate prep-v8 arm on a fresh sample of docs whose `prep["skipped_tiles"] > 0` or whose v4 direction was `junk`, scored on junk-vs-irrelevant precision only.

Files: `src/eval/scripts/build_eval/evidence_urn_run.py` (prompt at line 71, `read_doc` at 370, `aggregate_reads` at 326, `select_regions` at 240); pilot data at `src/eval/data/urn_runs/e1/results-00.jsonl`; new harness `src/eval/scripts/build_eval/read_ab.py`.