# Verify-loop version ledger

One entry per iteration of the post-verify loop (`pipeline/verify_tweet_claims.py`). The loop version
is stamped into every run record (`loop_version` in the result dict). Rules:
- Bump the version on ANY behavior change (prompt edit, guard, parameter, control flow).
- Every version entry links its runs, results, and audits. Never overwrite a prior version's
  artifacts — new runs get new out-dirs named `<set>_<version>/`.
- Component prompts keep their own internal iteration numbers from their eval loops (READ v4,
  STEP v4 as of loop v1); the loop version is the unit of comparison across runs.

---

## v1 — 2026-07-10/11 (commit 68e8b6b)

The initial locked loop: OPEN -> Serper (NewsGuard-tier rank) -> TRIAGE (read picks 0-5 +
snippet evidence) -> scrape + guards (origin exclusion, reprint-phrase republication, near-dup
shingles) -> READ v4 (pointer + 5-way stance + republication flag) -> resolve (±3-sentence
windows) -> STEP v4 (ledger + control + verdict) -> redundancy-escalated Exa round.
Config: max_rounds 3 + 1 Exa, cap_tok 2500, serper_k 10, pages cap 5.

**Runs**
- `dev50 main set` (50 checkworthy posts, seed 42, mixed NG): traces
  `eval/data/survey_claims/dev50_verify_trace/`, review doc
  `reports/dev50_verify_review_2026-07-11.html`. 50/50 ok, 146s wall, ~20 posts/min.
  Veracity 5:33 / 4:11 / 3:1 / 2:2 / 1:3, 6 nudges; 48/50 one-round; 1 Exa call.

**Audit** — `docs/dev50_run_audit.md` (10 agents, web ground-truthed)
- Verdict agreement 41/50 (8 disagree, 1 uncertain). Nudge accuracy 45/50; **precision 2/6
  (4 false flags), recall 2/3 (Grayzone missed)**.
- Root cause of all 3 hard false flags: temporal/event misalignment. Other systemic:
  single-voice/syndication-blind supported (~14), READ span violations (10), silent evidence
  loss (10), unsearched-claim closes (3), proposition substitution (6), repub-flag misses.

## v2 — 2026-07-11 (this commit)

Audit-driven fixes, mapping to `docs/dev50_run_audit.md` recommendations 1-6:
1. **Same-event rule** (STEP judging rules) + OPEN date anchoring — evidence must concern the
   same event/period as the post; off-period evidence neither supports nor refutes.
2. **No-unsearched-conclude** — `unsupported` now requires that a query specifically targeted
   the claim (else it stays `open`); STEP's search action emits `targets`; targeted-claim ids
   recorded per run (`targeted` in the result).
3. **Evidence-loss hardening** — every discarded candidate is logged in `dropped` with a
   reason (scrape-failed / too-short / junk-page / too-thin / dup-url / dup-domain / origin /
   near-dup / republication); junk-page detector (bot-block / paywall / error-page phrases,
   <4-sentence stubs); **backfill**: when a triage pick dies, the next ranked hit fills the
   read slot (explicit empty pick list still means read nothing).
4. **READ seg cap** — max 8 cited sentences per evidence entry, enforced in code (first 8,
   sorted; truncations logged). Prompt unchanged.
5. **Syndication-aware independence** (STEP) — same wire story / media group / press release
   across domains = one voice; count voices, not domains. Plus wire-byline republication
   heuristic in code for wire-agency posting outlets (AP/Reuters/AFP).
6. **Proposition-substitution guard** (STEP) — supported requires the full exact proposition
   including causal/agency attributions; an event affirmed without its attribution does not
   support the claim, and an unverified causal hook that is the post's takeaway feeds the
   framing check.

**Runs**
- `main set re-run` (same 50 posts): runs/dev50_main_v2/. 50/50 ok, 132s wall.
- `low-NG set` (50 new posts, NG < 40, seed 42, excludes the 12 low-NG posts already in the
  main set): runs/dev50_lowng_v2/. 50/50 ok, 182s wall.

**Results — main-set diff vs v1** (`scripts/diff_verify_runs.py`)
- **All 4 false nudges fixed**: zerohedge 1->4 (same-event rule kept the claim open; Exa
  retrieved the official ADP release), FDRLST 2->5 (date-anchored query surfaced the Dec 15
  Wisconsin Examiner report), Breitbart 2->5 (2 rounds, correct blotter entry), townhall/Rubio
  3->4 (attribution + conflicting fixed). Nudges 6 -> 2 (the two audit-correct ones).
- infowars Maduro claim: now searched in round 2 and supported (no-unsearched-conclude worked).
- townhall/Hegseth: the spurious wrong-period refutation is gone (refuted -> unsupported).
- Mean rounds 1.04 -> 1.18; 17 verdict changes total.
- **Side effect to watch**: veracity compression 5->4 on ~11 accurate posts (33x5 -> 26x5,
  11x4 -> 22x4) — the stricter independence accounting docks a point for thin corroboration.
  No nudge impact; may reduce contrast between clean outlets. Judge against gold on dev-500.
- **Remaining miss**: Grayzone/GoFundMe still 4/NONE — STEP acknowledged the causal hook is
  unverified but dodged the substitution guard by treating the outlet's own assertion as an
  attribution claim ("the post presents it as his claim"). -> v3.

**Results — low-NG set (first run)**: veracity 5:16 / 4:20 / 3:10 / 2:1 / 1:3, 14 nudges
(28%), misinfo NONE 36 / MISLEADING 8 / UNSUPPORTED 3 / FALSE 3; rounds 1:39 / 2:9 / 3:1 /
4:1; nudges concentrate in Grayzone 4/7, FDRLST 4/5, MintPress 3/7 — the richer verdict
spread the eval needs.

## v3 — 2026-07-11

One-sentence scope fix to the STEP attribution rule, closing the loophole v2's Grayzone case
exposed: attribution treatment applies ONLY to third parties — the posting outlet's own
assertions are the post itself and are judged on substance, never as "the outlet said it".

Also: step_max_tokens 2600 -> 3200 (one 9-claim low-NG post hit the cap).

**Runs**
- main set: runs/dev50_main_v3/ + an identical-config noise-floor re-run
  runs/dev50_main_v3_rerun/. Review doc reports/dev50_verify_review_main_v3_2026-07-11.html.
- low-NG set: runs/dev50_lowng_v3/ (50/50 after the token-cap retry). Review doc
  reports/dev50_verify_review_lowng_v3_2026-07-11.html.

**Results**
- Target fix confirmed: Grayzone/GoFundMe -> 3/MISLEADING/NUDGE in run 1 (the audit's call)…
  but bistable: the noise-floor re-run gives 4/NONE again. The scope line helps; it does not
  pin the case.
- v2's veracity compression largely recovered (33x5 in v3 run 1, like v1) — most of the v2
  5->4 drift was noise + the independence wording, not a stable regime change.
- **MEASURED NOISE FLOOR (same prompts, same config, cached searches): 9/50 verdict changes,
  4/50 nudge flips between two identical v3 runs.** DeepInfra at temperature 0 is not
  run-to-run deterministic; posts sitting at the open/conclude or supported/unsupported
  boundary flip. Any single-run comparison overstates prompt effects by up to this floor.
- Stability of the audited v1 errors across both v3 runs: Breitbart 5/pass (stable, 2r),
  FDRLST 5/pass (stable), infowars-Maduro supported (stable, 2r), townhall-Rubio 5->4 pass
  (stable pass). **zerohedge REGRESSED and is tristable across v2/v3 samples:
  {4/pass, 3/UNSUPPORTED/N, 1/FALSE/N}** — the correct December-2025 ADP report rarely
  enters the pool; this is a retrieval gap (date-scoped search), not a judgment gap. v4
  candidate: period-anchored retrieval (Serper date operators / post-date in query) for
  period-anchored claims.
- Low-NG set (v3): veracity 5:22 / 4:13 / 3:13 / 1:2, 15 nudges (30%), misinfo NONE 35 /
  MISLEADING 11 / FALSE 2 / UNSUPPORTED 2 — same shape as the v2 low-NG run (v2: 14 nudges);
  richer spread, nudges concentrated in Grayzone/FDRLST/MintPress.

**Low-NG audit (10 agents, web-ground-truthed — `docs/dev50_lowng_audit.md`)**
- Nudge precision 10/15 (0.67), recall 10/11 (0.91), accuracy 44/50. The 10 correct nudges
  are real misinformation (ground-truthed); accurate-but-charged posts passed without firing
  on tone.
- Dominant false-nudge cause: commentary/attribution posts entering the ledger as bare
  factual claims (3 of 5 = Federalist op-eds) — substantially an UPSTREAM extraction issue.
- One missed nudge: Grayzone Haiti (author-level echo network — independence is domain-deep
  only). Confirmed code bugs: R-id collision across rounds; string-typed OPEN targets
  disabling the targeting guard; refuted has no targeted-query bar (adjacent-event refutation).

**Open decisions / v4 backlog** — ranked list in `docs/dev50_lowng_audit.md` (code-enforced
closure rules, person-level independence, same-event v2, opinion/attribution handling joint
with extraction, scrape-quality hardening, majority-of-3 STEP at conclude, period-anchored
retrieval for the zerohedge class, veracity-docking policy -> settle against fc-gold).
Roadmap through issues #14-#17/#21: `docs/verify_eval_roadmap.md`.

## v4 — 2026-07-12/13 (in progress)

**Change 0 (2026-07-12) — Exa over-escalation fix.** `_redundant_query`/`_content_tokens`
(verify_text.py) replace the overlap-coefficient `_too_similar` for escalation: only a query
with NO new content token is redundant. On redundancy, ONE forced diversified Serper query
(`REQUERY_SYSTEM` + `_diversify_query`) before the Exa jump.

**Change 1 (2026-07-13) — evidence-grounded closure guards.** The closure rules move from
STEP prompt prose into code (`_apply_step_ledger` + `_qualifying` + `_meets_bar`, pure
functions). Trigger: the corroboration audit (`scripts/step_corroboration_audit.py`, 238
supported/refuted closes over the three v3 runs) — 69% MET the bar, 28% WEAK, 3% FAILED;
diagnosis "information gap, not capability gap" (STEP counts voices fine but sees no
reliability signal, so unrated blogs pass as second sources).

Guards, applied between STEP's JSON and the ledger (every refusal/coercion recorded in a
per-post `guard_events` list; per-round count in `rounds[].guards`):
1. *Type coercion* — `_as_cid` accepts int or numeric string ("3"→3) for claim/target/read
   ids in STEP, READ, TRIAGE, and OPEN outputs (string ids were silently dropped).
2. *unsupported requires targeting* — open→unsupported only if a query targeted the claim
   (`targeted` now updated unconditionally, incl. the final round's targets); untargeted →
   stays open. At the forced conclude an untargeted open claim resolves unsupported (honest
   "never checked"), never refuted.
3. + 4. *Corroboration gate* — supported/refuted requires QUALIFYING evidence: matching
   claim, not republication-flagged, not an opinion piece (exempt for attribution claims —
   an op-ed quoting X does confirm the saying; Daniel 2026-07-13); full-read entries need a
   stance in the closing direction. Bar: ≥1 full-read PRIMARY, or ≥1 full-read secondary
   NG≥90 (`NG_STRONG` — deliberately stricter than rank_hits' NG≥75 read-ordering tier), or
   ≥2 independent reliable voices on distinct domains of which AT LEAST ONE is full-read
   (true author/wire independence is backlog #3). **Snippet amendment (2026-07-13, from the
   scrape-quality audit — NYT/WSJ-class sources are systematically unscrapeable, Jina
   included):** a snippet-only entry from a HIGH-RELIABILITY source (PRIMARY or NG≥90)
   qualifies as a CORROBORATING voice — it can fill the second slot of the 2-voice branch —
   but snippets can never be the basis of a close (no full-read in the closing direction =
   refused), never trigger the one-primary or strong-secondary branches, and are
   direction-less (stance inference from a snippet is unreliable; the full-read anchors the
   direction). Distinct-domain counting collapses a snippet duplicating a read source's
   domain (syndication rule). UNRATED/NG<60 snippets remain non-qualifying entirely. Every
   snippet-corroborated close is recorded in guard_events (`snippet-corroboration`) so the
   next audit can measure the branch. NYT full-text access (Developer API) is PARKED — this
   is the interim, possibly permanent, answer. UNRATED corroborates narratively only; NG<60
   never a basis (still readable as context — read-time floor deliberately NOT applied,
   re-check after next run's audit). Wikipedia: NG-unrated but counts as ONE reliable
   secondary voice, never single-sufficing (Daniel 2026-07-13). Kills the empty-evidence
   attribution close and the zero-contradiction refuted (audit posts 40/50 classes).
5. *Snippet R-ids namespaced per round* — `R{rnd}.{ri}` (bare R2 collided across rounds).

*Conclude override (Daniel 2026-07-13):* if a guard refuses a close and STEP said conclude
on a non-final round, the loop forces ONE diversified Serper query targeting the reopened
claims instead of concluding on unqualified evidence (falls through to conclude if the
requery is empty; if it fires in the last Serper slot it consumes the Exa slot as a Serper
round — budget never grows).

*Metadata surfacing:* every evidence entry now carries `rel`/`ng`/`opinion`
(computed once at entry creation via `_rel_info`; `_source_tier` extracted and shared with
`rank_hits`); the evidence table renders `PRIMARY` / `RELIABLE(NG=n)` / `UNRATED` /
`UNRELIABLE(NG<60)` / `OPINION piece`, and STEP_SYSTEM explains the tags + states the
enforced bar. Run records also persist claim `t` (assertion/attribution).

*Primary-source detection* (credibility.py): `is_primary_source` = government suffix
patterns (`.gov/.mil/.edu/.int/.ac.uk`, generic `.gov.XX`, `.gouv.fr`, `.gc.ca`, `.go.jp`…)
+ curated `PRIMARY_SOURCES` allowlist (~70 domains: IGOs, statistics agencies, central
banks, courts, election bodies, legislatures, journals of record) — built from the
2026-07-13 research-agent briefs. Deliberately excluded: PR wires (fake-release vector),
state-actor sites beyond the TLD patterns, preprints, aggregators, publisher umbrellas.
Caveat accepted: `.gov.XX` also privileges autocracies' official sites (primary for what
the state says/does). Growth loop: mine frequent UNRATED domains from guard_events, verify
offline, promote under version control.

**Verification (pure replay, no API — 2026-07-13, `scripts/replay_closure_gate.py`):** all
238 saved closes replayed through the gate. Audit-FAILED 7/7 refused; MET 164/164 pass;
WEAK 51 → 41 pass under the new rules (NG≥90 single / wikipedia-as-voice / 5 via
high-reliability snippet corroboration) + 26 refused. Total 33/238 (14%) refusals — these
would have forced another round or resolved unsupported. Snippets-alone still never pass;
opinion exclusion decisive in 0 closes on these runs (belt-and-braces). Regression checks to
re-run after the next live run: `scripts/step_corroboration_audit.py` +
`scripts/replay_closure_gate.py`.

**Change 2 (2026-07-13) — scrape-quality wiring** (logic from the scrape-quality session:
`pipeline/search.py` paywall dead-zone + Jina-first DONE there; NEW `pipeline/read_select.py`;
spec `docs/scrape_quality_handoff.md` — the audit found the filters destroyed 51 gate-eligible
sources; primary tier worst at 23.8%). Wired into verify_tweet_claims:
- `numbered_block` replaced by read_select's keyword-anchored version (over the cap: lede
  capped at ¼ budget + keyword-hit sentences with ±2 context, richest first; ORIGINAL
  sentence indices preserved so READ citations resolve; gaps marked `[...]`). Head-first
  truncation was silently cutting a median 56.6% of body on 18.9% of read docs.
- Content-aware junk gate (`is_junk`: junk phrase must ALSO lack ≥3 distinct claim terms) +
  new `off-topic` drop reason (`is_off_topic`: zero claim terms anywhere).
- `kws = claim_keywords(claims, query)` computed once per round, drives both gates + READ
  selection.
- Walk provenance: each doc records `src` = rank/pick/backfill (backfill fired invisibly in
  68/176 audited rounds); `read` = numbered_block stats {mode, kept, total, truncated}. Both
  in the round record's docs.
- Compat: `scripts/scrape_quality_audit.py` unpack sites updated for the 3-tuple.

*Verification (cached replay over the v3 runs, no network):* of 578 previously-READ docs,
5 now drop as off-topic — ALL FIVE produced 0 evidence entries in the saved runs and are
scraper-chrome pages (e.g. justice.gov DOJ-mission boilerplate instead of the Bolton PR):
zero yield lost, 5 wasted READ calls saved. Of 10 old junk-page drops, 6 recovered — all
.gov primaries (congress.gov bills ×2 posts, cdph.ca.gov). All 104 previously head-truncated
read docs now get keyword selection.

**Change 3 (2026-07-13) — v5: PER-CLAIM search budget (Daniel's design).** The post-level
round budget (3 Serper + 1 Exa) is replaced by a per-claim budget: each claim gets up to
`tries_per_claim` (2) TARGETED Serper queries. Mechanics:
- Every Serper query carries explicit targets; issuing it increments `tries` for each open
  target. The ledger rendered to STEP shows `tried n/2` per claim — the budget is visible state.
- An open claim at 2/2 tries is marked **unsupported by code** (guard event `budget-exhausted`)
  and the rotation moves to the remaining under-tried claims. STEP's targeting rule: most
  at-risk under-tried open claim(s) first + closely-related claims when one query serves them;
  a query aimed only at exhausted/closed claims is replaced by a forced diversified query for
  the under-tried set.
- A STEP `conclude` while under-tried open claims remain is overridden (generalizes the v4
  conclude-override) — rotation continues until every open claim has had its tries.
- Then ONE Exa round targets ALL unresolved claims (open + unsupported); STEP may flip a
  budget-unsupported claim if the neural round lands qualifying evidence. Final coercion +
  `coerced_open` unchanged. Redundant-query rewording burns a try, so it now always triggers
  a diversify attempt (per round, not once per post); Exa never fires early on redundancy.
- TRIAGE read cap REMOVED (was 0..5): the prompt states no min/max; `read_target = len(picks)`.
- `evidence_table` render-side dedupe: a snippet superseded by a full read of the same
  domain+claim is dropped from the PROMPT (95% of measured near-dup clutter); guards still
  count over the full evidence list.
Motivation (forensics 2026-07-13, dev50 v3): 6/8 zero-evidence unsupported closes were claims
NEVER targeted by any query on 3-8-claim posts concluding in one round; evidence-per-claim fell
~35% as claim count grew while the round budget stayed flat ("roundup starvation", lowng #8).
Worst case rounds = 2×n_claims+1 (queries multi-target, so typically far fewer).
*Verification:* offline fake-pools logic tests (scratchpad test_v5_rotation.py /
test_v5_override.py): rotation covers neglected claims, budget-exhaustion marks unsupported,
premature conclude overridden, exactly one Exa for all unresolved, termination bounded.
NO live run yet.

**Change 4 (2026-07-13, overnight) — v6: Arm C, "just focus on resolving claims" (Daniel).**
The controller is decomposed; routing and the verdict move to code:
- **QUERY** (absorbs OPEN + STEP's query duty + REQUERY's diversify): one call per Serper round,
  briefed with compact open-claim headers (status, tried n/2, type, GAP note) + prior queries —
  no evidence windows (echo-shaped-query risk). Targets validated ⊆ under-tried opens.
- **RESOLVE** (was STEP): judgment only — re-judge each open claim's DOSSIER, set statuses,
  write a per-claim `gap` note (what evidence is missing) that briefs the next QUERY. No
  action/query/verdict fields. Judging rules unchanged (same-event, attribution, proposition
  fidelity, refutation bar); closure bar text unchanged.
- **Dossier state** (ledger study 2026-07-13): per-claim blocks; code-computed bar check line
  per open claim (dry-run of `_meets_bar` in both directions); direction-aware diet — ALL
  refute-direction entries kept, others by full-read/reliability, cap `dossier_entries_cap`=5;
  snippet superseded by same-domain full read dropped from the prompt; CLOSED claims freeze to
  a one-line `resolution` (recorded at close time from the bar detail) and auto-re-expand when
  contrary-direction evidence lands later (measured freeze risk: 1 revision / 50 posts).
  Guards always count over the FULL evidence list — the diet is prompt-only.
- **Routing is code**: while any open claim is under budget -> QUERY; then ONE Exa for all
  unresolved; then done. No conclude decision exists anymore; conclude-override obsolete.
- **Verdict is code** (`code_verdict`): nudge iff >= `nudge_min_refuted` (1) refuted claims;
  misinfo_type FALSE/CONFLICTING/UNSUPPORTED/NONE from label counts; justification assembled
  from the refuted claims' resolution lines (traceable to domains). Conflicting/unsupported
  concentration thresholds TBD with Daniel — counts ship in every record for offline tuning.
  CONSCIOUS LOSS: misleading-framing detection (post-level by nature) is not covered by claim
  labels — parked per the Arm-C decision; candidate homes: extraction (causal hooks as claims)
  or a later post-pass. The verdict-retry call is gone (nothing LLM to retry).
- Record gains `resolutions` + per-round `gaps`; `veracity` is None (scale retired in v6 —
  review tooling must handle it before the next run).
- **±2 vs ±3 READ windows measured (subagent): ±3 KEPT.** 60% of refute-direction entries lose
  material content at ±2 (Llama judge, 1/30 placebo FP; structural: 75% of entries lose a
  claim-signal sentence); saving only ~292 tok/post mean. A targeted keep-rule doesn't pay
  (keeps 53% of the ring for 47/63 of the value). ctx_window stays 3.
*Verification:* offline fake-pools tests (scratchpad test_v6_loop.py): T1 rotation/starvation
(stubborn QUERY targeting one claim -> rotation covers all, one Exa, all unsupported, no-nudge
UNSUPPORTED verdict), T2 refuted close -> code nudge with bar-detail justification, T3 supported
round-1 clean exit without Exa. All pass. NO live run.

**Change 5 (2026-07-13, overnight) — scrape path: permissive admission, strict counting**
(Daniel's framing; drop census by subagent: 80 drops/100 posts, 25 of them PRIMARY/NG>=90 —
the gates were destroying proportionally more of the BEST tier, while NG<60 barely reaches the
walk because rank_hits buries it):
- `search._is_high_value` now consults `credibility.is_primary_source` — the Jina fallback
  previously missed the entire PRIMARY allowlist + non-US gov suffixes (2 measured deaths:
  europarl.europa.eu, hansard.parliament.uk; 7 primary domains exposed).
- **Cache unpinning**: a cached None/stub for a high-value domain gets ONE Jina retry and is
  overwritten (marked `jina_retried` so a dead URL costs exactly one credit). Without this,
  40/43 measured scrape-failures stay permanently dead in eval reruns and every Jina fix is
  invisible.
- **Snippet fallback**: scrape-failed / too-short / too-thin candidates now enter the evidence
  pool as SNIPPET-ONLY entries (short page text or SERP snippet, `auto_snippet` flag, same
  dedupe/circularity treatment) bound to the query's open targets — scrape failures stop being
  silent; guards still forbid snippet-alone closes. Recovers the vanishing-pick class
  (5/34 failed picks vanished entirely, incl. a PRIMARY ukraineoversight.gov).
- **PRIMARY domains may contribute 2 docs per round** (dup-domain killed 7 primary docs —
  distinct .gov records can bear on different claims; guards count distinct domains anyway).
- Floors KEPT: 4-sentence (1 drop/100 posts), 400-char as the full-READ bar (short real
  articles now survive as snippets instead of dying).
- Leak side unchanged: v4 guards already block UNRATED/UNRELIABLE/opinion/republication from
  COUNTING; residual model-reliant classes (outlet-level commentary, author-level echo
  networks, wire one-voice) stay on the backlog.

**Change 6 (2026-07-14) — output diet + parallel READs** (Daniel's latency question):
- RESOLVE was already changes+opens-only (never the full ledger — statuses are id+enum,
  already pointer-compact; the only free text anywhere in the loop's output is gap notes and
  query strings). Tightened further: an open claim with no new evidence and an unchanged gap
  gets NO row — code keeps its state. Saves ~20-35 tok/claim/round on multi-claim posts.
- **READ calls parallelized** (asyncio.gather; they were SERIAL — with the triage cap removed,
  a 5-doc round cost ~60s of avoidable wall time; the LLM pool semaphore governs concurrency).
  This is the dominant per-round latency win of the v6 series.

**Runs** — none yet (smoke gated on Daniel's go).

**Change 7 (2026-07-15) — v7: audit-driven closure fixes** (basis: 8-agent Opus audit of the
full dev-500 run — refuted 15/23 wrong, conflicting 19/23 artifacts, supported ~12-15%
"laundered specifics"; clog/150726.md):
- **Exact-stance anchor (code, `_meets_bar(entries, exact)`)**: a supported/refuted close now
  requires >=1 FULL-READ entry with the exact stance ("supports"/"refutes") — partials and
  snippets corroborate but never anchor. Kills partial/adjacent-evidence closes in both
  directions.
- **As-of-post-date rule, refute direction (prompt + code assist)**: SAME-EVENT rule extended
  — schedules, standing conditions, report-existence, and trailing-window superlatives are
  judged as of the post date; dossier entry lines now tag "PUBLISHED AFTER THE POST"
  (`_after_post`, ISO-date compare) so RESOLVE can apply it.
- **Quote-absence is never refutation (READ + RESOLVE)**: attribution claims refute only on
  evidence about the SAYING (denial/correction/different record); paraphrase with same
  substance supports; absence is neutral.
- **Load-bearing specifics (READ)**: exact figure/superlative/quoted words/causal-agency
  absent from the article -> at most partially-supports (pairs with the exact-stance anchor).
- **Syndication circularity (code)**: `is_republication` gains doc-domain awareness — yahoo/
  msn/aol/archive.*/ground.news pages naming the origin outlet early are the origin's own
  copy (audit: HuffPost->yahoo and Fox->yahoo self-confirms).
- **Conflicting tightened (RESOLVE)**: requires incompatible versions of the SAME proposition,
  SAME event and period; cross-period bleed / partial confirmation / attributed-debate is not
  conflict.
*Validation:* unit tests on all guards + 44-post live replay (the 22 nudged + all conflicting
+ weak-supported exemplars, caches warm): nudges 22->11; 11/15 audit-wrong refutations fixed
(both TRUE refutations held — OD UK-intel now closes on hansard.parliament.uk PRIMARY);
ChildrensHD vaccine-cancer conflicting->REFUTED+nudge (the audit's hidden dangerous case);
14 conflicting artifacts -> supported per audit rulings. Residuals: MotherJones sarcasm-
inversion + Rabb + MintPress-who-reported-first persist as wrong refutations (proposition-
fidelity, not time/quote classes); nd.edu essay counted PRIMARY (allowlist tightening TBD);
full-set supported->unsupported shift unquantified until a full v7 rerun.

**Change 8 (2026-07-15) — v7.2: snippet on-topic guard + publisher voices** (Daniel flagged
the off-topic-snippet risk; measured first: 5/503 v7 closes DEPENDED on snippet voices, all
junk — dailymotion player chrome, quora — snippets bind to claims by query targeting, never
content):
- `_qualifying`: a snippet voice must share >= `SNIPPET_MIN_OVERLAP` (0.15) of the claim's
  content tokens (`_on_topic`) — kills both unrelated results and scraped chrome.
- `_meets_bar`: voices are PUBLISHERS, not domain strings — `_voice_key` collapses subdomains
  to the registrable base and same-family sister sites (`_PUB_FAMILIES`: Sun/Scottish Sun/
  US Sun, Fox subdomains, NYPost/PageSix, Mail family) to one voice. Fixes the audit's
  "@TheSun closed on thesun.co.uk + thescottishsun.co.uk as 2 independent voices" and the
  7 single-domain "2+ voices" miscounts.
*Validation:* unit tests (sister-domain 2-voice blocked, junk snippet excluded, on-topic
snippet kept, genuine 2-voice passes). Not yet re-run at scale; affects ~5-12 closes of 503.

## Telemetry addendum (2026-07-17, no behavior change)
Per-round record now includes `results`: the FULL ranked hit list from the provider
(`{i, url, domain, date, snippet[:200]}`), so hits triage never picked — and hits never
walked as backfill — leave a trace. Visibility only; never fed to any LLM step (Daniel's
requirement: save everything found, pass nothing extra downstream). Runs before this
(v7-dev500a/b, averitec) do not carry the field; the review doc renders it when present
("all N results returned" expandable under each round's docs line).

## Change 9 — v7.3 (2026-07-17): both-directions-qualify guard
A supported/refuted close is COERCED TO CONFLICTING when the opposite direction also clears
the full closing bar (conflicting's own two-sided bar is met by construction; guard_events
records the coercion). Trigger case: ChildrensHD vaccine-study rerun flip — the flawed
study's own abstract (PRIMARY, and literally stating the claim's figures) closed "supported"
over BBC + FactCheck.org 90+ refutations in the same dossier. Blast radius measured on
dev-500 A+B: 12/1016 supported+refuted closes (1.2%), all in known failure classes
(attribution-axis, time-drift, editorial framing). Those runs are NOT rescored (Daniel:
guard is for later runs only). Cost: a guard-coerced conflicting never nudges — the
ChildrensHD content-axis nudge is lost until the two-axis redesign recovers it properly.
Verified by offline dossier replay (no live rerun). LOOP_VERSION stamp bumped to v7.3.

## Change 10 — v7.4 (2026-07-20): CONTEXT step, self-sourced closes, pair-aware loop
Three Daniel-signed changes for the dev-500-B verify pass (the first run on the v4.7
two-axis extraction):
1. **CONTEXT step** (once, before round 1): resolve the post's t.co link (curl primary —
   t.co 403s Python's TLS fingerprint; urllib fallback), scrape the linked article, one
   LLM call pins dangling referents into each claim (unnamed actors included: "an SNL
   star" -> the name the article gives). The loop then runs ENTIRELY on the resolved
   text; `c_orig` keeps the original in the record. The article is used only here and
   discarded — never evidence: its exact URL joins seen_urls, and a THIRD-PARTY linked
   domain is excluded from retrieval like the origin. Claims the extractor filed
   "unresolved" regain eligibility when resolution succeeds (`recovered_by_link`).
2. **Self-sourced closes** (single-source-of-truth, Daniel 2026-07-20): when the linked
   page on the outlet's OWN domain is itself the venue of the saying/act the claim
   describes (their interview/broadcast/publication act), the claim is trivially true —
   closed "supported — self-sourced" at round 0, no search spent. Code requires
   origin_link; a third-party venue is never self-sourced. Contrary evidence arriving
   later can still reopen it via the frozen-dossier rule. (Media authentication of the
   venue itself remains IDEA-015, end of roadmap.)
3. **Pair-aware loop**: group ids travel through the handoff (build_verify_input
   2026-07-20); dossiers and QUERY's open-claims block carry "paired with claim k";
   READ + RESOLVE get the pair rule — evidence merely REPORTING "X said Y" bears on the
   attribution member only; the bare content member closes only on a source's own voice
   affirming/contradicting Y. QUERY told pair members usually share one search. Record
   gains `pairs` (per-group member statuses — the raw material for attribution-aware
   nudging, policy still postponed) and `context` (the CONTEXT trace).
Smoke: CONTEXT validated live on dev500b posts (axios Daly = referent pinned + self-
sourced close; DailyWire SNL teaser = "An SNL star" -> "Michael Che", unresolved
recovered; video-only post = clean no-link skip). No full verify run yet.

### v7.4 amendment (same day): self-sourced = channel question + attribution-only code gate
The direct boolean design measured 29% precision on an 87-post CONTEXT-only smoke (6/21
correct: the model conflated "our article reports X" with "X happened in our article" —
press events, court rulings, speeches all fired). A venue-free-text redesign was WORSE
(37 firings; "this article" named as venue of the outlet's own reporting). Final design:
the model answers a narrow CHANNEL question for reported sayings only (to-this-outlet /
own-piece-here / elsewhere / unclear / n-a) and CODE additionally requires type ==
attribution (a world event is never self-sourced) + origin domain. Re-adjudicated: 8/9
correct, 0 misses (residual class: statement made on TV to another outlet, quoted by the
linked article — 1 case). Also fixed: _SOCIAL_HOSTS substring matching ('x.com' in
'vox.com', 't.co' in 'nypost.com') skipped real articles as social links.

## Change 11 — v7.5 (2026-07-21): institutional tier no longer single-closes
The tier the code calls PRIMARY (gov TLDs, .edu as of today, curated allowlist) is
renamed conceptually to INSTITUTIONAL and demoted from single-close authority: it now
counts as a strong voice in the two-voice branch only. Basis (Daniel): institutional
domains are not automatically the primary/definitive record for a given claim — a .edu
page or agency press page can be adjacent commentary. Single-close authority will move
to READ-level detection of truly primary/definitive records (roadmap; not yet built).
NG>=90 single-close unchanged. RESOLVE prompt bar text updated to match. Applies to all
future runs incl. the Truth Odds v2 profiles; no completed run rescored.
