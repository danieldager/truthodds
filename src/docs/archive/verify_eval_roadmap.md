# Verify-loop evaluation roadmap — 2026-07-11

Where the loop stands after v1-v3 and both deep audits, and the path through the bosses'
asks (issues #15, #16, #17, #14, #21). Companion docs: `verify_loop_versions.md` (iteration
ledger), `dev50_run_audit.md` (v1 main-set audit), `dev50_lowng_audit.md` (v3 low-NG audit).

## Where we stand

- **Main set (mixed NG, 50 posts)**: verdict agreement 41/50; the v1 false flags are fixed
  and stable. **Low-NG set (NG<40, 50 posts)**: nudge precision 0.67, recall 0.91, accuracy
  0.88 — 10 correct nudges on real misinformation, ground-truthed.
- **Deliverable preview (issue #15)**: on just these 100 posts, per-outlet mean veracity vs
  NewsGuard = Spearman **0.637**, nudge-rate **-0.545** (outlets with n>=3). The instrument
  points the right way before any calibration.
- **Known residuals**: (a) the dominant false-nudge source is upstream — commentary/attribution
  posts entering the ledger as bare factual claims (3 of 5 low-NG false nudges are Federalist
  op-eds; this INFLATES the #15 correlation for the wrong reason); (b) independence is
  domain-deep only (author-level echo networks pass; one missed Grayzone nudge); (c) prompt
  rules are advisory — STEP overrides its own guards in prose under pressure; (d) measured
  noise floor: identical runs differ on ~9/50 verdicts, ~4/50 nudges (DeepInfra temp-0
  nondeterminism).

## Workstreams

**WS0 — loop v4 (gate for everything below).** The audit backlog, in three buckets:
code-enforced closure rules (typed claim ids; never-targeted claims cannot close; the bar
extends to `refuted`; R-id namespacing), independence depth (bylines in READ, media-group
origin exclusion, aggregator/junk hardening, date-aware backfill), same-event v2 (later-event
+ trend-claim + cited-artifact wording). Plus the opinion/attribution fix, which is JOINT
with extraction (claim typing must preserve "X claims Y" and mark commentary). Re-run both
50-post sets (cached, ~free), diff, audit the diff. ~1-2 days.

**WS1 — gold anchor: fc-gold + AVeriTeC (issues #16, #21).** One adapter serves both: wrap a
claim as a single-claim post, run the loop, read the claim's LEDGER status as the 4-class
verdict — our statuses map 1:1 to {Supported, Refuted, NEI, Conflicting}; **no schema change
needed**. Assets exist: `factcheck_textonly.parquet` (4,260 claims, harmonised 4-class +
veracity gold) with the clean 400-claim dev split (old verifier baseline: flag-acc 0.915 /
recall 0.94 / precision 0.96 — direct old-vs-new A/B), and the AVeriTeC dev sample
(`claims_n100.parquet`, full ~500 loadable; ClaimCheck reference 76.4%). Two caveats:
`date_ceiling` on search (eval-only, supported by pools) to stop future-evidence leakage, and
report AVeriTeC accuracy under BOTH NEI mappings (ours vs ClaimCheck's no-evidence->Refuted)
for comparability. Frugal ramp: n=25 smoke -> n=100 -> full. This is also where the
veracity-docking and 3-vs-4 boundary get CALIBRATED against gold instead of audited by hand.
~1 day of adapter+scoring, then run time.

**WS2 — dev-500 -> issue #15 (the headline deliverable).** Run the full dev subset (283
checkworthy posts; 217 no-claim/no-checkworthy posts join as pass/NONE denominators), compute
per-outlet mean veracity and nudge rate vs NewsGuard. Run AFTER WS0 lands, else the
opinion-post bias contaminates outlet rates. Non-circularity (the bosses' note): the posting
outlet is excluded from its own evidence (argue in the writeup), NG is used only to rank
EVIDENCE sources — and we make that clean with an ablation arm: re-run a ~100-post subset with
NG-tier ranking off, show the correlation stands. Variance: two runs of the full set (or
majority-of-3 STEP at conclude, decided in WS0) so outlet rates carry error bars.
Cost: ~$1-2 + ~600-900 Serper + ~50 Exa per arm; ~2h wall. Then the 5k test set (~$8-10, ~7h)
once dev-500 numbers look sane.

**WS3 — regression suite (issue #21).** The two 50-post sets + every audit-disagreement post
become the pinned per-version regression set (cached searches make re-runs ~free); the
magnitude edge cases from #21 get seeded from our audit examples (post-50 metric substitution,
post-40 adjacent-event refutation, the ADP wrong-month case). `diff_verify_runs.py` is the
harness. Continuous, no extra build.

**WS4 — experiment plumbing (issues #14, #17).** #14 is a trivial exporter off any run
(source article, source, claim, veracity CSV). #17 (professional fact-checker subset): design
the sample so it doubles as OUR gold — stratify by our verdict (include disagreement-prone
strata: nudged posts, unsupported closes, low-NG passes) so the human labels also score the
pipeline where it is least certain.

## Suggested order (deadline: end of July)

1. **v4** (WS0) — this week. Includes the extraction-side opinion/attribution coordination.
2. **fc-gold n=25 smoke + AVeriTeC n=25 smoke** (WS1) — same week; the adapter is small and
   the numbers immediately tell us whether verdict calibration needs work before dev-500.
3. **dev-500 two-arm run + #15 analysis** (WS2) — next; the writeup with correlation, error
   bars, ablation.
4. **AVeriTeC full dev + fc-gold 400** (WS1) — parallel to 3 (different result pools).
5. **5k test run** (WS2) — after dev-500 review.
6. #14 CSV + #17 sample design — when the experiment side asks.

Decisions needed from Daniel/bosses: (a) noise mitigation choice (majority-of-3 vs two-run
averaging — recommend majority-of-3 at conclude, ~+25% LLM cost, kills the boundary flapping);
(b) whether opinion posts should be EXCLUDED from per-outlet misinfo rates or verdicted on
their factual predicates only (recommend the latter, it preserves sample size); (c) Exa
budget sign-off for the 5k (projected ~1.5-2k calls vs 1k free tier — a few dollars).
