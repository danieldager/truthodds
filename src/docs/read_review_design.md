# READ review step — design brief (draft for Daniel, 2026-08-21)

Status: DRAFT — no code. Protocol: agree this doc, then build the offline A/B; no
live run before the four-cell gate passes and weights are refit.

## What it is

One additional LLM call per claim, after the ten per-document reads and before the
urn weights: the REVIEW step. Input = the claim + a compact, code-assembled digest
of all ten reads (cited sentences, flags, dates, source metadata). Output =
per-document ADJUSTED flags plus new bookkeeping labels. The urn then scores the
adjusted flags with refit weights. Per-doc reads stay untouched and unconditioned;
review only re-labels.

Why after, not inside, the reader: three reader-prompt attempts inverted eval
strata and were reverted (read_v5_prompts.py revert note); the claim-rewrite fix
failed the four-cell gate on 2026-08-21 (fixed 3 false flags on TRUE-negation,
cost 11 correct flags on FALSE-negation, n=187/201). Adjudicating after the reads
is the only intervention point left that cannot corrupt the per-doc instrument.

## Failure classes it targets (measured this session)

| rank | failure | weight | reviewable because |
|---|---|---|---|
| 1 | correlated voices / wire echo | 0/25 harsh errors single-doc; 3/25 pure echo; 3/21 generous hoax-echo | echo is only visible across documents |
| 2 | polarity inversion on failure predicates | 9% of E2 claims, 15% of flags | review re-checks each refuting flag against its own cited sentence |
| 3 | stale / superseded evidence (incl. accumulating counts) | 8/25 harsh | date gradient is cross-document |
| 4 | adjacency / scope transfer | ~8 across both sides | second look with the exact proposition in front |
| 5 | hedge stripped | ~4 | explicit rule, applied once, centrally |
| 6 | reader self-contradiction | ~6 | citation and flag are side by side in the digest |

NOT targeted: gold axis mismatch (eval-side), unresolvable referents (screened),
attribution axis (prohibited territory; the review must not re-litigate whether
reporting a statement supports it — that rule stays reader-side and untouched).

## Non-LLM preprocessing (code, before the call)

Everything cheap and deterministic happens before the model sees anything:

1. **Voice grouping**: group docs by existing `voice` key; mark `mirror` docs.
   Additionally compute pairwise idf-trigram similarity of CITED sentences
   (cn_cluster machinery) and tag near-identical citation text across domains as
   `wire_copy` candidates — the syndication case the voice key misses.
2. **Date layout**: docs sorted by publication date; each annotated with
   (doc_date − claim_date) in days and a PRE/POST marker. Undated docs marked.
3. **Metadata line per doc**: domain, NG score / rel tier, fc_domain, provenance
   (scrape vs snippet), qc_flag (blanket-citation), original flag.
4. **Digest = cited sentences only** (plus one-sentence context where the read
   selected connectors), NOT full documents — keeps the context ~1.5-3k tokens and
   reproducible from saved runs.
5. **Numeric assist**: regex-extract numbers+units from cited sentences and list
   them beside the claim's numbers, so magnitude comparison is explicit.

## The review prompt — content commitments

Abstract principles only (standing rule: no dataset-derived examples). The call:

- may CHANGE any document's flag, with a one-word reason code per change;
- may mark a doc `echo-of: <rank>` (its vote will be discounted in code, not
  deleted — discount factor is a fit parameter, not a prompt decision);
- may mark a doc `superseded-by: <rank>` ONLY when all three hold (Daniel
  2026-08-21): same moving quantity or evolving state; the superseding doc is
  dated later; and the two values are consistent with ONE trajectory (85 cases
  then 120 cases = supersession; "no outbreak" vs "120 cases" = CONTRADICTION,
  both flags stand). Stale means overtaken by time, never merely disagreeing;
- must re-check every REFUTING flag against its cited sentence with the question
  "what would this document have to say for the claim to be FALSE — does it say
  that?" (the polarity check, applied at review where it cannot destabilise the
  reader);
- must apply the hedge rule: a flat contradiction of the unrestricted proposition
  does not refute a scoped/hedged claim (downgrade 1->2 or ->X);
- must NOT emit a claim-level verdict, probability, or summary of truth. Output
  is per-document labels only. If the model volunteers a verdict field it is
  discarded by the parser.
- must NOT use its own knowledge of the claim; the digest is the world.

Output schema (JSON): per doc `{rank, flag, change_reason?, echo_of?,
superseded_by?}` — flags in the existing 5/4/3/2/1/X/I vocabulary so fit_urn
loads it unchanged.

## Scoring changes

- Echo handling in code: an `echo-of` doc's voice weight is multiplied by lambda,
  lambda in {0, 0.25, 0.5, 1} swept in the refit — we LEARN how much an echo is
  worth rather than assert it.
- Weights refit on adjusted flags (fit_urn as-is, oof, same folds).
- The unreviewed score remains computed and stored beside the reviewed one —
  additive column, full comparability with E1/E2 history.

## Validation protocol (all offline, saved E1 reads)

0. **Error census (first, ~$1-2)**: one Flash pass over every mis-scored E1
   claim (score-label mismatch, ~1,700-2,000 claims): tag the suspected
   mechanism from the taxonomy OR "gold-problem" (axis mismatch). Motivation:
   the current class weights rest on a complete census of the two EXTREME bands
   only (25 flagged-TRUE + 21 passed-hard-false); the middle mass is unaudited
   (Daniel 2026-08-21: audit ~every wrongly-scored claim). Outputs: true
   per-class prevalence with denominators; the per-claim mechanism column that
   becomes the review A/B's regression casebook; and an exclusion-candidate
   list -> gold-problem claims REMOVED from the eval set (exclusion parquet,
   rows kept) and the urn refit before any review comparison.

1. **A0 ($0, second)**: no-LLM ablation — mirror/wire discount alone, refit,
   AUC/recall vs baseline 0.829/31.3%. If A0 captures most of the gain, the LLM
   call must beat A0, not the baseline.
2. **A1 (~$1.5)**: full review pass over the 4,157 saved E1 claims at Flash;
   refit; report AUC oof, recall@FPR<=2%, LR+.
3. **Four-cell gates (pre-registered)**: no recall loss on gold-FALSE cells, in
   BOTH the negation split and the numeric split. Casebook regression set run
   individually: war-powers, outbreak counts, the 25 harsh + 21 generous cases —
   each must move the right way or stay put; a table in the results names each.
4. **Stability**: 2 reps on a 200-claim subset; the review's marginal effect must
   exceed the measured read noise floor (sd 0.84/claim).
5. **Homogenization check**: distribution of adjusted flags per claim — if
   near-unanimity jumps, the urn is being fed one opinion ten times; report the
   voice-entropy change.
6. Only after 1-5 pass: E2 tweet corpus re-score (saved reads, $1-2), then decide
   whether the LIVE pipeline adopts review (that is a re-run-everything change).

## Open decisions for Daniel

- lambda sweep set for echo discount. Rationale for fitting rather than fixing
  (Daniel 2026-08-21): syndication is an editorial choice, so echo may carry
  real signal -- if so the fit lands lambda~1 and the discount is a no-op
- whether wire_copy tagging needs a similarity threshold probe first
- does review also run at claim-normalization... no: normalization stays a
  separate, single repo-wide prompt (Daniel 2026-08-21); its polarity clause is
  DROPPED after the failed gate; hedge-verbatim + referent naming remain its
  scope. This doc's polarity check lives at review only.
- model for the review call: Flash first; Qwen3-235B-Instruct (~same input
  price) is the one-knob upgrade if Flash under-performs

## Audit synthesis (2026-08-21, final) — 125 claims hand-audited + 751 census-labelled

Four audit rounds: tails (25 flagged-TRUE + 21 passed-hard-false, complete
censuses), mid-mass random (50), severity spectrum (50, incl. the near-miss
recall frontier). Corpus census: 751 mis-scored claims machine-labelled.
Artifact-anchor tag over all 4,035: 286 = 7.1% artifact-anchored (15.1% of
all-silent gold-TRUE).

### The two product risks have DIFFERENT causes

**False flags (FPR side, score <= -4.63 on gold-TRUE): 25 claims, fully
audited.** Severe cases are NEVER one bad read. Every one is a claim property
that makes a correlated doc set disagree systematically -- cumulative count
still rising, negated-identity claim, counterfactual -- times 5-7 near-duplicate
voices. Per-doc reader fixes barely touch them. Mitigation = echo discount +
cumulative-quantity rule + negation handling at REVIEW + gold exclusion of
counterfactual//article/ rows. Mild/moderate TRUE-side errors are idiosyncratic
reader slips that never reach the threshold (41% of a sparse claim's negative
score is silence drag, not the bad flag).

**Missed falses (recall side): 1,138 hard-FALSE in (-4.63,0].** 8/10 audited
near-misses have ZERO directional docs -- recall is lost UPSTREAM of reading:
retrieval drift (query loses the claim's entity/locale; Australian claim -> 10
US .gov pages), fabricated quotes (dense on-date coverage that never contains
the quote -- inexpressible negative evidence), miscaptioned media. Only 2/10
were reader-fixable (refutation sitting in X-flagged docs).

### Cross-cutting measured facts
- Missed-support asymmetry: docs whose cited text plainly SUPPORTS get flagged
  X/I (5/25 TRUE-side); refutations are not symmetrically dropped. Hurts FPR.
- Reader self-contradiction (flag vs own citation): 4/25.
- Support outweighs refute 1.26:1 in the fitted weights -- backwards for the
  product goal; revisit at refit on the abstain convention.
- Silence drag: -0.14 x 8-9 empty slots = -1.1..-1.25 on every sparse claim.

### Revised priority (product goal: min FPR, max FALSE-recall)
1. Abstain convention (all-silent -> CHECK) -- $0, measured AUC 0.825->0.888.
2. Echo/wire discount (review A0, code-only) -- attacks ALL severe FPR cases.
3. Retrieval: entity/locale-preserving query gen -- the recall lever; separate
   workstream from this doc.
4. Review LLM pass: X-relabel both directions, self-citation consistency,
   cumulative/supersession, negation polarity -- per this doc's protocol.
5. Gold exclusions: gold_axis (94 census + spot-check), /article/ URLs,
   counterfactuals, artifact-anchored (286 tagged, 7.1%).
