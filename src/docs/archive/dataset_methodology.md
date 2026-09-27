# Evaluation Dataset — Methodology

*Running methodology document. Maintained as the dataset is built, in a register intended to translate
directly into the methodology/data section of a paper. Purpose: to **justify** how and why the evaluation
dataset is constructed, so every downstream measurement rests on a defensible gold standard. Companion
documents: `enrichment_audit_spec.md` (engineering specification of the procedure), `verdict_eval_plan.md`
(the verdict-evaluation design). Last updated: 2026-06-28.*

---

## 1. Motivation and design objective

This project evaluates a three-stage automated fact-checking pipeline — **filter** (Stage 1: does a post
carry a check-worthy claim in scope?), **extract** (Stage 2: recover the atomic claims), and **verify**
(Stage 3: are they true?) — that issues misinformation warnings ("nudges") to social-media users. The pipeline is the instrument of a field experiment that empirically tests the theoretical model of
Guriev et al.; the credibility of every behavioural estimate downstream depends on first establishing that
the pipeline measures what it claims to measure. The evaluation dataset is therefore not incidental
plumbing but the **measurement standard** against which the instrument is validated, and its construction is
the principal internal-validity concern of the empirical design.

Four threats to validity govern every construction choice: (i) **circularity** — the gold standard must not
be produced by the system being evaluated; (ii) **evidence leakage** — the verifier must not have access, at
evaluation time, to the answer or to information unavailable in deployment; (iii) **label noise** — the gold
labels must be accurate and internally consistent; and (iv) **scope mis-definition** — the population of
posts the system is meant to act on must be delineated precisely, because a fuzzy boundary contaminates the
negative class used to tune the filter stage. The procedures below are organised around neutralising
these four threats.

## 2. Data source

The dataset is harvested from eight professional fact-checking organisations (PolitiFact, Snopes, Lead
Stories, Full Fact, 20 Minutes, AFP, AAP, FactCheck.org), spanning English and French. The collection window
is 24 months for the five web-scraped sources and 12 months for the three API-only sources, yielding a core
pool of approximately 11,800 fact-checks. Each record comprises a normalised **claim**, the publisher's
**rating** (its verdict on that claim), the **claimant**, the publisher's **cited sources**, and — where it
can be recovered — the **original social-media post** that circulated the claim, including its text and any
attached image.

Professional fact-checks are adopted as the gold standard for three reasons. They are produced independently
of our pipeline by trained adjudicators, satisfying the non-circularity requirement at the source. They are
broad in topical and temporal coverage at zero marginal cost. And the verdict is an explicit, citable label
rather than a latent quantity requiring our own adjudication. The principal cost — that fact-checkers select
which claims to examine — is treated as a known sampling bias and discussed under limitations (§9).

## 3. Unit of analysis and the three measurement targets

Each pipeline stage measures a distinct capability and therefore requires its own evaluation subset, derived
from the common core under explicit, documented inclusion rules:

- **Stage 1 (filter)** asks whether a post carries a check-worthy claim within the system's scope. It
  operates on the **post**.
- **Stage 2 (extract)** asks whether the system recovers the post's atomic, checkable claims, together
  with flags for whether the post's *attribution* or its media's *authenticity* must also be verified. It
  operates on the **post**.
- **Stage 3 (verify)** asks whether a claim is true. It operates on the **gold claim**.

The populations differ and this difference is consequential. The verify stage can be evaluated on any record
for which the gold claim and rating survive, even when the original post has been deleted; the filter and
extract stages can only be evaluated on records for which the post itself is recoverable and actually
expresses the claim. Building one undifferentiated dataset would silently mix unmeasurable rows into the
filter and extract tasks; constructing three subsets with explicit inclusion funnels makes the measured
population of
each task transparent.

## 4. Enrichment and audit procedure

The three subsets are not labelled independently. A single image-aware **enrichment-and-audit** pass over the
core produces an objective set of tags from which each subset is then derived deterministically. The pass is
split into two complementary stages, distinguished by what information the labeller is permitted to see —
because the appropriate evidence set differs by purpose.

**Blind post audit.** The first stage labels each post using *only what the deployed system sees*: the post
text and image, with no access to the gold claim or rating. It records the post's topic, whether it lies
within scope, whether it asserts a specific checkable claim, and where that claim resides (text, image,
both, or neither). Blindness is essential, not incidental: only a labeller operating under the system's own
information constraint yields (a) a valid *independent* check of the Stage-1 filter and (b) negative examples
that are uncontaminated by knowledge of the verdict. Supplying the gold claim here would inflate
claim-presence judgements and defeat both purposes.

**Sighted validation audit.** The second stage is given the full record — claim, rating, the rule-assigned
gold verdict, and the post with its image — and *validates the ground truth*. It records which proposition
the fact-checker adjudicated (the axis; §6), whether the rating-to-verdict translation is correct, whether
the recovered post in fact expresses the gold claim, whether the image supports the claim, and whether the
item is satire. This is the dataset-cleaning stage: it is the mechanism by which we gain confidence that each
record is what we expect without inspecting all eleven thousand by hand.

**Independent adjudication.** Both stages use a labelling model that is deliberately *different from and
larger than* the system under evaluation (a 235-billion-parameter vision–language model), so the audit is in
no sense the system grading itself. Each item is labelled a second time by an independent model
(a 27-billion-parameter model); the two are compared field by field, and **disagreement is used as the
operative signal of uncertainty**, routing contested items to human review (§8). This replaces a model's
self-reported confidence, which we found uninformative (effectively constant).

**Derived rather than elicited labels.** The actionable labels — the keep/discard/exclude decision for
filtering, and the borderline flag for review — are *computed* from the objective components rather than
requested from the model. A model asked directly whether to keep a post tends to smuggle in a judgement of
the claim's truth; a deterministic rule over scope, claim-presence and readability does not. This keeps the
filter label a function of observable post properties alone.

## 5. Non-circularity and leakage controls

The gold verdict is the **publisher's own published rating**, mapped to our scale by a frozen rule table; it
is never produced by running our pipeline. The mapping shares a codebook with the verifier but not an
assignment procedure: the harmoniser translates only a rating string and never sees the evidence, the
fact-check article, or our system's verdict, and its language-model fallback (for free-text ratings) is a
different model from the verifier. At evaluation time, leakage is further controlled by a date ceiling
and by excluding the fact-check's own URL from retrieval. The ceiling is set at the **date the claim
circulated** (`claim_date`), not the date the fact-check was published: the evaluation simulates *live*
fact-checking — the system is scored on the evidence it could have gathered the moment a user shared the
post, before the fact-check (or same-claim sibling debunks, which postdate the claim) existed. Where the
claim date is unrecoverable the review date is used as a more lenient fallback, and those rows are tagged
so they can be reported, or excluded, separately. A blanket fact-check-domain block is deliberately not
applied: the date ceiling already withholds the answer, and the deployment-faithful question is how the
verifier performs against the open web of the claim's day.

Image leakage is controlled *by construction* at harvest: the collector ingests only the original post's
media or the source page's preview image, never the fact-check article's own graphics (which would display
the verdict). Finally, post–claim correspondence is audited explicitly (§4, sighted stage): records in which
the recovered post does **not** carry the gold claim — typically a deleted tweet for which only an unrelated
preview image survives — are excluded from the filter and extract subsets, where they would be
unmeasurable, while being retained for verification, which needs only the claim.

## 6. The verdict gold: a five-point veracity scale

The verification target is re-harmonised from the raw publisher rating onto a five-point **veracity** scale
(1 clearly false … 5 clearly true) together with a **subtype** tag, rather than the coarser four-class scheme
used previously. The four-class scheme collapsed "true" and "mostly true" and discarded gradations that the
nudging decision requires. Critically, the middle of the scale is split by subtype into two mechanisms that
share a value but differ in kind — *contested* claims (evidence exists on both sides) versus *unprovable*
claims (no evidence either way) — a distinction on which the construct-validity tests of the system's
confidence dimensions depend. The mapping is rule-based for the curated publishers and numeric-scale-based
for the two publishers that rate on a numeric scale, with a language-model fallback for free-text verdicts;
it covers 96% of records by rule and was validated for coherence against the prior four-class labelling
(no contradictory assignments remained after correction). Media-authenticity and attribution ratings are
tagged separately so that the headline veracity evaluation can be run on factual-content claims alone, the
other axes being different measurement problems.

## 7. Scope definition for filtering

The filter targets a specific quarry: **misinformation that drives political polarization**, in a global
sense — any country, and any political division (left/right, pro- versus anti-government, immigration,
religion or ethnicity as politics). A post is in scope when the **veracity of its claim has
political-polarization implications**: if believed, it would sow division — make one political side or group
look unreasonably bad, a favoured side unreasonably good, or otherwise shift sentiment for or against a group
or those in power. Crucially, scope is decided by **implications, not topic**. A claim is in scope even when
its surface topic is not explicitly political, provided its truth carries political implications (a fabricated
migrant-crime story; a politically weaponised health or science claim; a culture-war claim); and it is out of
scope even when its topic is political or it names political actors, when its veracity carries no polarization
implication (a politician's mundane biography; a neutral government-process fact; a true, uncontested event).
This topic-agnostic, implication-based criterion ties the filter directly to the polarization mechanism the
wider project tests, and it is materially narrower than a generic "societal consequence" scope. The boundary
is developed iteratively against deliberately adversarial, borderline-enriched samples, with residual
disagreements adjudicated by hand and folded back as labelled contrastive examples. Its precision is
load-bearing: out-of-scope fact-checks supply the highest-quality negative examples for the filter, so an
imprecise boundary would systematically corrupt the negative class.

Satire is handled by how *deceptive* it is, not by a blanket rule. Fact-checkers debunk satire when a
satirical item escapes its original context and circulates as believed-real misinformation, so the corpus
contains a non-trivial number of such items. A post that reads as an **obvious** joke or parody is labelled
negative — the tool should not flag it — whereas a satirical claim that is **hard to distinguish from a
genuine one** is treated as a real check-worthy claim, because that deceptive content is precisely what the
filter must catch. This judgement is made blind, from the post alone, which makes it deliberately demanding,
and it is acknowledged to be imperfect. The originated-as-satire flag from the publisher's rating is retained
as metadata (and the verification gold excludes satire by default, its veracity being degenerate).

## 8. Human adjudication and reproducibility

The procedure is human-in-the-loop by design but economises the human's attention. Items on which the two
independent models disagree, or on which derived components conflict, are surfaced for adjudication; the
remainder are accepted with spot-checking. Every label assignment and every human ruling is recorded in an
append-only **provenance ledger**, and each dataset is constructed as a pure function of the raw harvest and
the ledger, so that a rebuild reproduces the published dataset exactly and every label is traceable to its
origin (rule, model, or human) with a recorded reason. Train/validation/test partitions are assigned by a
hash of the record's stable identifier and frozen, so the split is invariant to changes in the pool.

## 9. Limitations

The corpus is English-dominant; French coverage is thin and other languages absent. Fact-checkers select
which claims to examine, so the dataset over-represents claims salient enough to be debunked and is verdict-
skewed toward falsehood (which we counter by down-sampling the dominant false class when forming balanced
validation and test partitions, while retaining the natural distribution in a separate slice). The
adjudicating model is itself imperfect — it can over- or under-apply scope carve-outs — which is why its
output is checked against a second model and, on disagreement, by a human. Finally, the adjudicated-axis tag
is non-exhaustive (a post may admit multiple readings; the tag records the one the fact-checker chose), so it
is used as a recall floor rather than an exhaustive ground truth.
