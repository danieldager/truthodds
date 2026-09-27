## Read-step design menu — grounded in the saved pilot data

I re-derived every number below from `src/eval/data/urn_runs/e1/results-00.jsonl` (420 claims, 4,032 docs, 6,535 region reads, $1.037), `read_ab.jsonl` (184 × A0/A2), and the hand diagnoses at `/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/69db2d82-c54d-4359-9aaf-5ca51736c75f/scratchpad/diagnoses.json`. Four measurements change the picture before the menu starts.

**M1 — The 34% is a stratified-sample number; the real read error rate is 22.8%.** The 184 diagnoses are drawn from 16 (veracity-class × pilot-direction) cells that heavily over-sample directional reads (45% of the sample vs 28.6% of the population). Reweighting each cell's error rate by its true frequency (98.2% population coverage): **22.8%**, not 31.5%/34.2%.

**M2 — Only 10.7% of reads carry an error that can move the urn at all.** Mapping labels to urn sign (+1/−1/0), natural-weighted:

| error type | weight | share of sign error |
|---|---|---|
| false direction (0 → +/−) | **8.6pp** | 80% |
| sign flip (+ ↔ −) | 0.9pp | 8% |
| missed direction (+/− → 0) | 1.1pp | 10% |
| label wrong, sign unchanged (irrelevant↔junk↔neutral) | 12.1pp | **0%** |

Over half the measured "read error" is churn among sign-0 labels and is worth exactly nothing. And **89% of the sign-error mass sits inside the 28.6% of docs where read 1 returned a direction**: P(sign wrong | first read directional) = **32.9%**. Of the 40 wrong directional reads in the diagnosed set, **34 should have been neutral/irrelevant** — the fix action is almost always *downgrade*.

**M3 — 25.0% of reads are stable errors; noise is a 4.1pp side-show.** Pairing the pilot label against the A0 re-run: 21/184 disagree (the 11.4% floor), but on those 21 the oracle pick is 0.714 vs a single draw's 0.357 → **eliminating temp-0 noise perfectly buys at most +4.08pp** read accuracy. Meanwhile 46/184 = **25.0% are identically wrong twice** (mode composition: off-claim-directional 9, missed-evidence 9, scope-mismatch-object 6, mirror-as-support 6). The failure is capability, not sampling.

**M4 — Half the read budget is buying zero AUC, and input quality is not the problem.** Using the pilot reads with a fixed-LR voice scorer (+1.5/support voice, −2.0/refute voice; reproduces the reported 0.781 as 0.787, so the proxy tracks):

| config | region reads | % of pilot spend | AUC |
|---|---|---|---|
| top-10 docs × K≤3 regions (current) | 6,535 | 100% | 0.787 |
| top-5 docs × K≤3 | 3,241 | 50% | 0.791 |
| top-5 docs × 1 region | **2,019** | **31%** | **0.788** |

ΔAUC(top5,1region − top10,all) = **+0.0006, 95% CI [−0.016, +0.019]** (2,000-boot). Ranks 6–10 alone are 50.4% of all region reads for ΔAUC −0.004 [−0.020, +0.011]. Separately, error rate is flat across input quality: scraped 31% vs snippet-only 33%; whole-doc 33% vs windowed 29%; coverage ≥0.7 34% vs coverage <0.3 0/8. Prep artifacts are 3 of 63 diagnosed errors (~1.4pp weighted).

---

### The ranked menu

Effect sizes use the oracle as denominator: perfect reading = +0.083 AUC, and it comes from fixing the 10.7pp of sign error. An intervention removing fraction φ of sign-error mass ≈ +0.083φ AUC (linear approximation, stated as such). Unit costs measured: $0.000158/region read; ~$0.0001 per short (≈860-in/60-out) call; a repeat of an identical prompt is ~40–45% of a first read via the prefix cache.

**1. Trim the read surface to top-5 docs × best-1 region. — DO THIS FIRST.**
- *Mechanism:* `MAX_REGIONS = 1` (take the highest-BM25-mass region, not document-order-first as my simulation conservatively did) and `hits[:5]` in `run_claim`.
- *Failure attacked:* none. This is not an accuracy fix — it is the funding mechanism and the precondition for every other row.
- *Effect:* ΔAUC +0.0006 [−0.016, +0.019] — measured null. Full run drops from 64.7k reads / **$10.2** to 20.0k reads / **$3.16**. Wall: LLM stops being the bottleneck; run becomes Serper-gate-bound at ~70–90 min vs the projected 2.8h.
- *Risk:* the proxy scorer is hand-set LRs, not the fitted Truth-Odds model; and fewer draws changes what "silence" means, so the LRs must be refit on the trimmed urn. Upper CI bound (+0.019) is not zero.
- *Cheapest validation:* $0, already-saved data — rerun the K-sweep above with the **fitted** Truth-Odds LRs and with recall@2%FPR instead of AUC. If the fitted scorer also shows ≤0.02, ship it.

**2. Asymmetric downgrade verifier on directional first-reads only.**
- *Mechanism:* second call only where read 1 returned supports/refutes (34.7% of top-5 docs = **1.7 docs/claim**, 7,055 calls full-run). Input is claim + the ≤8 cited sentences only (measured mean 7.2 sentences / 1,145 chars / ~286 tok), not the region. Output is `{keep | downgrade_to_neutral}` with a reason from a closed set (off_claim, slot_mismatch, mirror, verdict_relay, partial_conjunct). Cannot upgrade.
- *Failure attacked:* the 8.6pp false-direction + 0.9pp flip = 89% of sign-error mass; groups A (41%), C (11%), D (8%) of the diagnosis taxonomy.
- *Cost:* 7,055 × ~$0.0001 = **$0.71** on Flash; ~$3–5 with a 5–10× model. Wall +15–25 min at 12 workers.
- *Effect:* upper bound is φ = 0.89 → +0.074 AUC. **But the saved A2 data says a same-model second opinion does not get there:** every one of A2's 53 label changes on the 82 directional reads was a downgrade — 23 fixed, 20 broke, net **+3 (+3.7%)**. Downgrade *precision* 53% against a base rate of 49% wrong. Asymmetry alone is worth ~nothing; the verifier needs more capacity or more information than the first reader had.
- *Risk:* neutral inflation dilutes the fitted LRs (the F3 risk already named in the read-v5 spec); protect TRUE-sup, which is only 7% wrong today.
- *Cheapest validation:* run the verifier over the 82 already-diagnosed directional pids, cited-sentences-only, ~$0.05 and 5 min. Kill unless fixed − broke ≥ +12/82 (i.e. downgrade precision ≥ 65%).

**3. Buy capacity, but only on the directional subset.**
- *Mechanism:* the ledger's PENDING capacity arm, scoped. Strong/reasoning model as the reader (or as row 2's verifier) for the 17% of reads that are directional-doc reads. Everything else stays on Flash.
- *Failure attacked:* the 25.0% stable-error mass — the only lever that can touch it, since M3 proves it is not sampling and the A2 experiment proves it is not wording.
- *Cost:* 7,055 calls; at 5× input price and 10× output ≈ **$1.5–3**; at reasoning prices ($2.30/M out) budget **$8–12** — over budget even after the trim, so a reasoning model must be scoped tighter than "all directional" (e.g. directional ∧ scope=adjacent). Wall +60–90 min for a reasoning model at 12 workers.
- *Effect:* unknown until measured; this is exactly why it should be measured before anything is built. A useful prior: 3 samples of Flash (2× read-v4 + 1× A2) get at-least-one-correct on 84.2%, all-wrong on **15.8%** — that 15.8% is Flash's hard floor.
- *Risk:* cost-consult rule applies; verify $/M before launch (ledger row is explicitly gated on this).
- *Cheapest validation:* `read_ab.py --model <strong> --pids <82 directional>` — ~$0.10–0.50, ~15 min, paired against A0 on identical inputs. This is the highest-information dollar on the whole list.

**4. Extract the claim's slots once per claim, inject into every read.**
- *Mechanism:* 1 call per claim producing `{who, what, where, when, how_much, conjuncts[]}`; append to `claim_block`, which is already the `prompt_cache_key`-ed constant prefix — so the marginal per-read cost is the cached-token rate, ~0.
- *Failure attacked:* groups A + D (31 of 63 diagnosed errors, 49% by count). Genuinely different from the A2 arm: A2 made each read re-derive slots *while also judging*, so the same claim got a different decomposition in each of its 5–10 reads. Fixing them once removes that variance and lets the decomposition be longer than a read prompt can afford.
- *Cost:* 4,156 × ~$0.0001 = **$0.42**, wall +12 min.
- *Effect:* honest estimate +2 to +4pp read accuracy → φ ≈ 0.1–0.2 → **+0.008 to +0.017 AUC**. Low confidence; the A2 null is a warning.
- *Risk:* a bad slot extraction poisons all 5–10 reads of that claim instead of one — variance goes down, but correlated error goes up.
- *Cheapest validation:* offline on all 184 diagnosed pids, two-stage, ~$0.12 / 10 min, paired McNemar vs A0.

**5. Self-consistency k=3, directional reads only.**
- *Mechanism:* re-sample the directional reads; majority vote. Repeat samples are prefix-cache-priced.
- *Failure attacked:* the 11.4% noise floor.
- *Cost:* 14.1k extra calls × ~$0.00007 = **$1.0**, wall +40 min.
- *Effect:* hard ceiling **+4.08pp** read accuracy even with a perfect mode-picker, and majority-of-3 captures maybe half. My direct simulation with the three samples on hand gives 0.707 vs 0.685 single (+2.2pp, n=184, n.s.). In sign terms φ ≈ 0.1 → **+0.008 AUC**. It is cheap and it is real, but it is 1/10 of the oracle gap.
- *Risk:* none beyond cost; do not do it on all reads (that's 3× on a 22.8% problem whose noise component is 4pp).
- *Cheapest validation:* already done — the numbers above are from saved data. One more A0 replicate on the 82 directional pids ($0.05) would give a true 3-of-3 majority instead of my 2-prompt proxy.

**6. Split scope and direction into two separate calls.** Redundant with row 2, which *is* a scope check run second, on a triaged 17% of reads, over the cited text only. Do not pay for a universal second call. But **keep A2's `scope` field as a routing feature**: P(read wrong | scope=adjacent) = **0.49** vs 0.28 exact / 0.21 off_claim / 0.21 unreadable. Triage on (directional OR adjacent) covers 54% of the diagnosed sample and 78% of its errors — that is the right trigger set if a reasoning model is too expensive for all directional reads.

**7. Improve the input (boilerplate, titles, truncation marks, better regions).** Measured dead — see M4. Error rate is statistically flat across provenance, windowing and coverage, and prep artifacts are 6% of errors ≈ 1.4pp weighted, ceiling ~+0.01 AUC. Keep the page title because it is free; invest nothing else. Also measured dead: a **code-side lexical mirror detector** — on the diagnosed supports, mirror cases have novel-token fraction 0.78 vs 0.86 for correct supports (n=7). No separation. Mirror detection must stay in the LLM (row 2's `mirror` reason code).

**8. Spend on more evidence rounds instead.** Measured dead, and the most expensive way to be wrong: ranks 6–10 are 50.4% of the read budget for ΔAUC −0.004 [−0.020, +0.011]. Reading 10 instead of 5 reduces zero-directional claims from 160 to 147 of 420 and buys nothing. 35% of claims have **zero** directional docs and that is where the AUC is actually lost — but the recall audit (3.9%) says the evidence is not being discarded, so this is a *retrieval-distribution* problem, not a volume problem. The only version of this worth funding is a **different query distribution** — the refutation-seeking second query in `docs/verify_adversarial_design.md` / IDEA-016 — where silence carries a different sign. That is a separate experiment, not a read-step upgrade.

---

### What I would do first, and what I would not do

**First: row 1, the trim.** It is a two-constant change in `evidence_urn_run.py`, it is measured-null on the outcome metric, it takes the full run from $10.2 to $3.16 and from 2.8h to ~1.5h, and it converts "budget is tight, under 10 euros" into ~$7 of headroom. Every other row on this menu is currently gated on money the trim frees. It also has to go first for a methodological reason: all subsequent A/Bs should be paired against the trimmed configuration, not against a configuration we already know we are abandoning.

**Immediately second (same day, $0.15, 20 min): the capacity probe on the 82 diagnosed directional pids** — `read_ab.py --model <strong> --pids`. It is the only measurement that can distinguish "this step is under-powered" from "this step is at its model's ceiling", and rows 2 and 3 both hinge on the answer.

**What I would NOT do:** (a) any further prompt rewriting of read-v4 — two independent measurements now say wording is not the lever (A2 p=0.69 overall; on the directional subset A2's downgrade precision is 53% against a 49% base rate); (b) self-consistency on *all* reads — 3× cost against a 4pp ceiling; (c) input/prep engineering — flat error rate across every input-quality cut I could make; (d) a second retrieval round with the same query; (e) a code-side mirror heuristic; (f) any intervention evaluated on the unweighted 34%, which over-states the addressable error by 1.5× and hides that half of it is sign-neutral churn.

---

### What it would take to be confident this step is maxed out

Three measurements, none expensive, in this order.

1. **Establish the ground truth's own error rate.** The 184 diagnoses are a single pass; we do not know how often the diagnosis itself is wrong, so we cannot distinguish "model failure" from "label noise" in the residual. Take a stratified 60 of the 184 — over-weighting the 29 reads that all three of our samples got wrong — and diagnose them independently (Daniel by hand, or a different model with the diagnoses hidden). Report agreement. Cost: ~90 min human or ~$1. **Until this exists, every accuracy number on this page has an unknown ceiling below 100%.**

2. **Establish the model-capacity ceiling.** Strongest affordable model, same 184 inputs, paired. Its accuracy is the empirical answer to "can a better reader read these?" If it lands ≤75%, the remaining loss is irreducible on this input and the step is done.

3. **Establish the orchestration ceiling — already partly measured.** Over 3 samples of Flash (2 prompts, 3 draws), at-least-one-correct is **84.2%** and all-wrong is **15.8%**. So **no amount of sampling, voting, verifying or routing over this model can exceed ~84% read accuracy**, versus 68.5% today on the stratified sample. Propagating that to the outcome metric: 84.2% captures (0.842 − 0.685)/(1.000 − 0.685) = **50% of the gap to the oracle**, i.e. an AUC ceiling of roughly **0.78 → 0.82** against the oracle's 0.864.

That is where the evidence puts the ceiling. Orchestrating DeepSeek-V4-Flash better — verification passes, voting, slot injection, routing — is worth at most about **+0.04 AUC** and realistically +0.02 to +0.03, for roughly $2–4 and a day of work, all of it now affordable out of the trim. Getting past AUC ≈ 0.82 requires a **different reader on the directional subset**, and the 82-pid capacity probe is the $0.15 experiment that tells you whether one exists.