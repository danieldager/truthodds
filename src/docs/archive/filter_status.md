# Stage-1 FILTER — status & handoff (PAUSED 2026-06-28)

**What this is:** the resume doc for the **Stage-1 filter** workstream (which posts carry a check-worthy
claim worth flagging). **Supersedes** the old `claim_selection_status.md` (Session-1 "gate" work) — the scope
was reframed and the term is now **filter**. Companion docs: `dataset_methodology.md` (paper-facing
justification), `enrichment_audit_spec.md` (the 2-audit/3-dataset engineering spec). Narrative: `clog/280626`.

> **⏸ PAUSED — pivoted to Stage 3 (verify).** The filter is the project's **most subjective** stage, and we
> hit a real validity wall (below): we can't calibrate filter quality without a **human inter-annotator
> ceiling**, which we don't yet have. The infrastructure is built and sound; what's missing is the
> human-validation layer. Resume when ready to make filter-quality claims.

---

## 1. The TARGET (the core definition — locked with Daniel)

The filter targets **misinformation that drives POLITICAL POLARIZATION**, in a **global** sense (any country;
any political division — left/right, pro/anti-government, immigration, religion/ethnicity-as-politics). A post
is **IN** iff the **veracity of its claim has political-polarization implications**: if believed, it sows
division — makes a political side/group look unreasonably bad, a favoured side unreasonably good, or shifts
sentiment for/against a group or those in power. **Decided by IMPLICATIONS, not topic:** IN even if the topic
isn't political (a fabricated migrant-crime story); OUT even if it names political actors but the veracity has
no polarization implication (a politician's mundane biography, a Kamala/Walz selfie). **Satire by
deceptiveness:** an *obvious* joke/parody → OUT; a *deceptive* satire that reads as genuine → IN.

This **supersedes** the earlier "societal consequence" + public-figure scope (materially narrower; ties the
filter to the polarization mechanism the project tests). Full rubric: `enrichment_audit_spec.md §2`;
paper prose: `dataset_methodology.md §7`.

---

## 2. What's BUILT (all working)

- **The audit** (`eval/audit.py`, `eval/scripts/run_enrichment_audit.py`): blind **Pass A** (post →
  `political_implication, has_claim, claim_locus, obvious_joke, readable`) + sighted **Pass B** (gold/verdict
  validation). Labeller = **Qwen3-VL-235B**, cross-check = **Gemma-3-27B**; disagreement = the borderline
  signal. Resumable, version-guarded (`--frac` dev / `--full` resumes; `--reset` on prompt change).
  Dev audited at **`--frac 0.12`** (1,404 rows / 508 has-input), audit_version `ca0a168cc6` (criteria-only).
- **The gold** (`eval/scripts/build_filter_dev.py` → `eval/data/filter_dev.parquet`, 508 rows):
  **317 positive / 131 negative / 60 drop**. Label = `positive ⟺ political_implication ∧ has_claim ∧
  ¬obvious_joke ∧ readable ∧ included`. **Inclusion (1)∨(2):** a row is in only if (1) raw post recoverable
  (`raw_tier∈{post,quote}`) OR (2) claim in image (`claim_locus∈{image,both}`); else drop. **(3) synthetic
  from fc-text — DEFERRED** (only build if volume is material + the distinctness gate validates).
- **Human review** (`eval/scripts/build_filter_review.py` → `filter_review.html`; `apply_filter_review.py`
  → `filter_provenance.parquet`): Daniel reviewed the **99 decisive borderlines** (label flips); applied as a
  `source=human` provenance ledger → reproducible. (Across the 99 he sided with 235B 42× / Gemma 44× / neither
  13× → essentially even → keep 235B as the labeller.)
- **The decoupled filter + harness** (`eval/filter.py` `FILTER_SYSTEM` + `eval/scripts/filter_eval.py`):
  the production filter is a SEPARATE prompt + small model, tested against the FIXED gold. `--model` swaps
  Qwen-30B / Gemma-26B; predictions cached.
- **Text hygiene:** `eval/textnorm.py` + `clean_harvest_text.py` (decoded ~3,300 HTML-entity cells).

---

## 3. ⚠️ THE VALIDITY ISSUE (why we paused) + the resolution path

**Decoupling the filter prompt from the audit/gold prompt was essential** — with them shared, the eval just
measured "30B replicates 235B on the same prompt" (an inflated 88%, a replication artifact). Decoupled, the
real baseline is **75% acc / F1 0.84, and the filter massively OVER-FLAGS** (96 FP; rejects obvious_joke 12%,
not_political 34%). The gross failure modes are real (flagging obvious jokes is wrong by any standard).

**BUT we cannot calibrate filter QUALITY from 75%**, because: (a) the construct is **subjective** (what's
"politically polarizing" is a judgment call), and (b) the gold is **LLM-built** (235B) with only **one human**
(Daniel) reviewing only the borderlines. For a subjective task the ceiling is **human↔human agreement**, which
**we never measured** → 75% is uninterpretable (could be poor, could be at the human ceiling).

**Accepted procedure (computational-social-science annotation + LLM-as-annotator lit, e.g. Gilardi 2023,
Zheng 2023):** 1) codebook ✅; 2) **≥2 human annotators** label a *representative random sample* (~100–200,
not just borderlines) → **inter-annotator agreement κ / Krippendorff's α = the ceiling** ❌; 3) **validate the
LLM annotator** vs the human gold (if LLM↔human ≈ human↔human, the bulk LLM labels are trustworthy) ❌;
4) bulk-label + adjudicate ✅; 5) report the model-under-test **relative to the ceiling** ✅ harness-ready.
**Missing = steps 2–3: a small multi-annotator human-IAA study.**

---

## 4. NEXT STEPS (to resume the filter)

1. **Methodology `/deep-research`** — accepted practice for LLM-assisted annotation of subjective constructs +
   evaluating models against an LLM-built gold + IAA/ceiling reporting. Grounds all 3 stages; cite in the paper.
2. **Human-IAA study (the gap):** Daniel + 1–2 lab colleagues independently label a representative random
   sample (~150) from the codebook → κ/α (the ceiling) + validate the 235B gold against it. *Then* 75% is
   interpretable.
3. **Only then tune `FILTER_SYSTEM`** for precision (kill the over-flagging — reject obvious jokes /
   non-political), re-run `filter_eval` vs the fixed gold, and the **Gemma-26B vs Qwen-30B** model bake-off.
4. Deferred: **(3) synthetic-post inclusion** (measure volume first); extend the audit `--full` (resumes on the
   dev; same version) once the prompt is final.

---

## 5. Decisions LOCKED (don't re-litigate)
- Target = political-polarization misinfo, global, **implications-not-topic** (§1).
- Satire by deceptiveness (obvious_joke → negative; deceptive → positive); gold `is_satire` = metadata only.
- Inclusion = (1) raw post `∈{post,quote}` ∨ (2) claim in image; (3) deferred. Sighted `claim_matches_post`
  does NOT gate the filter (it's a Stage-2/3 cleanliness signal).
- `recommend` DERIVED from objective fields (not asked); borderline = 235B↔Gemma disagreement.
- 235B = gold labeller (independent of the ~30B production model); Gemma-3-27B = cross-check.
- **Filter prompt DECOUPLED** from the audit/gold prompt — the gold is a fixed, independent ground truth.

## 6. File map
- `eval/audit.py` (PASS_A/PASS_B + pass_a/pass_b), `eval/scripts/run_enrichment_audit.py` (the 2-pass runner).
- `eval/scripts/build_filter_dev.py` → `filter_dev.parquet`; `build_filter_review.py` → `filter_review.html`
  + `filter_review_items.parquet`; `apply_filter_review.py` → `filter_provenance.parquet`.
- `eval/filter.py` (FILTER_SYSTEM, the system-under-test), `eval/scripts/filter_eval.py` (the harness) →
  `filter_eval_<model>.parquet`.
- `eval/harmonize.py` (veracity map — shared with Stage 3), `eval/textnorm.py` + `clean_harvest_text.py`.
- `scratchpad/`: `fewshot_validate.py` (the ceiling experiment), `fcpage_inspect.py`, `instress.py`.

## 7. Open questions
- The human-IAA ceiling (the blocker for filter-quality claims).
- Does deceptive-satire detection need a dedicated signal (satire-domain list / parody-account)?
- (3) synthetic-post: worth building? (measure volume + distinctness-gate validation first.)
- Filter output: direct `flag` (current, over-flags) vs fields→derive (stricter, like the gold) — a tuning choice.
