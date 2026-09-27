# Claim Selection (Stage-1 Gate) — status & handoff

> **ARCHIVED 2026-07-28 — code moved, work still resumable.** The whole claim-selection
> toolchain now lives in `src/_archive/claim_selection_gate/` (gate_eval.py with the v8
> prompt, filter/stage builders, the split builders, the review tooling). It was archived
> because the current program is Truth Odds and this line has been paused since
> 2026-06-27 — NOT because it is dead. Everything below still applies; prefix the script
> paths with `_archive/claim_selection_gate/`. `git log --follow` has the full history.


**What this is:** the resume doc for the "claim selection" workstream — the **Stage-1 check-worthiness /
scope gate** that decides, for each social-media post, whether it carries a claim worth fact-checking
(passes to verification) or should be ignored. Cheap multimodal VLM, runs on **all** posts, front of the
3-stage cascade (`docs/core_pipeline_spec.md`). Detailed narrative: `clog/260626.md`, `clog/270626.md`.

> **⏸ PAUSED (2026-06-27).** Two things reshaped the plan: (1) a full audit found the 12mo eval labels are
> contaminated (~198 ambiguous cases sit in an HTML review tool; **nothing applied yet**); (2) **a parallel
> session is extending the harvest to 24 months and the real selection eval set will be rebuilt on that** —
> so the 12mo cleanup/HTML is interim. What carries forward: the v8/**v8g** prompt, all the harness/scripts,
> and the model-choice insight (§1b). To resume: see "▶ HOW TO RESUME" at the bottom.

---

## 1. Where the PROMPT landed (v8)

Current gate prompt = **v8**, in `eval/scripts/gate_eval.py:GATE_SYSTEM`. Architecture (SOTA-informed,
see `clog/270626.md`): **criteria first, examples after, output last**; one holistic decision but the
reasoning + criteria are ordered **SCOPE → RISK → VERDICT** (`check_worthy = in_scope AND specific_claim`);
8 balanced/alternating contrastive examples that cite *which* test decides; carve-outs stated once.

Prompt-iteration history (full detail in clogs): v1 baseline → v6 (sharpened OUT-list + image-claim
clause + concise reasoning) → v8 (scope-first restructure). **Key learnings:**
- **Reasoning length is a spec/recall lever:** verbose reasoning over-runs `max_tokens` and truncates JSON
  (a recall-killing bug we hit and fixed with concise ≤50-word reasoning + retry-on-empty-parse +
  `MAX_TOKENS=600`); too-short (25-word) reasoning starves the public_interest check. 50 words is the balance.
- **"Topic beats specificity"** is the core rule — the gate's failure mode is passing out-of-scope content
  because it's a *specific* fact (a celebrity's marriage, a sports score). v8's scope-first framing **halved
  the out-of-scope leak (85%→38% on `oos_factcheck`)**.
- **Cost:** 30B is **not** prompt-cached on DeepInfra (verified) → ~**$0.36 per 1,000 posts** (image tokens
  dominate, ~891/post; prompt ~1,070 tok). The 235B *is* cached but 2.5× variable cost — not worth switching.

**⚠️ v8's dev numbers are NOT trustworthy yet** — they were measured on the contaminated dataset (below).
On dev: spec 89.6%, text recall 91.3%, image-only recall 66.7% — but ~17 of 22 text "recall drops" and
34 of 48 image-only "drops" were v8 *correctly* rejecting out-of-scope fact-checks. **Re-measure after the
data is clean before trusting anything.**

---

## 1b. Model bake-off + the v8g prompt fix (2026-06-27)

Quick VLM bake-off (`eval/scripts/model_bakeoff.py` → `eval/data/bakeoff_results.parquet`): same v8 prompt,
3 models, on the **clean** confident-label subset (250 pos + 250 neg, drops everything under review). All
numbers are on the *easy* subset → absolute values inflated; the **relative** comparison is the point
(±1–2pt non-determinism). **Architectural fact confirmed (Daniel): the gate passes images upstream, so
image-authenticity (`artifact`-axis) recall genuinely matters.**

Recall **by `judged_axis`** (this is the key lens — always report by axis; 33% of positives are `artifact`):

| model · prompt | spec | content | attrib | artifact | text(c+a) | $/1k | p50 lat |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B · v8 | 97.2% | 99% | 100% | 95% | 99% | 0.283 | **2.3s** |
| Gemma-4-26B-A4B · v8 | 99.2% | 98% | 98% | **83%** | 98% | 0.114 | 3.4s |
| **Gemma-4-26B-A4B · v8g** | **98.8%** | 99% | 100% | **95%** | 99% | **0.126** | 3.3s |
| ByteDance Seed-2.0-mini · v8 | 97.2% | 99% | 100% | 96% | 99% | 0.446 | 6.5s |

**Findings:**
- **Seed-2.0-mini: dropped** — slowest (6.5s / 13s p95) AND most expensive ($0.446/1k, it burns ~1,918 tok/call) with no accuracy edge.
- **The "Gemma −5pt recall" headline was entirely the `artifact` axis.** On text-claim check-worthiness
  (content+attribution) Gemma already tied Qwen (98% vs 99%). The gap was image-authenticity cases (AI/edited/
  miscaptioned images): Qwen flags them (image-as-claim), Gemma-v8 dismissed them as "personal/opinion/insult."
- **v8g closed the gap.** Daniel's fix: redefine **"personal" = a private individual's OWN private life only**
  (claims about PUBLIC figures' lives/conduct/images are IN; a private person's story with political/societal
  implications is IN) + a stronger **image-as-claim override**. Gemma-v8g: artifact **83%→95%**, spec held
  (98.8%). The 3 edits are in `eval/scripts/gemma_prompt_test.py:EDITS` — **model-agnostic scope improvements,
  promote to v9.**
- **Net:** Gemma-v8g now **ties Qwen on every recall axis, beats it on specificity, ~2.2× cheaper**; Qwen's
  only edge is **latency** (2.3s vs 3.3s). Model choice is **DEFERRED** to a rigorous re-test on the clean
  24mo dataset (Daniel: decide once the data is as good as we can get it).

---

## 2. The DATASET and every change made to it

**Built by `eval/scripts/build_gate_splits.py`** (single source of truth; deterministic; idempotent).
Original inputs: `synthetic_negatives.parquet` (547: 180 LLM + 96 hand-crafted + 271 from Daniel's X feed)
and the per-source `*_harvest.parquet` fact-check positives (12-month review window, 2025-06→2026-06;
claims span 2014–2026 incl. resurfaced content).

**Changes applied so far** (logged in `eval/data/gate_dataset_provenance.{md,parquet}` via
`gate_dataset_provenance.py`):
| change | n | why |
|---|---|---|
| negative→positive (recovered mislabels) | 7 | genuinely check-worthy (corruption/policy) — FP audit |
| negative→dropped (video) | 12 | claim in video → measured by the video-drop metric |
| positive→negative (out-of-scope, 1st scan) | 26 | `triage_positives_scope.py` conservative pass |
| positive ADDED (image-only) | 286 | pure-visual posts, gateable via a `uid` key |
| **split made STABLE** | — | `split_5050` now folds by `md5(key)%2` (population-independent). The old rank-parity reshuffled dev/test on every rebuild — **fixed**, but means folds changed once more vs v6 era |

**THE BIG AUDIT (done, but relabels NOT applied — pending HTML review).**
`eval/scripts/audit_dataset.py` ran an independent multimodal judge (**Qwen3-VL-235B**, 4× the gate, sees
images, cached) over **all 1,466 items** → `eval/data/dataset_audit.parquet`. Objective rubric per post:
topic / in_scope / has_claim / readable. **Recommendation is DERIVED from the objective components**, not
the model's `recommend` field (which is unreliable — sneaks in truth judgments; `confidence` was useless,
always "high"; `readable` never false).

Findings:
- **Positives (923): ~217 are out-of-scope/no-claim.** 185 "safe" out-of-scope (celebrity/personal/sports/
  business/product — tabloid debunks, 84% from Lead Stories + Snopes; **Daniel confirmed → make negatives**);
  32 "risky" (politics/carve-out judgment calls → HTML).
- **Negatives (543): 87 flagged check-worthy — but MOST are auditor ERRORS** (unfalsifiable rhetoric like
  "the system is broken" wrongly marked `has_claim=true`). → HTML to confirm the few real ones.
- **79 image-only positives had no checkable claim** — out-of-context photos whose tweet text never resolved
  (deleted/media-only). **RECOVERED** via `eval/scripts/recover_image_captions.py`: the fact-checker's
  `claim_text` scrubbed of framing + verdict-leak words ("authentically shows…", "is authentic", FR forms),
  re-audited with the caption. Result: 73/79 now have a claim, **48 in-scope**, 4 still leak (flagged).

**🔑 Auditor reliability caveat:** the 235B **over-rejects carve-outs** (a celebrity *death* hoax → it said
out-of-scope; death is our IN carve-out) and **over-accepts political rhetoric**. So we only auto-trust the
185 obvious out-of-scope; everything ambiguous goes to the human (HTML). The 706 "confirmed positives" and
456 "confirmed negatives" were NOT individually reviewed — accepted as-is (a known residual risk).

---

## 3. The HTML review tool (the pause point)

`eval/data/gate_review.html` (self-contained, images embedded; generated by
`eval/scripts/build_review_html.py`). **198 cases** Daniel must judge, 3 sections:
- **A — 32 risky positive→negative** (carve-out/politics; default FLAG=keep positive)
- **B — 87 negatives flagged check-worthy** (mostly rhetoric errors; default PASS=keep negative)
- **C — 79 image-only with RECOVERED caption** (image + recovered caption + original `claim_text` +
  ⚠️ leakage flag; default FLAG if in-scope, DROP if leaky)

Per card: `🚩 FLAG` = check-worthy → **positive** · `➡️ PASS` = ignore → **negative** · `🗑️ DROP` = remove.
Selections auto-save to localStorage; "Generate responses" emits a compact `A01:flag B07:pass C12:drop …`
block to copy back (or Download). The **ID→row mapping is `eval/data/gate_review_items.parquet`** (columns:
id, uid, current_label, text, image_path, bucket — for bucket C, `text` = the recovered caption to use).

---

## 4. ▶ HOW TO RESUME (exact next steps)

1. **Get Daniel's HTML responses** (the `A01:flag …` block).
2. **Build the apply step (NOT yet written):** parse `ID:choice` → join to `gate_review_items.parquet` by id
   → flag=positive, pass=negative, drop=remove. For **C-flag**, the post becomes a positive whose text is the
   **recovered caption** (carry it into the dataset; it's an image+caption positive now, not image-only).
3. **Apply all confirmed changes in `build_gate_splits.py`:** the 185 safe out-of-scope → negatives; the
   reviewed-198 per Daniel; keep the 706/456 confirmed. Wire it to read a persisted decisions file so it's
   reproducible. **Log every change to the provenance ledger** (`gate_dataset_provenance.py`).
4. **Rebuild splits** (stable `md5%2` fold) and **re-run the eval** (`run_v8dev`-style sweep) on the CLEAN
   data: negatives + positives + image-only + video + mislabels.
5. **Decide the prompt (v8g leading → promote v9) AND the model (Qwen vs Gemma-v8g)** on the clean numbers,
   **reported by axis**, dev first / test once. Gemma-v8g currently ties Qwen on recall + wins on cost; confirm on 24mo.

---

## 5. Open questions / TODO

- **v8 vs v6-flat:** unresolved — needs the clean re-run. v8 fixes the oos leak; confirm it holds recall.
- **Trust of the 706/456 "confirmed":** not human-reviewed; the auditor is imperfect. Spot-check?
- **Image-authenticity (`artifact`) axis — RESOLVED (Daniel):** the gate DOES pass images upstream, so it
  MUST flag image-authenticity cases → `artifact`-axis recall counts. **Always report gate recall by axis**
  (content / attribution / artifact); 33% of positives are artifact, and lumping them hid the whole Qwen↔Gemma story.
- **Model choice (Qwen vs Gemma) — DEFERRED to the clean 24mo set.** On the clean subset, **Gemma-4-26B +
  the v8g prompt ties Qwen on every recall axis, beats it on specificity, ~2.2× cheaper**; Qwen's only edge is
  latency (2.3s vs 3.3s p50). Re-test rigorously (by axis) on 24mo before committing. Seed-2.0-mini is out.
- **Promote v8g → v9:** fold the v8g scope fixes into the canonical `GATE_SYSTEM` — they're model-agnostic
  (personal = a private individual's OWN life only; public-figure private-life/conduct/image is IN; private
  person with societal implications is IN; image-as-claim override). The 3 edits live in
  `eval/scripts/gemma_prompt_test.py:EDITS`; also test them on Qwen.
- **Broader caption recovery:** we recovered only the 79 no-claim image-only; the other ~230 image-only carry
  the claim in the image already. Worth recovering captions more broadly? (leakage risk.)
- **Harness robustness fix:** derive `kept = public_interest AND misinfo_if_false AND locus!=video` (the prompt
  already mandates it) → recovers ~2 self-inconsistency misses where the model set check_worthy=false despite pi∧mif.
- **Video policy:** gate routes by locus; only ~25% of video posts are truly video-dependent (the rest carry
  text/poster-frame claims). Revises clog Decision 4 "drop ALL video" — needs a product call.
- **Residual `opinion_personal` FPs:** the Qatar/Macron cluster is *defensibly* check-worthy — don't chase it.

## 6. File map (the workstream's code + artifacts)
- `eval/scripts/gate_eval.py` — the gate + eval harness; **`GATE_SYSTEM` = v8**. `--tag` namespaces prompt
  versions; `uid` key handles image-only rows; retry-on-empty-parse; resume skips only *successful* rows.
- `eval/scripts/build_gate_splits.py` — dataset assembly + **stable split** + the oos reclassification logic.
- `eval/scripts/audit_dataset.py` — full 1,466-item audit (235B) → `dataset_audit.parquet`.
- `eval/scripts/recover_image_captions.py` — caption recovery + scrub → `recovered_captions.parquet`.
- `eval/scripts/build_review_html.py` — generates `gate_review.html` + `gate_review_items.parquet`.
- `eval/scripts/triage_positives_scope.py` — the first conservative oos topic scan.
- `eval/scripts/gate_dataset_provenance.py` — the provenance ledger generator.
- `eval/scripts/model_bakeoff.py` — 3-model VLM bake-off (spec/recall-by-axis/latency/cost) → `bakeoff_results.parquet`.
- `eval/scripts/gemma_prompt_test.py` — the **v8g** prompt (3 model-agnostic scope edits in `EDITS`) + Gemma re-test.
- Reports: `eval/data/gate_eval_report.md` (v6 results + method), `eval/data/gate_fp_triage.md` (107-FP audit),
  `eval/data/gate_dataset_provenance.{md,parquet}` (every label change).
- Splits: `eval/data/{negatives,positives,image_only}_{dev,test}.parquet`, `video_sample.parquet`,
  `gate_mislabels.parquet` (7 must-pass).
