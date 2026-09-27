# Prompt-Engineering SOTA (2025–2026) for the Claim-Extraction + Authenticity Prompt

**Status:** research + design guidance only. This brief does **not** rewrite the prompt; the
final wording is decided collaboratively afterward. It maps 2024–2026 evidence onto the specific
prompt at `src/eval/scripts/image_extraction_eval.py` (`EXTRACT_SYS_SINGLE`, lines 65–74).

**Target prompt (recap).** Qwen3-VL, OpenAI-compatible API, `temperature 0`,
`response_format: json_object`. Input = post text + image(s) + optional quoted tweet. Two coupled
jobs in one call: (1) **extract** the single primary check-worthy claim as an atomic,
decontextualized proposition; (2) emit an **orthogonal boolean** `authenticity` (true iff the
*media's own* provenance is itself checkable), plus `purported` when true. Output JSON:
`{image_serialization, primary_claim, authenticity, purported, flags}`.

**Four failure modes** this brief targets: (a) claim **over-fragmentation**; (b) **identity
hallucination**; (c) JSON **truncation**; (d) **over-triggering** the `authenticity` boolean.

**Legend.** Each cited finding is flagged **[GENERAL]** (cross-model) or **[MODEL: X]**
(tied to a specific model/provider). Confidence is noted where it matters. "Provider-official" =
OpenAI/Anthropic/Google/Alibaba docs; "peer-reviewed" = published venue; "preprint" = arXiv only.

---

## 1. Executive summary — highest-leverage changes

In rough priority order (leverage × evidence-strength):

1. **Convert the run-on paragraph into explicitly delimited, single-purpose sections** and **order
   them deliberately.** Format sensitivity is a real, measurable accuracy knob — spurious format
   choices alone move accuracy by up to ~76 points on some models ([Sclar et al., ICLR 2024](https://arxiv.org/html/2310.11324v2)).
   Anchor the most important rules at **both top and bottom** of the prompt; never bury a decision
   rule mid-prompt ("lost in the middle," [Liu et al., TACL 2024](https://arxiv.org/abs/2307.03172)). **[GENERAL]**

2. **Keep `image_serialization` (perception) first in the JSON, and add a *compact* reason field
   immediately before the `authenticity` boolean.** Autoregressive models condition the answer on
   earlier tokens; forcing answer-before-reasoning is the dominant cause of the "JSON hurts
   reasoning" effect — reason-then-answer ordering recovers it (e.g., Claude-3-Haiku GSM8K
   23%→87% just by loosening/ordering; [Tam et al., EMNLP 2024](https://arxiv.org/abs/2408.02442)).
   The current schema already puts `image_serialization` first — that is correct; extend the same
   "describe/justify before commit" logic to the boolean. **[GENERAL]**

3. **Fix `authenticity` over-triggering with an explicit decision rule + 1–2 near-miss *negative*
   contrastive examples** (image merely illustrates a content claim → `authenticity=false`).
   Hard-negative / contrastive demonstrations are the best-evidenced lever for a precision-sensitive
   boolean (C-ICL: NER +3.76 F1, RE +26 F1, [arXiv:2402.11254](https://arxiv.org/abs/2402.11254);
   contrastive ICL 76% vs 64%, [arXiv:2401.17390](https://arxiv.org/abs/2401.17390)) — but examples
   teach the *rule* weakly, so **state the rule explicitly too** ([OpenAI GPT-4.1 guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)). **[GENERAL]**

4. **Harden the JSON contract for Qwen + `json_object`.** `json_object` guarantees only *syntactic*
   validity (it can drop fields, add fields, mis-type) — there is no schema enforcement, so you must
   **validate every output downstream** ([OpenAI structured-outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)).
   Qwen-specific: the prompt **must contain the literal word "json"** or the API 400s, and Alibaba
   **warns against setting `max_tokens` in JSON mode** (it truncates mid-string)
   ([Alibaba Model Studio JSON-mode docs](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)).
   Structurally, **single-claim output + a capped `flags` list** is the most robust anti-truncation
   design ([general guidance](https://dev.to/pockit_tools/llm-structured-output-in-2026-34pk)). **[MODEL: Qwen]**

5. **Do not add heavy chain-of-thought.** CoT's benefit is concentrated in math/symbolic tasks
   (+12–14pp) and is ~**+0.7pp** on knowledge/"soft" tasks ([Sprague et al., ICLR 2025](https://arxiv.org/abs/2409.12183));
   it can *hurt* rule-with-exceptions classification by up to −36pp ([Liu et al., ICML 2025](https://arxiv.org/abs/2410.21333)).
   Use a *bounded* one-phrase rationale on the authenticity decision only — not a free-form
   scratchpad that bloats output (and risks truncation, failure mode c). **[GENERAL]**

6. **Identity:** keep the "name only if confident, else describe generically + flag" rule and
   **explicitly allow abstention** ("if not legible / not visible, say so"). Abstention is the
   cheapest, largest single hallucination reduction ([risk-aware selective prompting, 2026](https://arxiv.org/abs/2605.28123); survey [arXiv:2507.19024](https://arxiv.org/abs/2507.19024)).
   Note: general VLMs are materially worse than specialist face models at identity (Qwen2-VL 81%
   vs IResNet 97%, with demographic gaps; [arXiv:2510.14866](https://arxiv.org/abs/2510.14866)) —
   so confidence-gated naming is essential. **[GENERAL / MODEL: Qwen-VL]**

**One thing to drop:** the implicit assumption that a "You are an expert…" persona buys accuracy —
peer-reviewed evidence says it does not (see §1 below). Keep a *minimal* role only to scope the task.

---

## 2. Section per research question

### Q1 — Structure & organization of a multi-objective prompt

- **Persona doesn't buy accuracy.** Across 4 model families, 2,410 factual questions, 162 personas,
  system-prompt personas gave **no consistent accuracy gain** and best-persona selection did "no
  better than random" ([Zheng et al., Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.888/)). A
  Wharton benchmark found expert personas gave **no factual-accuracy benefit** and can slightly hurt
  ([Prompting Science Report 4, 2025](https://arxiv.org/pdf/2512.05858)). **[GENERAL]** Provider guides still
  recommend a role, but explicitly for *tone/behavior scope*, not accuracy ([Anthropic best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)).
  → *Keep a one-line role for scope; expect zero-to-slightly-negative effect on the boolean.*

- **Delimiters: no universal winner; it is a real accuracy knob.** Anthropic recommends **XML tags**
  to separate instructions/context/input ([Anthropic XML guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags)) **[MODEL: Claude]**;
  OpenAI recommends **Markdown headers** first, says XML also works, and found JSON delimiting
  "performed particularly poorly" in long context ([GPT-4.1 guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)) **[MODEL: GPT-4.1]**.
  Empirically the optimal format is **model-specific** — GPT-3.5 favored JSON, GPT-4 favored Markdown,
  swings up to ~40% ([He et al., 2024](https://arxiv.org/html/2411.10541v1)); spurious format choices
  alone span up to 76 accuracy points and don't vanish with scale ([Sclar et al., ICLR 2024](https://arxiv.org/html/2310.11324v2)). **[GENERAL]**
  → *Pick one delimiter style, apply it consistently, and treat it as a tunable to A/B test on the
  68%→ held-out set — not cosmetics. For Qwen specifically, no public delimiter benchmark exists; XML
  or `### Markdown` are both reasonable; test.*

- **Two coupled objectives multiply failure rates.** "Curse of instructions": prompt-level
  (all-constraints-satisfied) accuracy decays roughly as *pᴺ* in the number of instructions
  ([ManyIFEval, Findings EMNLP 2025](https://arxiv.org/abs/2509.21051)). Multi-task interference is
  real but model-dependent ([MDPI Electronics 2025](https://www.mdpi.com/2079-9292/14/21/4349)). **[GENERAL]**
  *But:* for correlated multi-field extraction a **single joint call is more robust** than separate
  per-field calls ([arXiv:2503.16868](https://arxiv.org/pdf/2503.16868)). → *Keep one call, but give
  each objective its own crisp, minimal, separately-delimited sub-block so they don't bleed.*

- **Ordering effects.** Conflicting instructions resolve toward the **end** of the prompt, and the
  recommended order is Role → Instructions → (Reasoning) → Output-format → Examples → Context
  ([OpenAI GPT-4.1 guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)) **[MODEL: GPT-4.1]**.
  Put key instructions at **top and bottom**; Anthropic reports query-at-the-end can lift quality
  ~30% ([best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)).
  A U-shaped positional bias appears even in short instruction lists (first-stated objective wins
  attention; [Order Effect, 2025](https://arxiv.org/html/2502.04134v2)). → *State the objective you
  most care about first; restate the output-format/contract last.*

### Q2 — Structured / JSON-output reliability

- **`json_object` ≠ schema enforcement.** It guarantees only valid JSON syntax; it can omit required
  keys, add keys, or wrong-type values. Native `json_schema`/constrained decoding (OpenAI, Anthropic
  Nov-2025, Gemini 2.5+) raises schema adherence from <40% (prompt-only) to ~100%
  ([OpenAI 2024](https://openai.com/index/introducing-structured-outputs-in-the-api/);
  [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs);
  [Google 2025](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-api-structured-outputs/)). **[MODEL-SPECIFIC; principle GENERAL]**
  **Qwen/DashScope supports only `json_object`, not strict `json_schema`** — so we are in the
  "validate-yourself" regime ([Alibaba docs](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)). **[MODEL: Qwen]**

- **Field ordering — reasoning before answer.** This is the single most important schema decision.
  In "Let Me Speak Freely?", **100% of GPT-3.5 JSON-mode responses put the answer key before the
  reason key**, suppressing CoT; reordering/loosening recovered it (Claude-3-Haiku GSM8K 23%→87%)
  ([Tam et al., EMNLP 2024](https://arxiv.org/abs/2408.02442)). A 2026 study reports answer-before-
  reasoning causes up to a **67% relative** accuracy drop in AR models ([arXiv:2601.22035](https://arxiv.org/abs/2601.22035), preprint — treat the exact figure as indicative). **[GENERAL]**
  → *Order keys: `image_serialization` (perception) → a short `authenticity_reason` → `authenticity`
  → `purported` → `primary_claim` → `flags`. The perception-first ordering you already have is the
  right instinct.*

- **Schema design.** Flatten over nesting; use unambiguous field names (the model uses the field
  name as a semantic cue); use **enums for categoricals** (constraining a classification answer space
  *helps* — DDXPlus improved under JSON; [Tam et al.](https://arxiv.org/abs/2408.02442)). LLM-driven
  schema optimization (mostly *flattening* + richer field descriptions) cut extraction errors ~92%
  and lifted accuracy up to +64.7% ([PARSE, arXiv:2510.08623](https://arxiv.org/html/2510.08623v1)). **[GENERAL]**

- **Truncation (failure mode c).** Constrained decoding prevents *invalid tokens*, not *truncation*;
  always check `finish_reason == "length"` and retry ([OpenAI guide](https://developers.openai.com/api/docs/guides/structured-outputs)). **[GENERAL]**
  **Qwen-specific:** the prompt must contain the word "json"; **do not set `max_tokens` in JSON mode**
  (truncates mid-string) ([Alibaba JSON-mode docs](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)). **[MODEL: Qwen]**
  Structural mitigations beat just raising `max_tokens`: **single-claim output, cap the `flags` list
  (e.g. ≤3), avoid verbose echo fields** ([general guidance, 2026](https://dev.to/pockit_tools/llm-structured-output-in-2026-34pk)).
  The single-claim `EXTRACT_SYS_SINGLE` is already the right anti-truncation choice vs the multi-claim variant.

### Q3 — Few-shot / in-context examples (extraction + classification)

- **Do they help?** For span-extraction they are often load-bearing (removing few-shot dropped GPT-4
  NER F1 ~50 points; [npj AI 2025](https://www.nature.com/articles/s44387-025-00062-2)). For a
  well-specified binary, marginal gain over a strong zero-shot instruction is smaller. **[GENERAL]**

- **How many?** Provider consensus is **3–5** ([Anthropic multishot](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/multishot-prompting)).
  For **classification specifically, accuracy peaks ~5–20 shots then declines** ("over-prompting";
  [arXiv:2509.13196](https://arxiv.org/html/2509.13196v1)) — unlike reasoning/translation where
  many-shot keeps paying ([Many-Shot ICL, NeurIPS 2024](https://arxiv.org/abs/2404.11018)). **[GENERAL]**

- **Selection.** Similarity-retrieved + diverse exemplars beat random ([survey arXiv:2401.11624](https://arxiv.org/abs/2401.11624));
  **near-miss / decision-boundary ("hard negative") examples sharpen precision** more than easy
  clear-cut ones ([curriculum selection arXiv:2411.18126](https://arxiv.org/abs/2411.18126)). **[GENERAL]**

- **Surface-form overfitting — and a key 2023+ correction.** Min et al. famously found random demo
  labels barely hurt ([EMNLP 2022](https://arxiv.org/abs/2202.12837)) — **but at frontier scale,
  labels DO matter**: large models follow flipped labels (ChatGPT 96%→17% with 8 flipped demos;
  [Wei et al. 2023](https://arxiv.org/html/2303.03846)). **Treat every example label as gold — wrong
  labels get followed.** Examples teach *format* strongly, sometimes over content
  ([arXiv:2408.08780](https://arxiv.org/html/2408.08780)), so **keep example format byte-identical to
  the target JSON**, and don't rely on examples to *imply* the rule. **[GENERAL]**

- **Negative / contrastive examples to stop over-triggering (failure mode d) — the core item.**
  - **C-ICL** appends near-miss *wrong* predictions tagged as wrong: NER +3.76 F1, RE +26 F1; and
    notes **too many negatives add noise** — there is an optimal positive:negative ratio
    ([arXiv:2402.11254](https://arxiv.org/abs/2402.11254)). **[MODEL: open]**
  - **Contrastive ICL** pairs preferred/less-preferred examples and asks the model to articulate the
    contrast: 76% vs 64% standard few-shot ([arXiv:2401.17390](https://arxiv.org/abs/2401.17390)). **[GENERAL]**
  - Pair negatives with the **explicit rule** — examples under-specify a boolean and literal-following
    models (GPT-4.1) may not generalize from examples alone ([OpenAI guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide); [Gemini](https://ai.google.dev/gemini-api/docs/prompting-strategies)). **[GENERAL]**
  - **Balance true/false** examples and don't end the example block on the label you don't want
    over-predicted (recency + majority-label bias; [Zhao et al., ICML 2021](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)). **[GENERAL]**
  - **Caveat on "explain why":** per-example rationales help *reasoning/generation* tasks but did
    **not** help flat sentiment classification ([arXiv:2307.05052](https://arxiv.org/html/2307.05052)).
    → Add a one-line "why-negative" to the contrastive authenticity example, but **A/B-test whether it
    actually moves precision** before committing.

### Q4 — Reasoning: CoT / think-then-answer vs direct

- **CoT mostly helps math/symbolic.** +12.3pp math, +14.2pp symbolic, but **+0.7pp** on
  knowledge/commonsense/soft tasks; ~95% of MMLU's CoT gain comes from questions with an "=" sign
  ([Sprague et al., ICLR 2025](https://arxiv.org/abs/2409.12183)). **[GENERAL]**
- **CoT can backfire** on rule-with-exceptions classification and visual recognition — up to −36pp
  ([Liu et al., ICML 2025](https://arxiv.org/abs/2410.21333)). The authenticity boolean is exactly a
  rule-with-exceptions classification — so a long deliberation is a liability, not an asset. **[GENERAL]**
- **The useful pattern: a *bounded* reason field before the answer.** Reason-then-answer ordering
  recovers what strict JSON suppresses ([Tam et al.](https://arxiv.org/abs/2408.02442); [Anthropic
  "let Claude think"](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-engineering/chain-of-thought) — "without outputting its thought process, no thinking occurs"). **[GENERAL]**
- **Keep it compact.** Minimal "chain-of-draft" reasoning (~5 words/step) matched full CoT at ~7–32%
  of the tokens ([Chain of Draft, arXiv:2502.18600](https://arxiv.org/html/2502.18600v1)), with a
  task-intrinsic floor below which accuracy drops ([token-complexity, arXiv:2503.01141](https://arxiv.org/html/2503.01141v1)).
  → *One short phrase, not a paragraph — directly mitigates truncation (c).* **[GENERAL]**
- **Providers say turn reasoning OFF for extraction/classification:** OpenAI `reasoning_effort: none`
  for latency-critical classification; Google `thinking_budget: 0` "right call for structured data
  extraction" (and thinking output is billed/priced higher, e.g. Gemini-2.5-Flash ~5.8×)
  ([OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning); [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)). **[MODEL-SPECIFIC, convergent]**
- **Qwen caveat:** Qwen **thinking mode is incompatible with `response_format`** — enabling both
  errors and reasoning text can leak into `content` and break JSON parsing; do not enable Qwen
  thinking on this call ([Alibaba docs](https://www.alibabacloud.com/help/en/model-studio/json-mode)). **[MODEL: Qwen]**
- **Self-consistency is overkill** for extraction (multiplicative cost, diminishing/negative returns
  on modern models; [arXiv:2511.00751](https://arxiv.org/html/2511.00751)). **[GENERAL]**

### Q5 — VLM-specific prompting

- **Object/attribute hallucination is real and benchmarked** (POPE, CHAIR, HallusionBench — SOTA
  GPT-4V only 31% question-pair accuracy on HallusionBench), driven by **"language dominance"**: the
  model under-attends to the image and lets text priors fill gaps
  ([HallusionBench, CVPR 2024](https://arxiv.org/abs/2310.14566); [POPE, EMNLP 2023](https://arxiv.org/abs/2305.10355)). **[GENERAL]**
- **Highest-leverage prompt levers (ranked):**
  1. **Allow abstention** ("if not visible / not legible, say so") — cheapest, largest error
     reduction ([selective prompting 2026](https://arxiv.org/abs/2605.28123)).
  2. **Transcribe / describe *before* judging** — separate perception from inference; describe-then-
     reason improved faithfulness 2–33% ([visual-description grounding, arXiv:2405.15683](https://arxiv.org/abs/2405.15683)).
     *The current prompt already does step (1) transcribe + non-judgmental describe — keep it; it is
     well-supported.*
  3. **Neutral, non-leading phrasing** — don't presuppose an object/identity is present (POPE
     adversarial split is hardest precisely because it presupposes; [POPE](https://arxiv.org/abs/2305.10355)).
- **Identity:** confidence-gated naming + abstention. General VLMs are materially worse than
  specialist face models (Qwen2-VL 81% vs 97%), with demographic disparities (76.7% vs 60.4% across
  groups; [arXiv:2510.14866](https://arxiv.org/abs/2510.14866)), and they tend to **guess rather than
  abstain** under misleading context ([survey](https://arxiv.org/abs/2507.19024)). Provider stance
  **differs**: GPT-4V and Claude **refuse to name people** ([Claude Vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)),
  whereas **Qwen3-VL advertises celebrity/landmark recognition as a feature** ([Qwen3-VL tech report, arXiv:2511.21631](https://arxiv.org/abs/2511.21631)).
  → On Qwen3-VL we *can* name people, but unreliably — so the existing "name only if confident, else
  describe generically + flag, never guess" rule is exactly right and should stay prominent. **[MODEL: Qwen-VL]**
- **OCR / in-image text:** Qwen3-VL is a documented OCR strength — 32 languages, robust to
  low-light/blur/tilt ([Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)); lean on it. **[MODEL: Qwen-VL]**
- **Multi-image:** label images explicitly ("Image 1:", "Image 2:"); VLMs have **position bias**
  (first/last favored, mid-images neglected) ([arXiv:2503.13792](https://arxiv.org/abs/2503.13792);
  [Anthropic](https://platform.claude.com/docs/en/build-with-claude/vision) / OpenAI vision guides). **[GENERAL]**
- **Grounding tooling (Set-of-Mark, bounding boxes)** improves grounding but needs segmentation
  tooling, not prompt-only ([SoM, arXiv:2310.11441](https://arxiv.org/abs/2310.11441)) — out of scope
  for a single text-prompt change; noted for future. **[MODEL: GPT-4V/Gemini]**

### Q6 — Prompt-level mitigations mapped to our four failure modes

| Failure mode | Mitigation (prompt-level) | Evidence |
|---|---|---|
| **(a) Over-fragmentation** | Explicit "ONE primary claim" rule + restate it as a hard constraint at the *end* (recency); 1 contrastive example showing a fragmented answer as **wrong** and the consolidated one as right; keep single-claim output (no list to over-fill). | Curse-of-instructions / order ([2509.21051](https://arxiv.org/abs/2509.21051), [2502.04134](https://arxiv.org/html/2502.04134v2)); contrastive demos ([2402.11254](https://arxiv.org/abs/2402.11254)) |
| **(b) Identity hallucination** | "Name only if confident (from in-image text/caption/quoted tweet/clear recognition), else describe generically + add `flags` note; never guess." Explicit **abstention** permitted. Keep transcribe-before-judge. | Confidence-gated abstention ([2605.28123](https://arxiv.org/abs/2605.28123)), VLM identity weakness ([2510.14866](https://arxiv.org/abs/2510.14866)) |
| **(c) Truncation** | Single-claim output; cap `flags` (≤3); compact (one-phrase) reason field, no free-form scratchpad; **don't set `max_tokens` (Qwen)**; check `finish_reason=="length"` + retry; validate downstream. | OpenAI/Alibaba docs; structural anti-truncation guidance |
| **(d) Authenticity over-trigger** | Explicit positive **and negative** decision rule ("authenticity=true ONLY when the media's *own* provenance is the checkable point; FALSE when the image merely illustrates a content claim"); 1–2 **near-miss negative** examples; compact `authenticity_reason` field placed *before* the boolean; balanced true/false example labels. | Contrastive ICL ([2401.17390](https://arxiv.org/abs/2401.17390), [2402.11254](https://arxiv.org/abs/2402.11254)); reason-before-answer ([2408.02442](https://arxiv.org/abs/2408.02442)); label bias ([Zhao 2021](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)) |

---

## 3. Recommended prompt skeleton (layout only — wording decided collaboratively)

Sections, top to bottom. Use one consistent delimiter style (XML tags **or** `###` Markdown — test
which Qwen prefers; both are defensible). Keep the literal word **"json"** present (Qwen requirement).

```
[ROLE]            One line. Scope only ("You extract the single checkable claim a fact-checker
                  would verify, and assess whether the media's own authenticity is itself a
                  checkable point"). No "expert" persona inflation.

[INPUTS]          What the model receives: post text, image(s) labelled "Image 1/2…", optional
                  quoted tweet. Note any image may be absent.

[TASK 1 — EXTRACT]   The claim job. "ONE primary, most salient/disputed proposition… atomic,
                  decontextualized, bare proposition (not 'the image shows…')." Locus may be text,
                  in-image text/visual, or endorsed quoted tweet.

[TASK 2 — AUTHENTICITY]  The orthogonal boolean job, kept in its OWN block so it doesn't bleed into
                  Task 1. Positive rule + explicit NEGATIVE rule (false when image merely
                  illustrates a content claim). When true, give `purported` (who/what/when/where the
                  media claims to show). State that the two tasks are independent and BOTH can be true.

[GROUNDING RULES] Identity (name-only-if-confident-else-describe-+-flag, never guess); abstention
                  ("if not legible/visible, say so"); transcribe/describe before judging; describe
                  visuals non-judgmentally (do NOT assess truth); use only what is visible.

[EXAMPLES]        3–5 few-shot, byte-identical to the output JSON. See §4. Includes the contrastive
                  authenticity-negative(s) and one anti-fragmentation example.

[OUTPUT CONTRACT] The JSON schema, last (recency). Field order matters:
                  image_serialization → authenticity_reason (short) → authenticity →
                  purported (or null) → primary_claim → flags (≤3). "JSON only."

[FINAL REMINDER]  Restate the 2–3 load-bearing constraints (ONE claim; never guess a name;
                  authenticity=false if image only illustrates) — top-and-bottom anchoring.
```

Key structural deltas from the current one-paragraph prompt:
- Split the run-on text into the labelled blocks above (esp. isolate Task 2 from Task 1).
- Add an explicit **negative** authenticity rule (currently only the positive notion exists, and the
  artifact axis is even *excluded* from the current eval — see `image_extraction_eval.py:22`).
- Add a **short `authenticity_reason`** field *before* the boolean (compact reason-then-answer).
- Cap `flags`; keep single-claim; restate constraints at the end.
- Add few-shot examples (the current prompt is zero-shot).

---

## 4. Few-shot strategy

**How many:** **4–6 examples** (provider default 3–5; classification saturates ~5–20). Don't exceed
~8 — over-prompting degrades binary classification ([2509.13196](https://arxiv.org/html/2509.13196v1)).

**Composition (suggested mix):**
1. **Canonical positive-extraction** — a clear single-claim post (authenticity=false), shows the
   decontextualized-proposition format and a clean `image_serialization`.
2. **Authenticity-true** — media whose *own* provenance is the claim (e.g. "this photo shows event X";
   doctored/AI-generated/out-of-context), so `authenticity=true` + `purported` filled. This sets the
   positive boundary.
3. **Contrastive authenticity-NEGATIVE (the key one)** — a post where an image is *present and
   on-topic but merely illustrates a content claim* → `authenticity=false`. This is the near-miss that
   stops over-triggering. Add a one-line `authenticity_reason` like "image illustrates the stated
   claim; its provenance is not itself in dispute." (A/B-test whether the rationale helps — rationales
   don't always help classification, [2307.05052](https://arxiv.org/html/2307.05052).)
4. **Anti-fragmentation** — a post that *could* be split into many trivial sub-claims, shown
   collapsed to the ONE primary proposition (the over-fragmented version is the wrong answer).
5. **Identity-abstention** — a depicted person the model shouldn't confidently name → generic
   description + a `flags` note, demonstrating the never-guess rule.
6. *(optional)* a multi-locus example where the claim lives in the quoted tweet, not the post text.

**Construction rules (evidence-backed):**
- **Every label must be gold** — on a frontier VLM wrong example labels get *followed*
  ([Wei 2023](https://arxiv.org/html/2303.03846)). Draw examples from your gold/held-out set, not improvised.
- **Format byte-identical** to the target JSON, same key order ([2408.08780](https://arxiv.org/html/2408.08780)).
- **Balance the boolean** (≈ equal true/false) and **do not end the example block on `authenticity:true`**
  — recency/majority bias would push over-triggering ([Zhao 2021](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)).
- **Prefer near-miss negatives over obvious ones** — obvious non-cases teach little about the boundary
  ([2402.11254](https://arxiv.org/abs/2402.11254), [2411.18126](https://arxiv.org/abs/2411.18126)).
- You already have a `hard_negative` category and a synthetic-negatives pipeline
  (`build_synthetic_negatives.py`) — reuse that machinery to source near-miss authenticity-negatives.

---

## 5. What NOT to do / myths

- **Myth: "You are an expert fact-checker" improves accuracy.** It doesn't; may slightly hurt
  ([EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.888/), [Wharton 2025](https://arxiv.org/pdf/2512.05858)).
- **Myth: add chain-of-thought to make extraction better.** ~+0.7pp on non-math tasks, and it can
  *hurt* rule-with-exceptions classification ([ICLR 2025](https://arxiv.org/abs/2409.12183), [ICML 2025](https://arxiv.org/abs/2410.21333)).
- **Myth: `json_object` guarantees the schema.** It guarantees only valid *syntax* — validate every
  field downstream ([OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs)).
- **Don't raise `max_tokens` as the only truncation fix** — and on **Qwen, don't set `max_tokens` in
  JSON mode at all** (it truncates mid-string) ([Alibaba](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)). Fix structurally.
- **Don't put the answer before its reasoning** in the JSON — it suppresses the reasoning
  ([Tam et al.](https://arxiv.org/abs/2408.02442)).
- **Don't enable Qwen "thinking" mode on this call** — incompatible with `response_format`, leaks
  into `content` ([Alibaba](https://www.alibabacloud.com/help/en/model-studio/json-mode)).
- **Don't over-stuff few-shot** — binary classification over-prompts past ~5–20 examples
  ([2509.13196](https://arxiv.org/html/2509.13196v1)); and don't use sloppy/improvised example labels.
- **Don't presuppose presence** in phrasing ("describe the doctored area…") — leading prompts inflate
  hallucination ([POPE](https://arxiv.org/abs/2305.10355)).
- **Don't assume one delimiter is universally best** — it's model-specific; test on Qwen
  ([He et al. 2024](https://arxiv.org/html/2411.10541v1)).

---

## 6. Open questions / tradeoffs to decide

1. **Temperature vs Qwen's recommendation.** The pipeline runs `temperature 0` (greedy) for
   determinism, but Qwen3-VL's **official model card recommends temp 0.7 (Instruct) / 0.6 (Thinking)
   and explicitly sets `greedy=false`** ([Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL),
   verified). Tradeoff: reproducible eval (temp 0) vs the sampling regime the model was tuned for.
   *Decide:* keep temp 0 for eval determinism (and accept possible off-distribution behavior), or
   test a low-but-nonzero temp. Worth a small A/B on the held-out set.
2. **Does the compact `authenticity_reason` field help enough to justify the extra tokens?** Reason-
   before-answer helps in general, but rationales don't always help flat classification
   ([2307.05052](https://arxiv.org/html/2307.05052)). A/B with/without it.
3. **Delimiter style for Qwen** (XML vs `###`) — no public Qwen benchmark; pick via a quick A/B.
4. **Few-shot images cost tokens** (Qwen3-VL ~1024 tokens/MP per image). Few-shot *with images* is
   expensive; consider text-only example posts where the image content can be conveyed in the
   serialization field, reserving real images for the 1–2 authenticity examples where the *image* is
   the point.
5. **Native structured output is unavailable on Qwen** (`json_object` only). If schema-adherence
   problems persist, consider a validate-and-repair second pass, or routing this call to a provider
   with `json_schema` — a larger architectural decision, out of scope here.
6. **Where does the artifact/authenticity axis get *scored*?** The current extraction eval *excludes*
   it (`image_extraction_eval.py:22-23`). Adding `authenticity` to the live prompt means deciding the
   gold + judge for that axis before its precision can be measured (relevant to failure mode d).

---

## Source-quality & confidence notes

- **Strongest (peer-reviewed, load-bearing):** Tam et al. EMNLP 2024 (format/ordering); Sprague et al.
  ICLR 2025 + Liu et al. ICML 2025 (CoT scope); Zheng et al. EMNLP 2024 (personas); Sclar et al. ICLR
  2024 + Liu et al. TACL 2024 (format sensitivity / lost-in-the-middle); POPE EMNLP 2023 +
  HallusionBench CVPR 2024 (VLM hallucination); Zhao et al. ICML 2021 (label bias).
- **Provider-official (high confidence, model-specific):** OpenAI structured-outputs/GPT-4.1 guides;
  Anthropic vision + prompt-engineering docs; Google Gemini structured-output/thinking docs; Alibaba
  Model Studio Qwen JSON-mode docs; Qwen3-VL README + tech report (arXiv:2511.21631).
- **Weaker / treat exact numbers as indicative (preprints or secondary):** the 67% answer-before-
  reasoning figure (arXiv:2601.22035); the "~15% from key ordering" and "30–50% from abstention"
  estimates; some 2026 preprints (arXiv:2603.*, 2604.*, 2605.* — directionally consistent but not yet
  peer-reviewed). The brief's recommendations do not hinge on these.
- **Notable contradictions, flagged in-text:** (i) providers disagree on delimiters (Anthropic XML vs
  OpenAI Markdown) — no universal winner; (ii) "JSON hurts reasoning" (Tam) is contested by dottxt's
  rebuttal ([blog](https://blog.dottxt.ai/say-what-you-mean.html)) and [arXiv:2501.10868] — reconciled
  as "the harm is from answer-before-reasoning + unequal prompts, not constraint per se"; (iii) Min
  et al. "random labels are fine" is superseded at frontier scale by Wei et al.; (iv) identity stance
  differs by provider (GPT-4V/Claude refuse; Qwen3-VL recognizes).
