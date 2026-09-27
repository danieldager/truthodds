# Automatic Fact-Checking — Literature Briefing

*Compiled 2026-06-27. Part 1 = the papers that actually shaped our system (traced to where
they appear in the repo). Part 2 = a 2025–2026 scan of the field by pipeline stage. Part 3 =
what's new since we designed, and where it could feed back in.*

---

## Part 1 — Papers that shaped our system

Grouped by the stage they influenced; every entry is traceable in the repo (`src/CLAUDE.md`,
the `src/clog/` logs, `src/docs/conversation_log.md`, the slides, `src/pipeline/verify.py`, and
the archived SOTA reviews under `src/eval/scripts/_archive/feed_study/docs/`).

### Tier-3 verification loop (core architecture)
- **ClaimCheck** — Putta, Devasier & Li (UT Arlington), arXiv:2510.01226 (2025). Our Tier-3
  verifier is a near-faithful clone of its **plan → search → summarize → synthesize → verdict**
  loop; Serper top-3 retrieval and the 4-class verdict prompt come from here. 76.4% AVeriTeC with
  Qwen3-4B is our accuracy anchor. → `src/pipeline/verify.py`, `src/CLAUDE.md`, slides.
- **FIRE** — Xie et al., Findings of NAACL 2025, arXiv:2411.00784. Source of our
  **confidence-gated iterative retrieval** — the synthesis call emits `next_query: null` when
  confident/done, else a new query. (FIRE reports ~7.6× fewer LLM calls / ~16.5× fewer searches.)
  → IDEA-006.
- **HerO 2** — Yoon et al., FEVER-8 2025, arXiv:2507.11004. Per-document **summarization**,
  answer reformulation, and HyDE-style query expansion. → IDEA-005, IDEA-007.
- **CTU AIC** — Ullrich & Drchal, FEVER-8 2025 (AVeriTeC-2025 winner), arXiv:2508.04390.
  **Per-label Likert confidence** (4 independent 1–5 scores → argmax) and MMR diversity rerank
  (λ=0.75). → IDEA-010.

### Verdict scale & evaluation
- **AVeriTeC** — Schlichtkrull et al., NeurIPS 2023 D&B, arXiv:2305.13117. Our **4-class label set**
  (Supported / Refuted / Not Enough Evidence / Conflicting-Cherrypicking) and primary benchmark.
- **FEVER / AVeriTeC shared tasks** — arXiv:2410.23850 and the FEVER-8 2025 overview. Validate the
  **retrieve → reason → verdict** decomposition and in-context (not dedicated-NLI) verdicts.

### Claim extraction & normalization
- **Molecular Facts** — Gunjal & Durrett, Findings of EMNLP 2024, arXiv:2406.20079. The target is
  **decontextualized + minimal** ("molecular") facts, not maximally shredded atoms — justifies our
  single-primary-claim output.
- **ClaimNorm** — Sundriyal et al., Findings of EMNLP 2023, arXiv:2310.14338. Extraction target =
  standalone + verifiable + entity-resolved, without inventing facts.
- **Decomposition Dilemmas** — Hu et al., NAACL 2025, arXiv:2411.02400. Over-decomposition is an
  empirically documented failure mode — reinforces "don't over-fragment."
- **AFEV (Fact in Fragments)** — Zheng et al., arXiv:2506.07446 (2025). Feedback-driven, iterative
  granularity — basis for IDEA-013 (sequential evidence reading with a running accumulator).
- **CheckThat! 2025** — Alam/Struß et al., arXiv:2503.14828. Validates the staged
  **detect → normalize → verify** architecture (normalization is its own task).

### Harm & prioritization
- **FABLE** — Sehat et al., CSCW 2024, arXiv:2312.11678. The 5-dimension harm rubric
  (Fragmentation, Actionability, Believability, Likelihood, Exploitativeness), Likert 1–5 each,
  summed 5–25. Run as an orthogonal check to the 4-prong misinformation criterion.
- **Guriev et al.** (dynamic-nudges / misinformation model). The theoretical model our project is
  the empirical arm of; shapes the 4-prong scope gate (factual / verifiable / public-consequence /
  news-substitutable).

### Detection (Tier 0) & prompt design — supporting evidence
- **Setty**, SIGIR 2024 industry, arXiv:2402.12147 — fine-tuned encoders beat prompted LLMs for
  detection (basis for the Tier-0 encoder ensemble).
- `src/docs/extraction_prompt_sota.md` collects **50+** prompt-engineering citations behind specific
  extraction-prompt decisions (format sensitivity, reason-before-answer, few-shot range, CoT
  trade-offs, VLM hallucination, structured outputs).

**Key design pivots → anchor paper**

| Decision | Anchor |
|---|---|
| Tier-3 loop: plan→search→summarize→synthesize→decide | ClaimCheck (2510.01226) |
| Confidence-gated re-query (`next_query`) | FIRE (2411.00784) |
| Per-document summarization + HyDE expansion | HerO 2 (2507.11004) |
| Per-label Likert confidence + MMR rerank | CTU AIC (2508.04390) |
| Single decontextualized primary claim | Molecular Facts / ClaimNorm / Decomposition Dilemmas |
| 4-class verdict + justification | AVeriTeC (2305.13117) |
| Harm prioritization (5 dims) | FABLE (2312.11678) |
| Encoder ensemble for detection | Setty (2402.12147) |

---

## Part 2 — 2025–2026 field scan

### Claim extraction / check-worthiness
- **Claimify** — Metropolitansky & Larson (Microsoft), **ACL 2025**, arXiv:2502.10855. *Method SOTA.*
  Selection → Disambiguation (abstain on ambiguity) → Decomposition; 99% of claims entailed by
  source. Dataset: `microsoft/claimify-dataset`. (We already clone this approach — this is the
  canonical reference + dataset.)
- **FEVERFact** — Ullrich, Mlynář, Drchal, arXiv:2502.04955 (2025). *Benchmark/metric SOTA.* 17K
  atomic claims; key finding: atomicity/faithfulness are **saturated** — the real discriminators are
  **Focus & Coverage** (did you extract the *right* claims, and *all* of them).
- **CheckThat! 2025 Task 2** (claim normalization, 20+ languages) — arXiv:2503.14828; strong system
  notes: UNH (2509.06883, fine-tune vs prompt; METEOR under-credits good extractions), AKCIT-FN
  (2509.11496, resource-aware routing), DS@GT (2508.17402, retrieval-first).
- **CheckMate / CheckIt** — Sundriyal et al., arXiv:2309.09274. Fine-grained, Twitter-native
  check-worthiness with *reasons* (factuality, public impact, harm).
- **When Hate Meets Facts** — Ocampo et al., 2026, arXiv:2603.25269. Check-worthiness + harm are
  mutually reinforcing — supports a joint scope+harm gate.
- **AFaCTA / PoliClaim** — Ni et al., ACL 2024, arXiv:2402.11073. LLM-assisted annotation recipe
  (useful for bootstrapping our pending gold set).

### Evidence retrieval
- **FIRE** (2411.00784) and **ClaimCheck** (2510.01226) remain the closest blueprints (already in
  our design).
- **Veri-R1** — He et al., arXiv:2510.01932 (2025). *SOTA RL.* Trains the LLM↔search-engine
  interaction with online RL (+up to 30% joint accuracy). Backbone: **Search-R1** (2503.09516).
- **RAV (Recon-Answer-Verify)** — EMNLP 2025 Industry, arXiv:2507.03671. Iterative
  sub-question → query → answer; ships PolitiFact-Only with leakage stripped (mirrors our
  `judged_axis` concern).
- **EASE** — Wei et al., arXiv:2510.11277 (2025). Explicit **evidence-scarcity gating** —
  evidence-based → reasoning-based → sentiment fallback; ships RealTimeNews-25.
- **CONFACT** — Ge et al., IJCAI 2025. **Conflicting-evidence** resolution via source-credibility
  signals injected into retrieval + generation.
- **DeReC** — arXiv:2511.04643. Dense (FAISS) non-LLM retriever as a cheap first-stage.
- **Ev2R** — Akhtar, Schlichtkrull, Vlachos, arXiv:2411.05375 (TACL). Metric to **score the evidence
  stage itself** (adequacy, not just label) — the official AVeriTeC-2025 scorer (recall ≥ 0.44 gate).

### Fact verification / veracity + confidence
- **ClaimCheck** (2510.01226) — small-model SOTA; closest analog to our verifier.
- **Reasoning-CV** — Zheng & Lee, arXiv:2505.12348 (2025). Whole-claim "CoT-Verify" beats
  decompose-then-verify; an 8B reasoner rivals GPT-4o+CoT.
- **DebateCV** — He et al., **WWW 2026**, arXiv:2507.19090. Two debaters + moderator; explicitly
  fights NEI over-selection and improves justification quality.
- **MERMAID** — Cao et al., arXiv:2601.22361 (2026). Multi-agent retrieval+reasoning with persistent
  evidence memory across related claims.
- **PCC** — Wang et al., arXiv:2601.02574 (2026). *Most relevant confidence work.* Verbalized
  confidence is poorly calibrated; use probabilistic certainty + reasoning consistency instead — and
  it routes search depth by certainty.
- **(Fact) Check Your Bias** — Bakke & Heggelund, arXiv:2506.21745 (2025). The HerO/LLaMA verifier
  labels ~half of claims "NEI"; cautions on NEI over-prediction and prompt-induced evidence bias.

### Multimodal (relevant to our ~2000 post images)
- **DEFAME** — Braun et al., **ICML 2025**, arXiv:2412.10510 (repo: `multimodal-ai-lab/DEFAME`,
  Apache-2.0; **`DEFAME+SOCIAL` branch**). *SOTA multimodal*, 70.5% AVeriTeC; emits a structured
  fact-check **report**. Closest open social-media-tuned multimodal checker.
- Benchmarks: **AVerImaTeC** (2505.17978), **MMM-Fact** (2510.25120, 125k claims w/ retrieval
  difficulty), **VeriTaS** (2601.08611, dynamic), **TSVer** (EMNLP 2025, 2511.01101, time-series),
  **RealFactBench** (2506.12538, Unknown-Rate metric), **DeepFact** (2603.05912, audit-then-score).

### Surveys (good anchors for the boss)
- **Claim Verification in the Age of LLMs** — Dmonte et al., arXiv:2408.14317 (2024, rev. Feb 2025).
  Best single entry point for the verifier-stage taxonomy.
- **Hallucination to Truth** — Rahman et al., *AI Review* 2026, arXiv:2508.03860. 2020–2025 review of
  factuality evaluation; good "why verbal metrics are unreliable" framing.

### SOTA tools / systems
| Tool | What / why | Status |
|---|---|---|
| **ClaimCheck** (idir.uta.edu/claimcheck) | small-LLM agentic verifier; our reference | active 2025 |
| **DEFAME + DEFAME+SOCIAL** (Apache-2.0) | multimodal, structured report | active 2025 |
| **VeriScore** (Apache-2.0, HF weights) | reusable **verifiable-only** claim extractor | usable |
| **Loki / OpenFactVerification** (MIT) | 5-stage skeleton (decompose→checkworthy→query→evidence→verify) | code stale (Apr'24) |
| **HerO 2** (open-weights) | reproducible retrieve→summarize→QA→verdict, HyDE | active 2025 |
| **CTU AIC** | AVeriTeC-2025 winner reference impl | reference |
| **SAFE** (DeepMind) | canonical agentic per-claim search loop, F1@K | reference |
| **Google Fact Check Tools API** | "already debunked?" lookup (we use this in `claimreview.py`) | live |
| **Factiverse / Full Fact AI / ClaimBuster** | callable check-worthiness / claim-detection / prior-debunk APIs | commercial/live |
| **X Community Notes + RLCF** (2506.24118; field eval 2604.02592) | "AI drafts, humans rate" + bridging — relevant to our nudging-tool framing | live pilot |

---

## Part 3 — What's new since we designed (candidate follow-ups)

These weren't in the original design and are worth a look (not yet acted on):

1. **Evaluate the retriever, not just the label** — adopt **Ev2R** (2411.05375) to score evidence
   adequacy; pair with AVeriTeC's "label *and* evidence adequate" bar.
2. **Measure extraction by Focus & Coverage** (FEVERFact, 2502.04955) rather than overlap metrics —
   directly tests our single-primary-claim policy.
3. **Validate our Likert confidence** — **PCC** (2601.02574) and the bias paper (2506.21745) both
   say verbalized confidence is miscalibrated and NEI is over-predicted; worth a calibration check.
4. **Conflicting-evidence handling** — **CONFACT** (IJCAI 2025) argues for source-credibility
   weighting in ranking + verdict; relevant to our "Conflicting-Cherrypicking" class.
5. **Evidence-sufficiency gate** — **EASE** (2510.11277) for an explicit "evidence not good enough →
   fall back" stage before committing a verdict.
6. **Multimodal verifier** — **DEFAME+SOCIAL** is the natural fit for the image harvester.
7. **Reusable extractor** — **VeriScore** (Apache-2.0, HF weights) as a verifiable-only complement
   to / check against our Claimify clone.
8. **Learned search policy** (longer-term) — Search-R1 → **Veri-R1** if hand-prompted query/stop
   becomes the bottleneck.
