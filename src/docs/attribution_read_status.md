# Attribution reader (READ_SYS_ATTRIB) — status + defect log

**2026-08-25.** Daniel's ruling: fc-gold failure does NOT kill this thread —
"if the attribution reader prompt is operating at least somewhat decently, we
should try to keep pushing on, log what's wrong with it, and come back to it
later." The reader is NOT used in production or the TH audit (v5 everywhere:
the audit corpus must stay read-prompt-homogeneous with the 1,055 already-
scored pool claims or the s7 banding mixes instruments).

## Where it stands

- Prompt: `read_v5_prompts.py::READ_SYS_ATTRIB` (v3). Separate prompt routed by
  claim mode, NOT a rule injected into v5 (v5.1–5.3 all inverted a stratum).
- Rulings encoded (Daniel 2026-08-25): utterance is the unit; a document must
  VOUCH for the utterance itself (own reporting/transcript/journalistic voice);
  an FC of the attribution scores by its FINDING, never its restatement;
  absence is never refutation (threshold's job); exactness = same words, same
  occasion.
- fc-gold gate result (E1, 737 attribution claims, offline re-read of saved
  docs): stratum AUC .800 vs v5 baseline .824. Verdict: fc-gold structurally
  rewards v5's wrong-reason refutes (silence/adjacency/content harvest — 137 F
  docs, 91% non-FC) and contains content-graded rows tagged attribution, so
  this gate does not measure what we want. Raw reads:
  `eval/data/urn_runs/e1_ctx/attrib_reread/run_v3.jsonl`.

## Defect log (open)

1. **FC-finding rule misses.** Some fact-check pages still read as 5 when
   their finding is fabricated/distorted (Greta Thunberg: snopes + AFP 1→5).
   The restatement-vs-finding distinction fails when the page's structure is
   claim-first with the finding buried.
2. **Negation polarity.** "X did NOT say Y" claims invert: a report that X
   said it must REFUTE, but the reporting-supports rule fires (Modi
   'infiltrators' 1→5; quote-of-a-denial claims are the same family).
3. **Adjacent-event matching residue.** Thematically-adjacent real events
   still occasionally read 4/5 (Kane armband; a different Kirk quote).
4. **True-support cost of exactness.** v3's tightened exactness drops genuine
   support (55 T docs sup→neu, e.g. paraphrases across venues). The 4-band
   ("same point, other words, same occasion") may be too narrow.

## Axis audit + clean-frame regate (2026-08-25, later)

- Axis audit of the 281 gold-FALSE attribution rows (flash over claim +
  original_rating + enrich_note; NOISY — no FC body on disk):
  **utterance 159 / content 112 / unclear 10** → ~40% of the stratum's FALSE
  gold is content-graded (FC never disputed the utterance). File:
  `attrib_reread/fc_axis_false.parquet`.
- Regate on the axis-clean frame (FALSE limited to utterance-graded):
  v5 AUC .840 / rec .314 vs v3 .824 / .229 — **v5 still wins on clean gold**,
  so frame contamination is real but not the whole story. Loss decomposition
  on clean-FALSE refutes: FC docs retained 16/26 (finding rule fails on
  classic debunks — snopes fabricated-quote pages going 1→X when the finding
  isn't cleanly located); news docs retained 78/156 (part legitimate
  absence-honesty, part genuine contradictions lost).
- v4 (targeted at defects 1–2: finding-outranks-everything + claim-polarity
  rule for "did NOT say" claims): targeted smoke, all 159 utterance-graded
  FALSE + 25 trues. Refute retention on utterance-FALSE vs v5's 26 FC / 156
  news: v3 16/78 → **v4 19/87**; TRUE support kept 76/79 (v4's polarity rule
  did not reopen the true-side losses). 17 F docs still newly-supportive.
- **Ceiling insight:** v4 retains ~56% of v5's refute mass on genuine
  fabricated-quote claims. The remaining ~44% is news docs whose v5 "refute"
  was silence/adjacency — unrecoverable by honest utterance reading. The gap
  to v5 on fc-gold therefore cannot be closed at the read step; it must come
  back at scoring (silence-leans-refute for attribution) or at synthesis.
  Full v4 regate deferred to the CN frame (hand labels).

## Next steps (when resumed — CN dataset era)
- Fix defects 1–2 (finding-first rule; explicit negation-polarity rule).
- Real eval frame: hand-labelled attribution claims (TH5 Friday hand-audit
  seeds this: the review HTML carries an attribution-form badge per claim).
- Scoring side: silence-leans-refute for attribution via stratum threshold or
  synthesis-level judgement (synthesis already weighs axis jointly).
