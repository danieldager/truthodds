# Archived 2026-07-13 — claim-level Tier-3 verify loop (v0.3, "ClaimCheck-faithful")

The pre-post_verify verification stack, superseded by `pipeline/post_verify.py` (post-level
accumulating ledger loop, LOOP_VERSION v4). Archived after the 2026-07-13 course-correct
(too many stale versions cohabitating — see `docs/system_snapshot_2026-07-13.md`).

- `verify.py` + `verify_prompts.py` — the ClaimCheck-faithful loop (Serper→snippet-always→
  confidence-gated synthesis→4-class + Likert). Its **flag-acc 0.915 on the clean 400-claim
  fc-gold dev split** is THIS loop's baseline number; post_verify has no equivalent yet.
- `verify_text.py` — the raw-text "Option-A" controller probe; post_verify was built from it.
  Its one live function (`_redundant_query`) moved into `pipeline/post_verify.py`.
- `extract.py` — never-implemented pipeline extraction stub (raise NotImplementedError);
  the real extractor lives in `eval/scripts/claim_sourcing/`.
- `verify_smoke.py`, `verify_run.py`, `verify_text_run.py`, `replay_synth.py`,
  `rescore_likert.py` — smoke/eval runners for the archived loop.

Kept LIVE in `eval/scripts/verification_grading/`: the fc-gold/AVeriTeC loaders, scorers, gold
data and audits — the eval roadmap (WS1) reuses them for the NEW loop's adapter.
