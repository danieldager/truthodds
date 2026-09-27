# Current System Map — Extraction + Verification (as implemented)

**Purpose.** A detailed, faithful map of the claim-**extraction** and claim-**verification**
steps exactly as they exist in the code today, to compare against two parallel
SOTA literature reviews (one on claim extraction, one on claim verification).
This is **documentation only** — every model, prompt, and parameter below is
traced to a real `file:line`; prompts are quoted, not paraphrased. No
assessment, no recommendations, no redesign.

Repo root for all paths below: `src/` (i.e. `pipeline/...`, `eval/...`,
`config.py` resolve from `src/`).

---

## 0. Two layers, and which one actually runs

The codebase contains **two parallel implementations** of the extraction half:

| Layer | Location | Status | Role |
|---|---|---|---|
| **Production pipeline** | `pipeline/` | **partly STUBBED** | The intended deployable tiered pipeline. `extract.py`, `cache.py` (Tier 1), `fact_api.py` (Tier 2) raise `NotImplementedError`. `verify.py` (Tier 3) is **built**. |
| **Eval / research cascade** | `eval/scripts/claim_cascade/` + `eval/scripts/extraction_grading/` | **BUILT, this is what runs** | The extraction that actually executes today: a 7-stage resumable cascade over the 862-post X capture. Stages 3 and 4 *reuse* `extraction_grading/extract.py` and `judge.py`. |

So: **extraction-as-run = the cascade** (Part A). **Verification-as-built =
`pipeline/verify.py` Tier 3** (Part B), exercised by the
`verification_grading/` harness against AVeriTeC. The production orchestrator
`pipeline/pipeline.py` wires Tier 0→1→2→3 but cannot run end-to-end because
`extract.extract` is a stub.

### Model assignments (the two layers do not fully agree)

| Where | Constant / default | Model | `file:line` |
|---|---|---|---|
| Production extraction | `EXTRACTION_MODEL` | `qwen/qwen3-32b` | `config.py:15` |
| Production verification | `VERIFICATION_MODEL` | `openai/gpt-oss-120b` | `config.py:19` |
| Cascade Stage 2 scope | `DEFAULT_MODEL` | `openai/gpt-oss-120b` | `eval/scripts/claim_cascade/stage2_scope_llm.py:52` |
| Cascade Stage 3 extract | `DEFAULT_MODEL` | `openai/gpt-oss-120b` | `eval/scripts/extraction_grading/extract.py:46` |
| Cascade Stage 4 judge | `DEFAULT_JUDGE_MODEL` | `qwen/qwen3-32b` | `eval/scripts/extraction_grading/judge.py:55` |
| Cascade Stage 5 FABLE | `DEFAULT_MODEL` | `qwen/qwen3-32b` | `eval/scripts/claim_cascade/stage5_fable.py:60` |
| Embedding (shared) | `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` (dim 384) | `config.py:29-30` |

All LLM calls go to **Groq** (`https://api.groq.com/openai/v1`) via the OpenAI
client. Both `gpt-oss-120b` and `qwen3-32b` are reasoning models that may emit
`<think>...</think>` blocks; every parser strips those before JSON-decoding
(`extraction_grading/prompts.py:173-182`, reused by `prompts_scope.py`).

**Note (cascade vs production extractor model):** the cascade's live extractor
(Stage 3) defaults to `gpt-oss-120b`, whereas production `EXTRACTION_MODEL` is
`qwen3-32b`. They are not the same model.

### Canonical run referenced throughout

Cascade first full run, **2026-06-01**, over `fourcat/exports/x_collection_010626.ndjson`
(862 X posts). Funnel from `eval/scripts/claim_cascade/results/funnel_report.md`:

| Stage | Unit | Count | % of unit | % of prev |
|---|---|---:|---:|---:|
| Captured posts | posts | 862 | 100.0% | 100.0% |
| In scope (LLM, Stage 2) | posts | 208 | 24.1% | 24.1% |
| Has ≥1 claim (Stage 3) | posts | 163 | 18.9% | 78.4% |
| Extracted claims | claims | 410 | 100.0% | 251.5% |
| Check-worthy (4-prong OR FABLE) | claims | 391 | 95.4% | 95.4% |
| FCT-matched (Tier-2 short-circuit) | claims | 14 | 3.6% of check-worthy | — |

`in_scope_embed = 378` (43.9%) is a **parallel measurement, not a gate**.
4-prong vs FABLE agreement = 18.3% (75/410): both 56, 4-prong-only 332,
FABLE-only 3, neither 19. (The project CLAUDE.md headline "has-claim 19% cascade
vs 43% flat-pass" = 163/862 here vs extraction on all 862 without the scope gate.)

---

# Part A — EXTRACTION (as actually run via the cascade)

```
Stage 0  build posts        posts_x862.parquet          (862, no API)
Stage 1  topic embed        topics_x862.parquet         (862, local MiniLM)   ┐ detection
Stage 2  LLM scope          scope_x862.parquet          (862, gpt-oss-120b)   ┘ (Stage 2 = real gate)
         └─ gate            posts_inscope.parquet       (in_scope_llm rows)
Stage 3  extract claims     stage3_extractions.parquet  (gated, gpt-oss-120b)   decomposition (+ has_claim detection)
Stage 4  per-claim judge    stage4_judgments.parquet    (claims, qwen3-32b)     harm-prioritization (misinfo_candidate) + extraction-eval (3 Likerts)
Stage 5  FABLE harm score   fable_x862.parquet          (claims, qwen3-32b)     harm-prioritization
Stage 7  FCT cross-check    fct_x862.parquet            (check-worthy union)    harm-prioritization (Tier-2 short-circuit)
Stage 6  join               graded_posts_x862 + graded_claims_x862              extraction-evaluation
         analyze            results/funnel_report.md
```

Every stage writes a joinable, resumable parquet keyed on `post_id` (claims
also on `claim_index`), so analysis never re-runs the LLM. Source of the diagram:
`eval/scripts/claim_cascade/README.md:8-19`.

---

## A.0 Stage 0 — Post construction (no model)

- **Purpose.** Parse the Zeeschuimer/4CAT X capture (NDJSON of full X GraphQL
  tweet objects) into a flat per-post parquet with the author's own text,
  quoted text, and link-card snippet kept separate.
- **File.** `eval/scripts/claim_cascade/stage0_build_posts.py`; record parser
  `parse_record` at `:110-153`.
- **Model / params.** None (pure parsing).
- **Technique.**
  - `post_id = sha1("x:" + rest_id)[:16]` (`:76-78`), mirroring
    `extraction_grading/load_posts.py`.
  - `own_text` prefers the expanded note-tweet (`note_tweet...result.text`) over
    truncated `legacy.full_text` (`_own_text` `:89-94`, `_note_text` `:81-86`).
  - Reader-visible `text` composition (`:119-123`):
    `own_text [+ "\n[QUOTED] " + quoted_text] [+ "\n[LINK] " + card_snippet]`.
  - `created_at` parsed with Twitter format `"%a %b %d %H:%M:%S %z %Y"` (`:50`).
- **Output schema** (`OUTPUT_SCHEMA` `:52-73`, one row/post): `post_id, rest_id,
  text, own_text, quoted_text, card_snippet, lang, created_at, favorite_count,
  retweet_count, reply_count, quote_count, bookmark_count, views_count,
  author_handle, author_followers, author_verified, has_community_note,
  is_quote_status, char_len`.

---

## A.1 Detection — "is there a public-consequence checkable claim here?"

Two layers run on **all 862** posts: a local embedding classifier (Stage 1,
logged but **not** a gate) and an LLM scope gate (Stage 2, the **real** gate).
A third detection signal — the extractor's `has_claim` boolean — is produced in
Stage 3 (see A.2).

### A.1.1 Stage 1 — Embedding topic / scope classifier (local MiniLM) — parallel measurement, NOT a gate

- **Purpose.** Cheap, recall-favoring topic classification by cosine similarity
  to per-topic anchor vectors. Logged for embedding↔LLM agreement analysis; the
  LLM stage is the actual gate.
- **Files.** `eval/scripts/claim_cascade/stage1_topic_embed.py`
  (`classify` `:60-95`); anchors in `eval/scripts/claim_cascade/anchors.py`.
- **Model.** `paraphrase-multilingual-MiniLM-L12-v2` via the shared singleton
  `pipeline.embedding.embed` (`stage1:36`, `embedding.py:17-23`). L2-normalized,
  so cosine == dot.
- **Technique.**
  - Each topic LABEL's anchor = L2-normalized mean of the normalized embeddings
    of a handful of short multilingual descriptor phrases (EN/FR/ES/AR), built
    by `anchors.build_anchor_matrix` (`anchors.py:195-209`).
  - Score = `matrix @ vec`; `top_topic = argmax`; `in_scope_embed = (top_topic ∈ IN_SCOPE) and (top_cosine ≥ TAU)` (`stage1:81-90`).
  - **`TAU = 0.0`** (`anchors.py:31`) — starts recall-favoring. **`MIN_CHARS = 5`**
    (`anchors.py:34`): posts shorter than this get `top_topic="empty"`,
    `in_scope_embed=False`, zero vectors (`stage1:68-78`).
  - Persists the full 384-d post embedding + full score vector for reuse.
- **Taxonomy** (`anchors.py:42-187`). 9 IN_SCOPE topics: `politics_elections,
  health_medicine, science_climate, economics, crime, conflict_war,
  crisis_disaster, public_figure, identity_public`. 5 OUT_SCOPE:
  `sports, entertainment_celebrity, personal_lifestyle, promotional_ads,
  social_format`. (These slugs are mirrored verbatim into the LLM scope prompt —
  see A.1.2.)
- **Output schema** (`stage1:47-57`): `post_id, top_topic, top_cosine,
  in_scope_embed, margin, topic_scores (list), topic_labels (list),
  post_embedding (list[f32], 384)`.

### A.1.2 Stage 2 — LLM scope gate (gpt-oss-120b) — the real gate

- **Purpose.** Post-level binary scope decision: does the post plausibly carry a
  factual assertion in a **public-consequence** domain that could **substitute
  for news**, and is therefore worth passing to extraction? Runs on **all 862**
  (not just embedding-positives) so embedding↔LLM agreement is measurable.
- **Files.** Runner `eval/scripts/claim_cascade/stage2_scope_llm.py`; prompt
  `eval/scripts/claim_cascade/prompts_scope.py` (`SCOPE_CONFIRM_SYSTEM` `:101-119`).
- **Model & params.** `openai/gpt-oss-120b` (`:52`), `temperature = 0.1` (`:54`),
  `max_tokens = 4000` (`:55`, reasoning headroom), `MAX_RETRIES = 5` with
  exponential backoff on 429 / timeout / connection / 5xx (`_retryable` `:82-91`,
  `call_llm` `:94-108`), 12 workers.
- **Technique — prong reuse.** The scope prompt embeds, **verbatim**, prongs 3
  (PUBLIC-CONSEQUENCE DOMAIN) + 4 (NEWS-SUBSTITUTABLE) of the shared
  `_MISINFO_CRITERION`, sliced out at import time by a regex
  (`prompts_scope.py:46-56`). If the upstream criterion text is restructured the
  slice fails loudly. This guarantees the cheap gate and the downstream judge
  share one definition of "matters for public discourse" — so gate↔judge
  disagreement is meaningful, not a definitional artefact.
- **Exact prompt** (`SCOPE_CONFIRM_SYSTEM`, `prompts_scope.py:101-119`; the
  `{SCOPE_PRONGS}` placeholder interpolates prongs 3+4 quoted in A.2.0):

  ```
  You are a scope filter for a misinformation fact-checking pipeline. Given a single social media post, decide whether the post is IN SCOPE — i.e. whether it plausibly carries a factual assertion in a PUBLIC-CONSEQUENCE domain that could substitute for news, and is therefore worth passing on to claim extraction.

  You are NOT extracting individual claims, you are NOT judging whether anything is true, and you are NOT scoring harm. You only decide whether this post is in the domain where misinformation matters.

  Apply these two criteria — the public-consequence and news-substitutable prongs of the project's misinformation-candidate definition:

  {SCOPE_PRONGS}

  DECISION:
  - in_scope = true  if the post plausibly contains at least one factual assertion meeting BOTH criteria above (a public-consequence domain AND news-substitutable). When genuinely borderline, lean TRUE — a later extraction + per-claim judge is the precise gate; this step only removes posts that are clearly out of domain.
  - in_scope = false if the post is wholly personal, promotional/advertising, entertainment/sports, or a social-media-specific format (boost/prayer/mutual-aid request, now-playing auto-share, lyric quote) with no public-consequence factual content.

  TOPIC: classify the post's dominant topic with exactly ONE label from this list:
  {_TOPIC_GUIDE}

  Output strictly valid JSON matching this schema, and NOTHING else:
  {"in_scope": <true|false>, "topic": "<label>", "reason": "<one short sentence>"}

  Do not wrap the JSON in markdown fences. Do not add commentary.
  ```

  `_TOPIC_GUIDE` (`:89-94`) lists the IN/OUT topic slugs (mirroring `anchors`)
  plus `"empty"` (media-only) and `"other"` (nothing fits). User message:
  `f"Post:\n{post}"` (`build_scope_messages` `:122-127`).
- **Parsing.** `parse_scope` (`:273-287`) requires keys `in_scope, topic,
  reason`; coerces `in_scope` to bool; `topic` logged verbatim (off-taxonomy
  label is not an error).
- **Output schema** (`stage2:71-79`): `post_id, in_scope_llm, topic_llm, reason,
  raw_response, latency_s, error`. Errored rows default `in_scope_llm=False`
  (excluded from the gate) and are retried on re-run.
- **The gate.** `write_gate` (`:150-159`) re-derives `posts_inscope.parquet`
  (`post_id, text` where `in_scope_llm`) from the **full** scope table on every
  run, so it always matches the latest scope decisions. This is the Stage-3 input.

---

## A.2 Decomposition — post → atomic claims

### A.2.0 The shared misinformation-candidate criterion (`_MISINFO_CRITERION`)

A single criterion string is embedded **verbatim** into both the extractor and
the judge (and prongs 3+4 sliced into the scope gate), so all three use an
identical definition. `extraction_grading/prompts.py:34-70`:

```
A claim is a POSSIBLE-MISINFORMATION CANDIDATE if and only if ALL FOUR criteria hold:

1. FACTUAL ASSERTION — a specific past or present claim about the world (event, statistic, attribution, causal claim, state of affairs).
   NOT: opinions, value judgments, predictions, hypotheticals, questions, wishes, hopes, feelings, instructions, requests.

2. VERIFIABLE IN PRINCIPLE — can be checked against external evidence (expert consensus, public records, scientific findings, news archives).
   NOT: vague generalizations that can't be tested ("rights are being taken away", "workplace is changing", "mosquitoes are wild"), pure subjective generalizations ("X is the best"), abstract sociological musings ("positive externalities are vanishing").

3. PUBLIC-CONSEQUENCE DOMAIN — politics / elections, public officials, health / medicine, vaccines, science / climate, economics / markets, crime, conflict / war, crisis events (disasters, attacks), identity-group claims that affect public discourse, history of public record.
   NOT: personal experiences, interpersonal anecdotes, promotional content, event logistics, podcast / show plugs, social-media-specific formats (boost requests, prayer requests, mutual aid, now-playing auto-shares).

4. NEWS-SUBSTITUTABLE — could plausibly be shared as informational content that mimics news, not just as expressive / social content. The post does not have to be news-styled, but the claim itself must be the kind that could appear in a news article or fact-check piece.
   NOT: first-person emotional framings even when they contain a kernel of political content (e.g. "I'm distressed her rights are being taken away" — the kernel is too vague and the framing is expressive).

QUOTED CLAIMS: if the post quotes someone else (op-ed, news clip, public figure's statement, retweet) and the quoted content is itself a verifiable claim in a public-consequence domain, TREAT IT AS A MISINFORMATION CANDIDATE. Amplifying false claims by quoting is a misinformation vector — the speaker is effectively asserting by sharing.

EMBEDDED CLAIMS: if a post about a private person contains an embedded claim that itself meets criteria 1-4, extract the embedded claim only ("My coworker said vaccines cause autism" — extract "Vaccines cause autism").
```

(Followed by 6 POSITIVE and 11 NEGATIVE reference examples, `prompts.py:52-70`.)
Prong 3 starts at the line `3. PUBLIC-CONSEQUENCE DOMAIN` and the scope slice
(`SCOPE_PRONGS`) is everything from there up to `QUOTED CLAIMS:` — i.e. prongs
3+4 with their NOT-clauses (`prompts_scope.py:46-56`).

### A.2.1 Stage 3 — Claim extraction (gpt-oss-120b) — reuses `extraction_grading/extract.py`

- **Purpose.** One LLM call per in-scope post → `{has_claim, claims}`. Decomposes
  into the smallest atomic misinformation-candidate statements (one sentence
  each), resolving pronouns using only the post. Produces both the
  **decomposition** (`claims`) and a **detection** signal (`has_claim`).
- **Files.** Runner `eval/scripts/extraction_grading/extract.py`; prompt
  `extraction_grading/prompts.py` (`EXTRACTION_SYSTEM` `:77-97`). Invoked on the
  cascade gate via `-i posts_inscope.parquet -o stage3_extractions.parquet -m
  openai/gpt-oss-120b` (README `:39-43`).
- **Model & params.** `openai/gpt-oss-120b` (`extract.py:46`), `temperature =
  0.1`, `max_tokens = 4000` (`call_llm` `:73-80`), 12 workers, resumable on
  `post_id`.
- **Exact prompt** (`EXTRACTION_SYSTEM`, `prompts.py:77-97`; `{_MISINFO_CRITERION}`
  interpolated, see A.2.0):

  ```
  You are a fact-checking assistant. Given a social media post, extract only atomic claims that are POSSIBLE-MISINFORMATION CANDIDATES — the kind of claims that beg to be fact-checked because misinformation is a documented danger in their domain.

  {_MISINFO_CRITERION}

  For posts that DO contain at least one misinformation-candidate claim:
  - Decompose into the smallest factual statements that assert it. One sentence per claim.
  - Resolve pronouns and references using only information in the post. Do not introduce facts not in the post.
  - Sarcastic or emotional framing does NOT disqualify a claim — what matters is whether the underlying assertion meets all four criteria.
  - Do NOT pre-judge whether a claim is true or false; that is determined later.

  If the post contains no misinformation-candidate claim, return has_claim=false with claims=[].

  Test before extracting: would a professional fact-checker, paid per claim, take this one? If unsure, lean toward has_claim=false. The cost of a false positive (extracting non-misinfo content) is higher than the cost of a false negative (missing one).

  Output strictly valid JSON matching this schema, and NOTHING else:
  {"has_claim": <true|false>, "claims": [<string>, ...]}

  Constraints:
  - If has_claim is false, claims MUST be [].
  - If has_claim is true, claims MUST be a non-empty list, each entry a single clear sentence.
  - Do not wrap the JSON in markdown fences. Do not add commentary.
  ```

  User message: `f"Post:\n{post}"` (`build_messages` `:100-105`).
- **Parsing.** `parse_extraction` (`prompts.py:189-228`): strips think/fences,
  enforces `has_claim` bool + `claims` list[str], strips empties, and enforces
  the cross-constraint (`has_claim=true ⇒ non-empty`, and vice-versa).
- **Output schema** (`extract.py:62-70`, one row/post): `post_id, has_claim,
  n_claims, claims (list[str]), raw_response, latency_s, error`.

---

## A.3 Harm-prioritization — which extracted claims are worth checking?

Two **independent** rubrics run on the **same claim set with the same model**
(`qwen3-32b`) so the "4-prong vs FABLE" comparison isolates the *rubric*, not the
model. Then an FCT cross-check measures Tier-2 short-circuit rate on the union.

### A.3.1 Stage 4 — Per-claim 4-prong judge (qwen3-32b) → `misinfo_candidate`

- **Purpose.** One LLM call per *claim* (post supplied for context) scoring 3
  Likert dims + the boolean `misinfo_candidate`. The bool is the cascade's
  check-worthiness signal here (validated at precision 0.868 per README `:48-51`);
  the 3 Likerts are the extraction-quality eval (covered in A.4).
- **Files.** Runner `eval/scripts/extraction_grading/judge.py`; prompt
  `extraction_grading/prompts.py` (`PER_CLAIM_JUDGE_SYSTEM` `:112-151`). Cascade
  invocation: `-p posts_inscope -e stage3_extractions --per-claim-output
  stage4_judgments -m qwen/qwen3-32b -w 12` (README `:53-59`).
- **Model & params.** `qwen/qwen3-32b` (`judge.py:55`), **`temperature = 0.0`**
  (`_call_llm` `:85-92`), `max_tokens = 4000`, 12 workers. Resumable per
  `(post_id, claim_index)`. Reference-free (no gold shown).
- **Exact prompt** (`PER_CLAIM_JUDGE_SYSTEM`, `prompts.py:112-151`;
  `{_MISINFO_CRITERION}` interpolated for the bool):

  ```
  You are an expert fact-checker evaluating a single normalized factual claim that another system extracted from a social media post. Score the claim on three Likert dimensions (1-5) plus one boolean.

  DIMENSIONS:

  - fidelity (1-5): does the claim faithfully preserve the post's factual content without adding or omitting load-bearing detail?
    - 5: claim matches what the post asserts exactly
    - 1: claim misrepresents or fabricates content not in the post

  - decontextualized (1-5): can the claim be checked standalone — no pronouns, no unresolved references, named entities and dates resolved using only what the post provides?
    - 5: fully standalone, all entities named
    - 1: depends entirely on the post's context to be intelligible

  - verifiability (1-5): is the claim a CHECKABLE factual assertion in principle? This is INDEPENDENT of subject matter — purely about whether the text is verifiable text or not. A claim about personal experience can score verifiability=5 if it is concrete and falsifiable (even though it would not be a misinformation candidate).
    - 5: concrete falsifiable assertion with specific named entities, dates, or quantities.
         Example: "The Mexican President addressed the UN on October 12, 2024."
    - 4: specific factual assertion with some vagueness.
         Example: "Most economists agree tariffs raise consumer prices."
    - 3: factually framed but vague, contested, or generalised.
         Example: "The economy is recovering."
    - 2: largely subjective generalization with a factual surface.
         Examples: "Cats always win", "Mosquitoes are wild this summer".
    - 1: not a checkable claim at all — pure opinion, metaphor, prediction, feeling.
         Examples: "X is the best", "Drake is transforming into a corn cob", "I love this", "Things will get worse".

  MISINFO_CANDIDATE (boolean):

  misinfo_candidate is TRUE if and only if the claim is a possible-misinformation candidate per the criterion below. FALSE otherwise.

  {_MISINFO_CRITERION}

  Reasoning hints:
  - verifiability and misinfo_candidate are RELATED but DISTINCT. A claim can be verifiability=5 (concrete, falsifiable) AND misinfo_candidate=FALSE (e.g. a personal experience with specific numbers like "I scored 8 goals last Sunday" — concrete but not in a public-consequence domain).
  - A claim CANNOT be verifiability=1 AND misinfo_candidate=TRUE (pure opinion / metaphor can't be a misinformation candidate, since it's not a factual assertion).
  - For misinfo_candidate=TRUE you typically expect verifiability>=3.

  You see exactly ONE claim. Do NOT consider whether the post contains other claims; judge this one only.

  Return strict JSON only, with this shape and nothing else:
  {"fidelity": <1-5>, "decontextualized": <1-5>, "verifiability": <1-5>, "misinfo_candidate": <true|false>}
  No prose outside the JSON.
  ```

  User message: `f"POST:\n{post}\n\nCLAIM:\n{claim}"` (`build_per_claim_messages`
  `:154-159`).
- **Parsing.** `parse_per_claim` (`prompts.py:270-288`): three ints clamped to
  range 1–5, `misinfo_candidate` coerced to bool.
- **Output schema** (`judge.py:71-82`, one row/claim): `post_id, claim_index,
  claim_text, fidelity, decontextualized, verifiability, misinfo_candidate,
  raw_response, latency_s, error`. Job list is built only for posts the user
  selected (`posts ⨝ extractions`, `:173-191`); errored rows default scores to
  0 / bool to False.

### A.3.2 Stage 5 — FABLE harm rubric (qwen3-32b)

- **Purpose.** Per-claim **potential-harm / check-worthiness** rubric (independent
  of veracity), five 1–5 Likert dims = the FABLE framework (Sehat et al., CSCW
  2024 / arXiv:2312.11678). Runs on the same claims + same model as Stage 4.
- **Files.** Runner `eval/scripts/claim_cascade/stage5_fable.py`; prompt
  `prompts_scope.py` (`FABLE_SYSTEM` `:140-180`).
- **Model & params.** `qwen/qwen3-32b` (`:60`), `temperature = 0.1` (`:62`),
  `max_tokens = 4000` (`:63`), `MAX_RETRIES = 5` backoff (`:96-122`), 12 workers,
  resumable per `(post_id, claim_index)`.
- **The five dimensions** (high = more harmful if false / more worth checking):
  `fragmentation` (institutional-trust erosion), `actionability` (concrete real-
  world harm via call-to-action / logistics / identifying info), `believability`
  (target audience accepts as true), `spread_likelihood` (reach/virality, kept
  distinct from harm), `exploitativeness` (preys on fear/identity/vulnerable
  audience). Each has explicit 1/3/5 anchor definitions in the prompt.
- **Exact prompt** (`FABLE_SYSTEM`, `prompts_scope.py:140-180`):

  ```
  You are assessing the potential HARM and CHECK-WORTHINESS of a single factual claim that another system extracted from a social media post. Score the claim on five dimensions from 1 (low) to 5 (high). Higher scores mean the claim would be MORE harmful and MORE worth prioritizing for fact-checking.

  These five dimensions are the FABLE misinformation-harm framework (Sehat et al., "Misinformation as a Harm: Structured Approaches for Fact-Checking Prioritization", CSCW 2024): Fragmentation, Actionability, Believability, Likelihood-of-spread, Exploitativeness. Score the potential for HARM, NOT veracity — do not try to verify whether the claim is true.

  You are shown the originating POST for context, then the single CLAIM to score. Score the CLAIM; use the POST only to interpret it. Score each dimension INDEPENDENTLY; do not let one high dimension inflate the others.

  DIMENSIONS (score each 1-5):

  - fragmentation — damage to social cohesion / trust in shared institutions (government, courts, science, journalism, elections) or in a whole community. (Institutional-trust erosion, NOT mere missing context.)
    1 = self-contained claim with no institutional/societal target; believing it changes no one's trust in institutions, science, media, or elections.
    3 = glancingly questions the competence or honesty of one institution, or fits a distrust narrative about a single isolated case ("this one agency botched the response").
    5 = its core message is that a pillar of shared reality is corrupt/rigged/lying, or that a whole community cannot be trusted, slotting into a sweeping "how the world really works" narrative ("the election was systematically stolen", "mainstream science on X is a coordinated hoax").

  - actionability — believing or acting on the claim leads to concrete real-world harm, via an explicit call to action, coordination logistics, or identifying information.
    1 = purely descriptive or opinion; a convinced reader has nothing specific to do and no one to act against.
    3 = loosely encourages a harmful behavior or avoidance without specifics ("don't take the medication", "someone should deal with group X").
    5 = explicit call to act PLUS enabling specifics: coordination details (date/time/place), instructions for a dangerous act, or identifying info (names, addresses, workplaces) that points people at specific targets.

  - believability — how readily the TARGET audience could accept it as true (surface plausibility plus content credibility cues), judged relative to that audience, not to absolute truth. Score the content only; you cannot see the author's follower count.
    1 = implausible on its face or trivially debunked — obvious satire/fantasy, or instantly refuted by a quick search.
    3 = plausible to the target community but carries at least one checkable tell (internal inconsistency, missing source, easily found partial rebuttal) a careful reader could catch.
    5 = highly convincing to its audience: coherent, specific, fits what they already expect, no easily found rebuttal, and/or carries credibility cues (authoritative tone, fabricated citations, imposter framing mimicking a real outlet).

  - spread_likelihood — how far and wide the claim is likely to travel, from its framing (emotional charge, novelty, controversy, shareable format). This is REACH, kept distinct from harm: viral does not mean harmful.
    1 = niche or dull; no emotional hook, awkward to share, unlikely to leave a small audience.
    3 = circulates within one community or interest group on a mild hook (novelty or moderate emotion) but lacks the broad relatability to break out.
    5 = built to spread: strong outrage/fear/awe, broadly relatable or timely, punchy shareable format; primed to go viral across communities.

  - exploitativeness — preys on vulnerability: weaponizes fear, identity, or outgroup animus, or targets susceptible audiences (elderly, low-literacy, economically precarious, stigmatized groups); may exploit an active crisis.
    1 = neutral, non-manipulative; no fear/animus, does not single out a vulnerable group.
    3 = some emotional manipulation or group framing, but mild or secondary rather than the engine of the claim.
    5 = engineered to exploit: deliberately inflames fear or hatred of an outgroup, dehumanizes/scapegoats a marginalized group, or preys on an at-risk audience.

  Separating overlapping dimensions:
  - fragmentation vs exploitativeness: fragmentation attacks the trustworthiness of institutions/communities as an ABSTRACT target (erodes shared reality); exploitativeness PREYS ON or DEHUMANIZES a specific vulnerable audience.
  - actionability vs exploitativeness: actionability asks "is there a concrete act or target?"; exploitativeness asks "is a vulnerability or emotion being weaponized?". An incitement claim can legitimately score high on both.

  Output strictly valid JSON matching this schema, and NOTHING else:
  {"fragmentation": <1-5>, "actionability": <1-5>, "believability": <1-5>, "spread_likelihood": <1-5>, "exploitativeness": <1-5>}

  Do not wrap the JSON in markdown fences. Do not add commentary.
  ```

  User message: `f"POST:\n{post}\n\nCLAIM:\n{claim}"` (`build_fable_messages`
  `:183-188`).
- **Bespoke scoring layer (in code, not asked of the model).**
  `fable_total = sum(5 dims)` ∈ [5,25] (`fable_total` `:214-216`);
  `fable_checkworthy = (fable_total >= threshold)`, default
  **`FABLE_CHECKWORTHY_THRESHOLD = 15`** (`:211`, `:219-221`). The bool is
  **re-derived across the whole table on every run** (`stage5:259-261`), so
  recalibration is a re-run with a new `--fable-threshold` (no LLM calls). The
  paper itself uses binary diagnostics and no composite score — the Likert+sum+
  threshold is explicitly **bespoke** (`prompts_scope.py:200-211`). Known
  limitation noted in code: a pure sum misses concentrated high-harm claims
  (e.g. `actionability=5` doxxing with `total<15`).
- **Parsing.** `parse_fable` (`prompts_scope.py:290-311`): rejects bools and
  non-integral floats, enforces ints 1–5 for all five keys.
- **Output schema** (`stage5:79-93`, one row/claim): `post_id, claim_index,
  claim_text, fragmentation, actionability, believability, spread_likelihood,
  exploitativeness, fable_total, fable_checkworthy, raw_response, latency_s,
  error`.

### A.3.3 Stage 7 — Google FCT cross-check (Tier-2 short-circuit measurement)

- **Purpose.** Measure how many **check-worthy** claims already have a published
  fact-check vs. need novel verification. Runs **only** on the **union** of
  claims flagged by EITHER rubric: `misinfo_candidate (Stage 4) OR
  fable_checkworthy (Stage 5)`.
- **File.** `eval/scripts/claim_cascade/stage7_fct.py` (`build_union` `:206-234`).
- **Model / API.** No LLM — Google **Fact Check Tools** `claims:search`
  (`FCTAPI_ENDPOINT` `config.py:26`, key via `X-goog-api-key` header `:88`).
  `pageSize = 5` (`query_fct` `:135`), `timeout = 20s`, 3 retries on 429/5xx
  (`:128-153`), default 6 workers.
- **Technique.**
  - `languageCode` derived from the post's Twitter `lang` (`to_language_code`
    `:118-125`): remap `in→id`, `iw→he` (`_LANG_REMAP` `:68`); omit entirely
    (search all languages) for non-linguistic codes `und/zxx/qme/qst/qht/qam/
    qct/art/""` (`_NON_LANG` `:69`).
  - **Circuit breaker** (`:73-116`): 8 consecutive 429s across threads
    (`_QUOTA_ABORT_AFTER`) trips `_quota_tripped`, aborting remaining work
    (recorded as errors) rather than hammering an exhausted quota. API key kept
    out of all logs/exceptions (`_redact` `:93-94`).
  - `fct_match = True` iff the top returned claim carries ≥1 `claimReview`
    (`parse_fct` `:156-177`); a returned claim with an empty review list is NOT a
    match. Top review's `publisher.site || publisher.name`, `textualRating`,
    `url` are recorded.
  - On write, the table is dedup keep-last and **semi-joined to the current
    union** so it always covers EXACTLY the union (no stale rows) (`_finalize`
    `:331-349`).
- **Output schema** (`FCT_SCHEMA` `:52-64`, one row/check-worthy claim): `post_id,
  claim_index, claim_text, lang, fct_match, fct_publisher, fct_rating, fct_url,
  n_results, latency_s, error`.
- **Note.** This is a *separate, working* FCT client used for measurement; it is
  **not** the production Tier-2 path (`pipeline/fact_api.py`, which is a stub —
  see B.5.2). The two query the same Google API but share no code.

---

## A.4 Extraction-evaluation — measuring the cascade

### A.4.1 The per-claim judge as eval instrument

Stage 4's three Likert dims — `fidelity`, `decontextualized`, `verifiability`
(A.3.1) — are the extraction-quality measurement (reference-free LLM-as-judge,
`qwen3-32b`, different family from the `gpt-oss-120b` extractor). Detection
correctness is inferred from the agreement between the extractor's `has_claim`
and the judge's `misinfo_candidate` (the v3 separate detection judge was removed;
`prompts.py:11-15`). The cleanest precision signal is `extractor=true,
judge(misinfo_candidate)=false`.

### A.4.2 Stage 6 — join into analysis-ready tables

- **File.** `eval/scripts/claim_cascade/stage6_grade.py`, reusing the builders
  in `eval/scripts/extraction_grading/grade.py` (`build_graded_posts`,
  `build_graded_claims`, `_sanity_check`).
- **Inputs.** posts (S0), topics (S1), scope (S2), extractions (S3), judgments
  (S4), fable (S5), fct (S7). All LEFT-joined anchored on the 862-post frame so
  out-of-scope posts are preserved with nulls.
- **`graded_posts_x862`** (one row/post): post fields + topic/scope columns +
  per-post judge aggregates (Likert `*_mean`/`*_min`, `any_misinfo_candidate`,
  `n_misinfo_candidate`; `grade.py:64-97`) + per-post FABLE aggregates
  (`any_fable_checkworthy`, `n_fable_checkworthy`, `fable_total_mean`) + per-post
  FCT aggregates (`any_fct_match`, `n_fct_match`) (`stage6:120-146`). FABLE/FCT
  aggregates are first **semi-joined to the Stage-4 claim keys** so an orphan
  row can't inflate an `any_*` aggregate (`:128-129`).
- **`graded_claims_x862`** (one row/claim): judge Likerts + `misinfo_candidate`,
  FABLE dims + `fable_checkworthy`, FCT match fields, with curated post context
  broadcast onto each claim (`_classical` `:91-117`, `_claim_context` `:80-88`;
  the 384-d embedding is never broadcast).
- **Sanity checks.** unique `post_id` in posts; unique `(post_id, claim_index)`
  in claims (`grade.py:100-117`); `graded_posts.height == posts.height`
  (`stage6:196-199`).

### A.4.3 `analyze_funnel.py` — the funnel & rubric comparison report

- **File.** `eval/scripts/claim_cascade/analyze_funnel.py` → writes
  `results/funnel_report.md`.
- **What it computes** (`build_report` `:63-167`):
  1. **Funnel** as a monotone subset chain (posts: 862 → in_scope_llm →
     has_claim; then unit switches to claims: total → check-worthy
     (`misinfo_candidate OR fable_checkworthy`) ). `in_scope_embed` reported
     alongside as a parallel measurement, excluded from the monotonicity check
     (`:104-109`). FCT shown as a **short-circuit rate**, also excluded from the
     forward-gate chain (matched claims exit, not narrow).
  2. **Embedding↔LLM scope agreement** — 2×2 confusion of `in_scope_embed` vs
     `in_scope_llm` over all 862 (`confusion` `:47-56`).
  3. **4-prong vs FABLE** — 2×2 confusion of `misinfo_candidate` vs
     `fable_checkworthy` over all claims, with example claims per disagreement
     cell.
  4. **Over-decomposition** — posts with `n_claims > 8` (`OVERDECOMP_THRESHOLD`
     `:31`) + worst case.

### A.4.4 The production extractor stub (`pipeline/extract.py`) + `AtomicClaim`

- **`pipeline/extract.py`** — the *production* extractor, **STUBBED**:
  `extract(post) -> list[AtomicClaim]` and `_embed(text)` both raise
  `NotImplementedError` (`:12-29`). TODOs: lift the decomposition `_SYSTEM`
  prompt from `_archive/v0_1/...`, drop the questions step, use `EXTRACTION_MODEL`
  via Groq, embed each claim before returning.
- **`AtomicClaim`** (`pipeline/models.py:7-9`): `{text: str, embedding:
  list[float]}` — the `(text, embedding)` pair is the intended Tier-1 cache key.
  In eval, the verification harness constructs `AtomicClaim` directly from the
  AVeriTeC claim text + a MiniLM embedding (B.5.1), bypassing the stub.

---

# Part B — VERIFICATION (Tier 3, built)

## B.0 Orchestration + data models

### B.0.1 Tier 0→1→2→3 orchestration (`pipeline/pipeline.py`)

- **`run(post, *, risk_score=None, skip_cache=False, skip_fact_api=False,
  date_ceiling=None)`** (`:13-43`): `extract.extract(post)` → if no claims,
  empty result; else default `date_ceiling = today` as **`mm/dd/yyyy`** (`:36-37`)
  and resolve each claim.
- **`_resolve_claim`** (`:46-68`) — the tier ladder, in order:
  - **Tier 1** `cache.lookup(claim)` → return on hit (skipped if `skip_cache`).
  - **Tier 2** `fact_api.lookup(claim)` → on hit, `cache.write(hit)` then return
    (skipped if `skip_fact_api`).
  - **Tier 3** `verify.verify(claim, date_ceiling=...)` → `cache.write(verdict)`
    → return.
- **Tier 0** (encoder ensemble) lives outside this package and is passed in as
  `risk_score` (advisory only).
- **Runnable today?** No end-to-end: `extract.extract`, `cache.lookup/write`,
  `fact_api.lookup` are stubs. Only **Tier 3** (`verify.verify`) is built, and
  it is exercised directly by the eval harness (B.5.1).
- **Date-format note (factual).** `pipeline.run` passes `date_ceiling` as
  `mm/dd/yyyy` (`:37`), but `verify`'s post-filter string-compares it against
  ISO `YYYY-MM-DD` publication dates (B.2.1). The only **live** caller,
  `verify_run.py`, normalizes to ISO `YYYY-MM-DD` (`verify_run.py:22-35`), so the
  mismatch never bites in the eval path; the production path can't reach it
  (extractor stubbed).

### B.0.2 Data models (`pipeline/models.py`)

- **`EvidenceDoc`** (`:12-18`): `url, publication_date: str|None, summary: str,
  relevant: bool=True, source: Literal["scrape","snippet"]="scrape"`.
- **`VerdictScores`** (`:20-25`): per-dimension Likert 1–5 — `veracity,
  evidence_coverage, evidence_consistency, source_quality`.
- **`ClaimVerdict`** (`:28-51`): `claim, scores, tier_resolved∈{1,2,3},
  cap_hit, redundant_exit, evidence_urls, source_publisher, justification`,
  plus the Tier-3 trace: `rounds_used, past_queries, analysis, n_urls_seen,
  n_blocked_or_failed, n_snippet_used, n_irrelevant, elapsed_seconds, llm_calls`,
  plus the parallel 4-class verdict: `verdict_4class, verdict_4class_justification`.
- **`PipelineResult`** (`:54-56`): `post, risk_score, verdicts: list[ClaimVerdict]`.

### B.0.3 Tier-3 tunables (`pipeline/config.py`)

`MAX_ROUNDS = 4` (`:20`), `SEARCH_TOP_K = 3` (`:21`), `REDUNDANCY_THRESHOLD = 0.9`
(`:22`), `SCRAPE_TIMEOUT = 10` (`:23`), `DATE_CEILING_MODE = "today"` (`:24`),
per-domain delay `2.0–3.5s` (`:27-28`), `SCRAPE_BLOCKLIST` = login-wall domains
only (Facebook, Twitter/X, t.co, Instagram, TikTok, LinkedIn, YouTube; newspapers
deliberately NOT blocked) (`:30-42`). Token caps in `verify.py:34-37`: plan 1000,
summarise 2000, synthesise 4000, evaluate 2000; `_DOC_TRUNCATE_CHARS = 12000`
(`:39`).

---

## B.1 Retrieval & query-planning

Entry point `verify.verify(claim, date_ceiling=None, exclude_urls=None,
verbose=False, eval_sees_raw=False)` (`pipeline/verify.py:46-215`). One LLM
(`gpt-oss-120b`) throughout; always returns a verdict.

### B.1.1 Planning — opening query (`_plan_initial_query`)

- **Purpose.** Produce exactly ONE opening Google query from the claim.
- **File.** `verify.py` (`_PLAN_SYSTEM` `:222-230`, `_plan_initial_query`
  `:233-247`).
- **Model & params.** `VERIFICATION_MODEL` (`gpt-oss-120b`), `temperature = 0.2`
  (note: higher than every other verify call), `max_tokens = 1000`.
- **Exact prompt** (`_PLAN_SYSTEM`):

  ```
  You are a fact-checker preparing to verify a claim using web search. Your only job is to produce ONE Google search query that would best surface evidence about the claim.

  Guidelines:
  - Use specific named entities, dates, numbers, and exact phrases from the claim.
  - Prefer terms a journalist or institution would use, not informal phrasing.
  - Aim for the most direct evidence that could either confirm or contradict the claim.
  - Do not use search operators (site:, intitle:, quotes, AND/OR). Plain text only.

  Output ONLY the query, on a single line. No explanation, no JSON, no quotes.
  ```

  User: `f"Claim: {claim_text}\n\nProduce the opening search query."`.
- **Technique.** Robust to reasoning preambles: takes the **last** non-empty,
  de-quoted line; falls back to the raw claim text if empty (`:243-247`).

### B.1.2 Execution — SearXNG search (`search.search`)

- **Purpose.** Run a web search via the self-hosted SearXNG instance; return up
  to `top_k` results.
- **File.** `pipeline/search.py` (`search` `:42-87`), endpoint
  `SEARXNG_ENDPOINT` (default `http://localhost:8888/search`, `config.py:22`).
- **Params.** `{q, format=json, categories=general, language=en}` (`:63-68`),
  `timeout = 15s`. `SEARCH_TOP_K = 3`. No engine-level date filter (date leakage
  control is the verifier's post-filter, B.2.1). `.pdf` URLs dropped (`:75`).
- **Returns** per result `{"url", "snippet" (=SearXNG `content`),
  "date" (=`publishedDate` or None)}` (`:76-80`).
- **Pacing & cache.** Global min interval `_SEARXNG_MIN_INTERVAL = 0.5s` across
  all workers (`:37, :56-61`). Results cached in the `searxng` disk-cache
  namespace keyed on `(query, top_k)` (`:51-53, :83`). On any error returns `[]`
  (`:85-87`).

### B.1.3 Scrape (`search.scrape`) + blocklist + snippet fallback

- **Purpose.** Fetch a URL and extract main article text.
- **File.** `pipeline/search.py` (`scrape` `:90-129`).
- **Technique.** Skip if domain ∈ `SCRAPE_BLOCKLIST` (cache `None`) (`:100-104`);
  per-domain random delay 2.0–3.5s via `_wait_for_domain` (`:132-143`); random
  desktop User-Agent (`:22-27`); `requests.get(timeout=SCRAPE_TIMEOUT=10)`;
  403/404 → cached `None`; Trafilatura `extract(include_comments=False,
  include_tables=True)` (`:118`); `<100` chars after strip → `None` (`:127`).
  Transient errors are NOT cached (future retry); definitive misses ARE cached.

---

## B.2 Evidence-processing

### B.2.1 Summarisation (`_summarise_for_claim`)

- **Purpose.** Per source doc, one LLM call deciding relevance and writing a
  claim-relevant summary of **what the article says** (no outside knowledge), with
  date anchoring.
- **File.** `verify.py` (`_SUMMARISE_SYSTEM` `:254-266`, `_summarise_for_claim`
  `:269-308`).
- **Model & params.** `gpt-oss-120b`, `temperature = 0.1`, `max_tokens = 2000`,
  `response_format = json_object`. Doc text truncated to 12000 chars.
- **Exact prompt** (`_SUMMARISE_SYSTEM`):

  ```
  You are a fact-checker reading ONE source article to assess a specific claim. Decide whether the article speaks to the claim, and if so, write a short summary of WHAT THE ARTICLE SAYS — using only the article's text, never your own background knowledge.

  Output JSON:
  {
    "relevant": true | false,
    "publication_date": "YYYY-MM-DD" or null,
    "summary": "1-3 sentences. Begin with 'According to this article (dated <date>):' if a date is known. State only what the article asserts about the claim. Include numbers, names, dates, direct quotes when present."
  }

  Rules:
  - relevant=false if the article is off-topic, a paywall stub, an error page, a navigation page, or otherwise unhelpful for fact-checking THIS claim.
  - Do NOT take a position on whether the article supports or refutes the claim — only state what the article says.
  - Do NOT cite outside knowledge. The summary must be attributable to the article text alone.
  ```

  User message distinguishes provenance via `body_label`: `"Article text"` for a
  scrape vs `"Google search snippet (article body unavailable)"` for the
  snippet-fallback path (`:277-283`).
- **Caching.** Keyed on `(url, claim_text, source)` in the `summarise` namespace
  (`:272-275, :303`); errors are NOT cached (`:305-308`).
- **Output.** an `EvidenceDoc` (relevance, extracted `publication_date`, summary,
  source).

### B.2.2 Gather (`_gather_evidence`) — scrape → snippet fallback → summarise

- **File.** `verify.py:315-375`.
- **Flow.** (1) parallel `search.scrape` over all results; (2) build summariser
  inputs — prefer full scrape, else fall back to the SearXNG **snippet** if
  `len(snippet) >= _MIN_SNIPPET_CHARS = 40` (`:312, :351-356`), else count as
  blocked/failed; (3) parallel `_summarise_for_claim`. A doc is kept only if
  `relevant AND summary` non-empty; otherwise counted `n_irrelevant`.
- **Counters returned** (`stats`): `n_scraped, n_snippet_used,
  n_blocked_or_failed, n_irrelevant, llm_calls`.
- **Date post-filter & URL exclusion** happen in the caller `_consume`
  (`:77-112`): drop results whose `url ∈ exclude_urls` (eval-time gold
  fact-check article, exact match); then, if `date_ceiling` set, drop docs whose
  extracted `publication_date > date_ceiling` (**string comparison**, lenient on
  undated docs which are kept) (`:96-103`).

---

## B.3 Verdict-prediction

The verdict layer runs **three** LLM calls on the final `analysis`: synthesis
produces the analysis (B.3.1); two evaluators score it in parallel — our Likert
4-dim (B.3.2) and the ClaimCheck-style AVeriTeC 4-class (B.3.3).

### B.3.1 Synthesis (`_synthesise`) — coherent analysis + search-or-evaluate decision

- **Purpose.** Integrate the evidence pool into a 3–6 sentence analysis AND
  decide: emit a verdict-ready analysis (`evaluate`) or request ONE more targeted
  query (`search`). This is the loop's branch point (loop control in B.4).
- **File.** `verify.py` (`_SYNTHESISE_SYSTEM` `:382-406`, `_SYNTHESISE_FORCE_SUFFIX`
  `:408-410`, `_synthesise` `:413-451`, user formatter `:454-469`).
- **Model & params.** `gpt-oss-120b`, `temperature = 0.1`, `max_tokens = 4000`,
  `response_format = json_object`.
- **Exact prompt** (`_SYNTHESISE_SYSTEM`):

  ```
  You are a fact-checker who has gathered evidence about a claim through one or more rounds of web search. Your job is to:

  1. Synthesize the evidence into a coherent analysis: what does it collectively say about the claim? Note agreements, contradictions, gaps, and reliability of sources.
  2. Decide: is the evidence sufficient for a confident assessment, OR is a specific sub-question unresolved AND likely to be answered by one more targeted query?

  Output JSON. Choose ONE shape.

  To proceed to evaluation:
  {
    "action": "evaluate",
    "analysis": "<3-6 sentence synthesis>"
  }

  To request one more search:
  {
    "action": "search",
    "analysis": "<3-6 sentence synthesis>",
    "next_query": "<a search query targeting the specific gap; must NOT paraphrase any prior query>"
  }

  Rules:
  - Choose "search" only if you can name a specific unresolved sub-question AND a query that could plausibly find evidence for it.
  - Choose "evaluate" if evidence is sufficient, OR if further search is unlikely to resolve the gap (e.g., the gap is one of judgement, not retrieval).
  - The "analysis" field MUST be present in both shapes. It is the durable summary the evaluator will read.
  - If the evidence pool is EMPTY (no relevant docs found in any prior round), your next_query MUST take a substantively different angle — different entities, different aspects, different terminology — NOT a paraphrase of any previous query. Light rewordings hit the same Google results and waste the search budget. Try a fiscal angle, a different actor, a broader concept, or a different time period.
  ```

  When the cap is hit, `_SYNTHESISE_FORCE_SUFFIX` is appended (forces
  `action="evaluate"`, "the search budget is exhausted"). User message
  (`_format_synthesis_user`) presents the claim, the numbered evidence pool
  (`_format_evidence` `:672-677`: `[i] url (date: ...) <summary>`), the numbered
  past queries ("do not repeat"), and `Round {round_num} of {max_rounds}`.
- **Decision handling** (`:440-451`). If `force_evaluate` or `action != "search"`
  → `{action: "evaluate", analysis}`; a `search` with an empty `next_query`
  degrades to `evaluate`. On any exception → `{action: "evaluate", analysis:
  "Synthesis failed; treating as no usable evidence."}` (`:436-438`).

### B.3.2 Evaluation — our Likert 4-dimension (`_evaluate`)

- **Purpose.** Assign 4 independent Likert (1–5) scores from the analysis alone.
- **File.** `verify.py` (`_EVALUATE_SYSTEM` `:476-517`, `_evaluate` `:520-561`,
  `_default_eval` `:564-571`).
- **Model & params.** `gpt-oss-120b`, `temperature = 0.1`, `max_tokens = 2000`,
  `response_format = json_object`.
- **Dimensions** (each with explicit 1–5 anchors in the prompt): `veracity`,
  `evidence_coverage`, `evidence_consistency`, `source_quality`. The prompt
  insists on using the full scale ("including 2 and 4 deliberately") and pins
  notable rules — e.g. `evidence_consistency = 3` when only ONE source exists.
- **Exact prompt** (`_EVALUATE_SYSTEM`):

  ```
  You are a fact-checker assigning a structured assessment to a claim, based on a synthesized analysis of evidence prepared by an analyst. The analysis is the only context to use — do not add outside knowledge.

  Score each of four dimensions on a 1-5 Likert scale. Each integer has a specific definition below. Score each dimension independently against its own definition. They are not mutually exclusive.

  **Use the full scale, including 2 and 4 deliberately.** Partial, leaning, or qualified evidence belongs at 2 or 4 — do NOT collapse every judgement to 1, 3, or 5. The middle of the scale is the most informative range for typical fact-checks.

  veracity — does the evidence in the analysis support the claim as true?
    5: Strong evidence the claim is true, including all load-bearing qualifiers (numbers, dates, named entities). Multiple credible sources confirm.
    4: Evidence leans toward the claim. Most sources support the main assertion, but a load-bearing qualifier may be partially off (e.g., approximate number, close-but-not-exact date).
    3: Mixed or indeterminate. Evidence neither clearly supports nor clearly refutes the claim's main assertion.
    2: Evidence leans against the claim. At least one credible source contradicts a load-bearing element.
    1: Strong evidence the claim is false. Multiple credible sources contradict.

  evidence_coverage — does the analysis address the claim's substance?
    5: Multiple sources speak directly to the claim's main assertion AND its load-bearing qualifiers.
    4: Multiple sources directly address the claim, though not all of its qualifiers.
    3: At least one source touches the claim partially (e.g., covers the topic but not the specific qualifier).
    2: Sources are only topically adjacent; they don't speak to the specific assertion.
    1: No retrieved source addresses the claim's substance.

  evidence_consistency — do the credible sources in the analysis agree?
    5: Strong unanimous agreement across THREE OR MORE credible sources, whether supporting OR refuting the claim.
    4: Sources broadly agree; only minor wording or scope differences.
    3: Some disagreement among sources, with one side clearly weaker in credibility. **ALSO use 3 if only ONE source is available** — single-source "agreement" is vacuous and cannot support a higher score.
    2: Notable disagreement, with at least one credible source on each side.
    1: Credible sources actively contradict each other on a load-bearing element.

  source_quality — how reliable are the sources cited in the analysis?
    5: Primary sources (official records, peer-reviewed papers, named experts in their area, court/legislative documents) or top-tier institutional reporting.
    4: Multiple mainstream journalistic outlets; at least some institutional or named-expert voices.
    3: Mixed — some reputable outlets but corroboration is thin or relies on secondary reporting.
    2: Mostly blogs, opinion pieces, or low-traffic sites; few mainstream voices.
    1: Only weak, fringe, or anonymous sources; or no usable sources retrieved.

  Output strict JSON, no prose outside the JSON:
  {
    "veracity": <1-5>,
    "evidence_coverage": <1-5>,
    "evidence_consistency": <1-5>,
    "source_quality": <1-5>,
    "justification": "<one or two sentences citing what in the analysis drives the scores>"
  }
  ```

  User message: claim + `"Synthesized analysis:\n{analysis}"`; the **raw evidence
  pool is appended only if `evidence_pool` is passed** (`:526-533`). It is passed
  only when `eval_sees_raw=True`, which defaults **False** in `verify` (`:184-185`)
  and is not set by `verify_run.py` — so in practice the Likert evaluator sees the
  analysis only.
- **Robustness.** Scores clamped via `_clamp_1_5` (`:697-698`); on parse error or
  malformed scores → `_default_eval` = `{veracity:3, evidence_coverage:1,
  evidence_consistency:1, source_quality:1}` (`:564-571`).

### B.3.3 Evaluation — ClaimCheck-style 4-class (`_evaluate_4class`)

- **Purpose.** Assign one **AVeriTeC 4-class** label from the same analysis, run
  in parallel with the Likert evaluator for direct comparison.
- **File.** `verify.py` (`_EVALUATE_4CLASS_SYSTEM` `:583-611`, `_evaluate_4class`
  `:614-649`). Adapted verbatim from **ClaimCheck** (Devasier et al.,
  arXiv:2510.01226), Listing 4 + verdict descriptions (`:577-578`).
- **Model & params.** `gpt-oss-120b`, `temperature = 0.1`, `max_tokens = 2000`,
  `response_format = json_object`.
- **Labels.** `Supported | Refuted | Conflicting Evidence/Cherrypicking | Not
  Enough Evidence` (`_EVALUATE_4CLASS_LABELS` `:580-581`).
- **Exact prompt** (`_EVALUATE_4CLASS_SYSTEM`):

  ```
  Determine the Claim's veracity from the synthesized analysis, following these steps:

  1. Briefly summarize the key insights from the fact-check in at most one paragraph.
  2. Write one paragraph about which of the Decision Options applies best, and emit the chosen option at the end.

  Decision Options:
  Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence

  Rules:

  Supported - The claim is directly and clearly backed by strong, credible evidence. Minor uncertainty or lack of detail does not disqualify a claim from being Supported if the main point is well-evidenced.
  - Use Supported if the overall weight of evidence points to the claim being true, even if there are minor caveats or not every detail is confirmed.

  Refuted - The claim is contradicted by strong, credible evidence, or is shown to be fabricated, deceptive, or false in its main point.
  - Use Refuted if the central elements of the claim are disproven, even if some minor details are unclear.
  - Lack of any credible sources supporting the claim does NOT mean "Not Enough Evidence" — it means the claim is Refuted.

  Conflicting Evidence/Cherrypicking - Only use this if there are reputable sources that directly and irreconcilably contradict each other about the main point of the claim, and no clear resolution is possible after careful analysis.
  - Do NOT use this for minor disagreements, incomplete evidence, or if most evidence points one way but a few sources disagree.

  Not Enough Evidence - Only use this if there is genuinely no relevant evidence available after a thorough search, AND the claim is too vague or ambiguous to evaluate.
  - Do NOT use this if there is some evidence, even if it is weak, or if the claim is mostly clear but not every detail is confirmed.
  - This is a last-resort option only.

  Output JSON, no prose outside the JSON:
  {
    "verdict": "<one of: Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence>",
    "justification": "<the one-paragraph decision rationale from step 2>"
  }
  ```

  User: claim + analysis + "Assign one of the four decision options."
- **Normalisation.** Off-label outputs are coerced by substring match
  (support→Supported, refut/false→Refuted, conflict/cherry→Conflicting…, else
  Not Enough Evidence) (`:639-648`). On error → `Not Enough Evidence`.

---

## B.4 Loop-control

`verify.verify` (`:46-215`) implements the bounded ClaimCheck loop.

- **Opening round.** `_plan_initial_query` → `past_queries=[q]` (`llm_calls=1`) →
  `search.search(q, 3)` → `_consume` (`:114-124`).
- **Synthesis loop** `for round_idx in range(MAX_ROUNDS)` (`:129-176`), i.e. up to
  4 iterations:
  - `is_last = (round_idx == MAX_ROUNDS-1)`; call `_synthesise(...,
    force_evaluate=is_last)`.
  - `action == "evaluate"` → set `analysis`, **break**.
  - else `next_q`: **if `evidence_pool` non-empty AND
    `_is_redundant(next_q, past_queries)`** → force a `_synthesise(force_evaluate
    =True)`, set `redundant_exit=True`, **break** (`:152-166`). (Redundancy is
    skipped when the pool is empty so the model can keep trying different angles.)
  - else append `next_q`, `search.search(next_q, 3)`, `_consume`, continue.
- **Termination guarantees.** At most **4 queries** total (1 planning + up to 3
  follow-ups; the 4th iteration is forced to evaluate). `assert analysis is not
  None` after the loop (`:177`). `cap_hit = (len(past_queries) == MAX_ROUNDS)`
  (`:180`). `rounds_used = len(past_queries)`.
- **Redundancy check** (`_is_redundant` `:656-665`): cosine similarity (MiniLM
  `embed`) of `next_q` vs each past query; `≥ REDUNDANCY_THRESHOLD = 0.9` → True.
- **Final evaluation** (`:181-189`): Likert and 4-class evaluators run
  concurrently (`ThreadPoolExecutor(max_workers=2)`) on the same `analysis`;
  `llm_calls += 2`.
- **`llm_calls` accounting.** planning (+1) + one per summarised doc (in
  `_gather_evidence`) + one per `_synthesise` (+1 each, +1 more on a forced
  redundant exit) + 2 evaluators. Persisted on the `ClaimVerdict` along with the
  full retrieval funnel counters and `past_queries`.

---

## B.5 Verification-evaluation

### B.5.1 The harness — `verification_grading/` (AVeriTeC)

- **`load_claims.py`** — loads HF `pminervini/averitec` **dev** split (`:37`),
  reproducibly shuffles with `--seed 42` and samples `-n` (default 100), writes
  `data/claims_n{N}.parquet`. `claim_id = sha1(original_claim_url)[:16]`
  (`stable_id` `:22-23`). Columns (`:46-57`): `claim_id, claim_text, gold_label,
  claim_date, speaker, fact_checking_article, gold_justification,
  original_claim_url`.
- **`verify_run.py`** — for each claim builds `AtomicClaim(text, embedding=
  MiniLM)` (`:58-61`), computes `date_ceiling` from `claim_date` normalized to
  ISO `YYYY-MM-DD` (`to_date_ceiling` `:22-35`, accepts `%m/%d/%Y`, `%Y-%m-%d`,
  `%d-%m-%Y`, `%d/%m/%Y`; fallback = today), sets `exclude_urls =
  [fact_checking_article]` to drop the gold article, and calls
  `verify(claim, date_ceiling=..., exclude_urls=..., verbose=False)` (`:65-70`).
  **Default 2 workers** (`:104`). Resumable via `claim_id` dedup; every error is
  captured per-row (`_error_row` `:38-52`), the run never aborts.
- **Output `verdicts_n{N}.parquet`** (one row/claim, README `:26-32`): the 4
  Likert scores, `justification`, `analysis`, `evidence_urls`, `past_queries`,
  the diagnostics (`rounds_used, n_urls_seen, n_blocked_or_failed, n_irrelevant,
  elapsed_seconds, llm_calls, cap_hit, redundant_exit, tier_resolved`),
  `verdict_4class` (+ justification), `error`.
- **Gold labels.** Saved alongside but **not** scored in v1 (README `:5`): the
  harness collects distributions + the 4-class label; macro-F1 against ClaimCheck
  / PASS-FC / HerO is left for later derivation.
- **AVeriTeC note** (README `:35`): dataset is `chenxwh/AVeriTeC` family
  (FEVER-2024/2025); ClaimCheck's 76.4% dev accuracy reportedly used AVeriTeC 1.0
  dev — schemas/labels compatible. (Loader uses the `pminervini/averitec` mirror.)

### B.5.2 Tier 1 & Tier 2 stubs; harmonisation (built)

- **Tier 1 — `pipeline/cache.py` (STUB).** `lookup`, `write`, `init_schema` all
  raise `NotImplementedError` (`:20-48`). Intended: Neon + pgvector
  `claim_verdicts` table, ANN by `claim_embedding <=> %s::vector`, hit iff
  `cosine ≥ SIMILARITY_THRESHOLD (0.92, config.py:5)` AND age `<
  RECHECK_AFTER_DAYS (30, config.py:6)`.
- **Tier 2 — `pipeline/fact_api.py` (STUB).** `lookup` raises
  `NotImplementedError` (`:12-25`). Intended: query FCT, filter to
  `TRUSTED_PUBLISHERS` (`config.py:8-17`: snopes, factcheck.afp.com,
  newschecker.in, verafiles.org, rumorscanner.com, politifact, factcheck.org,
  fullfact), pick most recent review, harmonise its `textualRating` via
  `eval.harmonize.rule_map` → `llm_map` fallback, build a `tier_resolved=2`
  `ClaimVerdict`.
- **`eval/harmonize.py` (BUILT).** Maps publisher rating strings → the AVeriTeC
  4-class scheme (`Supported / Refuted / Not Enough Evidence / Conflicting
  Evidence`, `config.py:36-41`). Built for `eval_v1`; **referenced by the Tier-2
  stub's TODO but not yet wired into a live path**.
  - **`rule_map(publisher_site, rating)`** (`:104-109`): per-publisher lowercased
    lookup table `RULES` covering 7 of 8 publishers (`:26-101`) — e.g. PolitiFact
    `half true → Conflicting Evidence`, `pants on fire → Refuted`; Snopes
    `mixture → CE`, `miscaptioned → Refuted`. Returns `None` on miss.
  - **`llm_map(model, rating)`** (`:156-181`): LLM fallback (Groq REST,
    `EXTRACTION_MODEL = qwen3-32b`, `temperature 0`, `response_format
    json_object`) for Full Fact's free-text ratings and any rule miss. System
    prompt `LLM_SYSTEM` (`:117-153`) defines the 4 classes by **evidence state,
    not rhetorical framing** (notably: absence-of-evidence → *Not Enough
    Evidence*; positive counter-evidence → *Refuted*). Non-canonical labels are
    substring-matched, else raises.

### B.5.3 Deterministic disk cache (`pipeline/disk_cache.py`)

Tiny on-disk JSON cache backing Tier-3 eval re-runs. Three namespaces (`:5-8`):
`searxng` (query→results), `scrape` (url→{text}), `summarise`
((url,claim,source)→EvidenceDoc). Keys SHA-256-hashed (`:23-25`); env
`CLAUDE_PIPELINE_CACHE_BYPASS=1` skips reads but still writes (`:19, :28-30`).
This is what makes "re-runs cost zero" possible and is distinct from the
(stubbed) Neon Tier-1 cache.

---

# Part C — Built vs Stubbed summary

| Half | Component | File | Status | Model / API |
|---|---|---|---|---|
| Extraction (cascade) | Stage 0 build posts | `claim_cascade/stage0_build_posts.py` | **BUILT** | none |
| Extraction · detection | Stage 1 embed topic/scope (parallel, not a gate) | `claim_cascade/stage1_topic_embed.py` + `anchors.py` | **BUILT** | MiniLM (local) |
| Extraction · detection | Stage 2 LLM scope gate (real gate) | `claim_cascade/stage2_scope_llm.py` + `prompts_scope.py` | **BUILT** | gpt-oss-120b |
| Extraction · decomposition | Stage 3 extract `{has_claim, claims}` | `extraction_grading/extract.py` + `prompts.py` | **BUILT** | gpt-oss-120b |
| Extraction · harm-prioritization | Stage 4 4-prong judge (`misinfo_candidate` + 3 Likert) | `extraction_grading/judge.py` + `prompts.py` | **BUILT** | qwen3-32b |
| Extraction · harm-prioritization | Stage 5 FABLE 5-dim harm | `claim_cascade/stage5_fable.py` + `prompts_scope.py` | **BUILT** | qwen3-32b |
| Extraction · harm-prioritization | Stage 7 FCT cross-check (short-circuit measure) | `claim_cascade/stage7_fct.py` | **BUILT** | Google FCT API |
| Extraction · evaluation | Stage 6 join | `claim_cascade/stage6_grade.py` + `extraction_grading/grade.py` | **BUILT** | none |
| Extraction · evaluation | funnel/agreement/rubric report | `claim_cascade/analyze_funnel.py` | **BUILT** | none |
| Extraction (production) | `extract.extract` + `_embed` | `pipeline/extract.py` | **STUB** (`NotImplementedError`) | (intended gpt-oss/MiniLM) |
| Verification · retrieval/planning | opening query | `pipeline/verify.py` `_plan_initial_query` | **BUILT** | gpt-oss-120b |
| Verification · retrieval/planning | SearXNG search + scrape | `pipeline/search.py` | **BUILT** | SearXNG + Trafilatura |
| Verification · evidence-processing | per-doc summarise + gather | `pipeline/verify.py` `_summarise_for_claim` / `_gather_evidence` | **BUILT** | gpt-oss-120b |
| Verification · verdict-prediction | synthesise (analysis + decision) | `pipeline/verify.py` `_synthesise` | **BUILT** | gpt-oss-120b |
| Verification · verdict-prediction | Likert 4-dim evaluator | `pipeline/verify.py` `_evaluate` | **BUILT** | gpt-oss-120b |
| Verification · verdict-prediction | 4-class (AVeriTeC/ClaimCheck) evaluator | `pipeline/verify.py` `_evaluate_4class` | **BUILT** | gpt-oss-120b |
| Verification · loop-control | bounded synth↔search loop, cap, redundancy | `pipeline/verify.py` `verify` / `_is_redundant` | **BUILT** | gpt-oss-120b (+ MiniLM) |
| Verification · evaluation | AVeriTeC harness | `verification_grading/load_claims.py` + `verify_run.py` | **BUILT** | gpt-oss-120b |
| Verification · evaluation | deterministic disk cache | `pipeline/disk_cache.py` | **BUILT** | none |
| Verification (Tier 1) | vector-DB cache lookup/write | `pipeline/cache.py` | **STUB** (`NotImplementedError`) | (intended Neon+pgvector) |
| Verification (Tier 2) | FCT lookup + verdict mapping | `pipeline/fact_api.py` | **STUB** (`NotImplementedError`) | (intended Google FCT) |
| Verification (Tier 2) | rating→4-class harmonisation | `eval/harmonize.py` | **BUILT** (not yet wired live) | rules + qwen3-32b |
| Orchestration | Tier 0→1→2→3 ladder | `pipeline/pipeline.py` | **BUILT but not end-to-end runnable** (depends on stubs) | — |

**Net:** the **extraction half runs end-to-end today only as the eval cascade**
(production `extract.py` stubbed). The **verification half is built only at Tier
3** (`verify.py`), driven directly by the AVeriTeC harness; Tier 1 and Tier 2 are
stubs, and the production orchestrator cannot complete a run until `extract`,
`cache`, and `fact_api` are implemented.

---

## Appendix — parameter quick-reference

| Param | Value | Where |
|---|---|---|
| Scope/extract temperature | 0.1 | `stage2_scope_llm.py:54`, `extract.py:77` |
| Judge temperature | **0.0** | `judge.py:89` |
| FABLE temperature | 0.1 | `stage5_fable.py:62` |
| Extraction-stack `max_tokens` | 4000 | `extract.py:78`, `judge.py:90`, `stage2:55`, `stage5:63` |
| Cascade workers | 12 (FCT 6) | runners; `stage7_fct.py:255` |
| `TAU` (embed in-scope cutoff) | 0.0 | `anchors.py:31` |
| `MIN_CHARS` (skip embed) | 5 | `anchors.py:34` |
| `FABLE_CHECKWORTHY_THRESHOLD` | 15 (range 5–25) | `prompts_scope.py:211` |
| FCT `pageSize` / quota-abort | 5 / 8 consecutive 429s | `stage7_fct.py:135, :74` |
| `MAX_ROUNDS` | 4 | `pipeline/config.py:20` |
| `SEARCH_TOP_K` | 3 | `pipeline/config.py:21` |
| `REDUNDANCY_THRESHOLD` | 0.9 | `pipeline/config.py:22` |
| `SCRAPE_TIMEOUT` | 10s | `pipeline/config.py:23` |
| per-domain delay | 2.0–3.5s | `pipeline/config.py:27-28` |
| SearXNG min interval | 0.5s | `search.py:37` |
| Verify token caps | plan 1000 / summarise 2000 / synth 4000 / eval 2000 | `verify.py:34-37` |
| doc truncation | 12000 chars | `verify.py:39` |
| min snippet to summarise | 40 chars | `verify.py:312` |
| Verify call temperatures | plan 0.2; summarise/synth/eval/4class 0.1 | `verify.py:240, :292, :432, :543, :629` |
| `SIMILARITY_THRESHOLD` (Tier 1, stub) | 0.92 | `pipeline/config.py:5` |
| `RECHECK_AFTER_DAYS` (Tier 1, stub) | 30 | `pipeline/config.py:6` |
| harness workers | 2 | `verify_run.py:104` |
| AVeriTeC sample seed / default N | 42 / 100 | `load_claims.py:28-29` |
