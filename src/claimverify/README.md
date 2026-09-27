# claimverify

## Purpose

`claimverify` is a claim-level port of the production post-level verifier
`pipeline/verify_tweet_claims.py` (LOOP_VERSION v7.5, 2026-07-21), built to score the
loop on the AVeriTeC dev split for GitHub issue #28. The production loop judges the
several claims extracted from one news outlet's social-media post; this package runs the
same retrieval, reading, closure-bar and guard code on ONE free-standing claim with a
date and an optional origin URL, which is what an AVeriTeC row is. The scored population
is the 500-row `claims_dev_500_gold.parquet` (491 unique claim ids; ClaimCheck reports
76.4 percent accuracy on this split). Everything the loop produced for a claim is kept in
a per-claim trace so any verdict can be replayed offline.

## How to run

All commands from `src/`, always under `uv run`. Keys come from the repo root `config.py`
(`.env`: DEEPINFRA_API_KEY, SERPER_API_KEY, JINA_API_KEY, EXA_API_KEY).

    uv run python -m claimverify.run_averitec --arm top3 --dry-run
    uv run python -m claimverify.run_averitec --arm top3 --smoke 10 --k 10 --budget 1
    uv run python -m claimverify.run_averitec --arm top3 --budget 5
    uv run python -m claimverify.run_averitec --arm all10 --budget 8
    uv run python -m claimverify.score_averitec --run eval/data/claimverify_runs/averitec_dev/top3

`--dry-run` builds the 491 claims, prints five of them with their date ceiling, excluded
origin and gold label, and touches no network. `--smoke N` runs the first N claims after a
seeded shuffle (seed 707) and writes `sample.json` with the ids. `--budget` is a USD cap:
after ten completions the run stops launching new claims once the projected spend exceeds
it; in-flight claims finish. Runs are resumable, a claim already present in a
`shard-XX.jsonl` is skipped.

The two arms differ in one number. `top3` reads the first 3 hits of the credibility-ranked
Serper list per round, `all10` reads all 10. Triage is off in both. Both arms share the
disk caches under `src/pipeline/.cache`: the Serper call for a given query, date ceiling
and exclusion list is cached once (`serper_raw` and the search namespace), scraped pages
are content-addressed in `pages/` and the LLM cache (`llm` namespace, exact match on
model, temperature, max_tokens and messages) replays any READ call whose input is
identical. So the `all10` arm pays for the extra 7 reads per round and nothing else.

Exa is OFF by default (1000 free requests a month, see the run ledger). `--exa` enables
the single escalation round the production loop runs for a still-open claim after its two
Serper tries.

`--no-bar` (RESOLVE's proposed close is taken as final, recorded as a `bar-off-accepted` guard event), `--no-blocklist` (no UGC `-site:` suffix, no client-side drop, no scrape gate — the Serper cache key changes with the query string so the two never share a cached call) and `--no-origin` (keep the origin site) switch off the three code policies v7.5 adds over ClaimCheck; with `--no-ceiling` they give the ClaimCheck-protocol arm:

    uv run python -m claimverify.run_averitec --arm top3 --no-ceiling --no-bar --no-blocklist --no-origin

`score_averitec` reads the shards, joins on claim_id to every gold row (duplicates inherit
their claim's prediction), writes `<run>/scored.parquet` in gold row order plus
`<run>/metrics.json`, and prints 4-class accuracy and macro-F1 under both label maps, the
three binary variants, and the speed and cost table. Rows whose claim has no successful
record are listed as missing and excluded from the metrics; every n is reported. The 9
duplicate gold rows share claim_id, claim text and label with their partner row and differ
only in `gold_justification` (and `speaker` for two ids).

Label maps (identical to `eval/scripts/verification_grading/run_averitec_benchmark.py`):
RAW sends `unsupported` to Not Enough Evidence; the CLAIMCHECK CONVENTION sends it to
Refuted, which is how ClaimCheck reports "no supporting evidence found". Binary variants,
applied to system and gold alike: A = Supported is pass, everything else flag; B =
Supported and Conflicting pass; C = gold Conflicting rows dropped, Supported pass. Since
`unsupported` is flag under both maps the binary numbers do not depend on the map.

## What was removed vs v7.5

- CONTEXT step (resolve the post's linked article, pin referents, self-sourced closes,
  then exclude the article). An AVeriTeC claim has no linking post and no outlet article.
- Pair logic (attribution "X said Y" and Y judged as an extraction pair). There is one
  claim per call, so there is nothing to pair.
- Speaker-as-handle and origin-token republication (the posting outlet's handle and domain
  tokens used to recognise its own copy). No posting outlet exists here; the speaker is
  not shown to the model at all.
- `is_republication` (wire byline, syndication host, reprint phrase). Every branch keys on
  origin tokens or the origin being a wire agency, neither of which an AVeriTeC claim has;
  READ's `republication` flag went with it (`_qualifying` still honours the field, which
  is always False).
- Nudge verdict (post-level "nudge iff at least one refuted claim"). The unit of
  evaluation is the claim's own status.
- Triage (LLM picks which hits to read and how many). Replaced by a fixed
  `pages_per_round` walk in credibility-rank order so the two arms differ in one number
  and nothing the model decides.
- Likert confidence. Absent in v7.5 as well; the loop emits a status, never a scale.

## Prompt changes old to new

The six prompts were rewritten for one claim. Only the post, multi-claim, pair and
posting-outlet scaffolding changed; every rule about evidence quality, stance, dates,
attribution, syndication and the closure bar is kept verbatim. The listing below was
produced by a line diff between the string constants in
`pipeline/verify_tweet_claims.py` and `claimverify/prompts.py` (difflib, no autojunk);
each hunk shows the old lines it replaces and the new lines in full. RESOLVE_FINAL_NOTE
is listed with the six because RESOLVE appends it on the last round.

### QUERY_SYSTEM

3 lines old, 3 lines new, 1 hunk(s), 3 old line(s) replaced by 3 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | You write the next web search for a fact-check of a news outlet's social-media post. You receive the POST (with its date), the OPEN claims still needing evidence — each with its search budget ("tried n/2"), its type, and a GAP note saying what evidence is missing — and the queries already tried.
    OLD | Write ONE new keyword query. Target the most AT-RISK open claim — most likely false, most harmful if false, most concretely checkable first — and fold in closely related open claims when one query can serve them; put their ids in "targets". Claims marked "paired with" each other (an attribution "X said Y" and Y as its own claim) usually share one search — coverage of the saying also surfaces evidence on Y itself; target both ids. Chase what the GAP notes point to: a primary/official record, a named person or document, a specific date, an exact figure. Make the query meaningfully different from every prior query — do not reword one. For time-sensitive claims anchor the query to the post's date — never guess a period the post does not state. Compact keyword query, 4-10 words, no operators.
    OLD | Output JSON: {"targets": [<claim ids>], "query": "<the search query>"}

Hunk 1, new:
    NEW | You write the next web search for a fact-check of ONE claim. You receive the CLAIM (with the date it was made), its search budget ("tried n/2"), its type, a GAP note saying what evidence is missing, and the queries already tried.
    NEW | Write ONE new keyword query. Chase what the GAP note points to: a primary/official record, a named person or document, a specific date, an exact figure. Make the query meaningfully different from every prior query — do not reword one. For time-sensitive claims anchor the query to the claim's date — never guess a period the claim does not state. Compact keyword query, 4-10 words, no operators.
    NEW | Output JSON: {"targets": [1], "query": "<the search query>"}

### READ_SYSTEM

17 lines old, 15 lines new, 4 hunk(s), 6 old line(s) replaced by 4 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | You are an evidence pointer for a fact-check. You receive a news article whose sentences are numbered [S1], [S2], ..., plus a social-media POST and the CLAIMS extracted from it.
    OLD | For each claim that the article speaks to, output the sentence numbers that bear on it and a page-local stance.

Hunk 1, new:
    NEW | You are an evidence pointer for a fact-check. You receive a news article whose sentences are numbered [S1], [S2], ..., plus the CLAIM being checked.
    NEW | If the article speaks to the claim, output the sentence numbers that bear on it and a page-local stance.

Hunk 2, old:
    OLD | Pair rule: when a bare claim Y also appears among the claims as reported speech ("X said Y"), this article merely REPORTING that X said Y bears on the attribution claim only — for the bare claim Y it is "neutral", however many outlets repeat the saying. "supports"/"refutes" for Y require the article's OWN voice (its own reporting, records, or data) to affirm or contradict Y itself.

Hunk 2, new:
    NEW | (no line)

Hunk 3, old:
    OLD | - OMIT claims the article says nothing about. Do not paraphrase or quote text. Do not judge overall truth.
    OLD | - Set "republication" true if this page is substantially the POSTING OUTLET'S OWN reporting relayed — a reprint/mirror of that outlet's article, or a page that merely restates that outlet's report without independent reporting. A page that discusses, investigates, or disputes the outlet's reporting with its own sources is NOT a republication.

Hunk 3, new:
    NEW | - If the article says nothing about the claim, output an empty evidence list. Do not paraphrase or quote text. Do not judge overall truth.

Hunk 4, old:
    OLD | {"docs": [{"doc": "D1", "republication": <bool>, "evidence": [{"claim_id": <n>, "segs": [<sentence numbers>], "stance": "supports|partially-supports|neutral|partially-refutes|refutes"}]}]}

Hunk 4, new:
    NEW | {"docs": [{"doc": "D1", "evidence": [{"claim_id": 1, "segs": [<sentence numbers>], "stance": "supports|partially-supports|neutral|partially-refutes|refutes"}]}]}

### RESOLVE_SYSTEM

22 lines old, 21 lines new, 5 hunk(s), 10 old line(s) replaced by 9 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | You are the claim resolver of a fact-checking loop, judging the claims of one social-media post by a news outlet. You receive the POST (with its date), the queries already run, and one DOSSIER per claim. An open claim's dossier gives: its status, search budget ("tried n/2"), claim type, a code-computed bar check, and its evidence entries — each with source id, domain, a reliability tag, publication date if known, an advisory reader stance, and source text in which the reader's cited sentences are marked like <<this>>; unmarked sentences are surrounding context. A closed claim's dossier is a one-line resolution (its full evidence is retained outside this prompt) — leave closed claims alone UNLESS their dossier says new contrary evidence arrived, in which case re-judge them.

Hunk 1, new:
    NEW | You are the claim resolver of a fact-checking loop, judging ONE claim. You receive the CLAIM (with the date it was made), the queries already run, and the claim's DOSSIER. An open claim's dossier gives: its status, search budget ("tried n/2"), claim type, a code-computed bar check, and its evidence entries — each with source id, domain, a reliability tag, publication date if known, an advisory reader stance, and source text in which the reader's cited sentences are marked like <<this>>; unmarked sentences are surrounding context. A closed claim's dossier is a one-line resolution (its full evidence is retained outside this prompt) — leave a closed claim alone UNLESS its dossier says new contrary evidence arrived, in which case re-judge it.

Hunk 2, old:
    OLD | Re-judge each open (or contrary-flagged) claim YOURSELF from the evidence text. The advisory stance is a hint and may be imperfect — the marked sentences and their context are the ground truth.

Hunk 2, new:
    NEW | Re-judge the open (or contrary-flagged) claim YOURSELF from the evidence text. The advisory stance is a hint and may be imperfect — the marked sentences and their context are the ground truth.

Hunk 3, old:
    OLD | - SAME-EVENT rule: evidence must concern the same event or period the post describes — use the post's date. A similar event at a different date (an earlier incident, a later statement, another month's statistic) neither supports nor refutes the claim; a source published before the claimed event cannot confirm it. If all evidence is off-period, the claim is still "open". The REFUTE direction is time-anchored too: a claim about the state of things at post time — a schedule ("launching soon"), a standing condition ("remains barred"), the existence of reports or rumors, or a superlative over a trailing window ("biggest in 13 months") — is judged AS OF THE POST DATE; evidence marked PUBLISHED AFTER THE POST describing a later development, ruling, confirmation, or data release does not refute what was true when posted.
    OLD | - ATTRIBUTION claims ("X said/claims Y", "the report states Y"): the proposition is the SAYING, not Y. If credible evidence confirms X said it, the claim is supported — even if Y itself is dubious. This applies ONLY to third parties: the posting outlet's own assertions ARE the post — judge them on substance, never as "the outlet said it".
    OLD | - PAIR rule: a dossier marked "paired with claim k" is one member of an extraction pair — the attribution "X said Y" and Y as its own bare claim. Judge the members INDEPENDENTLY; their statuses may differ. Evidence that merely reports the saying counts for the attribution member only: for the bare content member it is neither support nor refutation, however many outlets repeat it — the content member closes only on evidence in a source's own voice affirming or contradicting Y itself.

Hunk 3, new:
    NEW | - SAME-EVENT rule: evidence must concern the same event or period the claim describes — use the claim's date. A similar event at a different date (an earlier incident, a later statement, another month's statistic) neither supports nor refutes the claim; a source published before the claimed event cannot confirm it. If all evidence is off-period, the claim is still "open". The REFUTE direction is time-anchored too: a claim about the state of things at claim time — a schedule ("launching soon"), a standing condition ("remains barred"), the existence of reports or rumors, or a superlative over a trailing window ("biggest in 13 months") — is judged AS OF THE CLAIM DATE; evidence marked PUBLISHED AFTER THE CLAIM describing a later development, ruling, confirmation, or data release does not refute what was true when claimed.
    NEW | - ATTRIBUTION claims ("X said/claims Y", "the report states Y"): the proposition is the SAYING, not Y. If credible evidence confirms X said it, the claim is supported — even if Y itself is dubious.

Hunk 4, old:
    OLD | Closure bar (ENFORCED IN CODE — a close below it is refused and the claim stays open; each open dossier shows its own bar check): "supported"/"refuted" requires one READ RELIABLE source with NG>=90, or two independent voices (PRIMARY/institutional and reliable sources both count) on distinct domains of which at least one is READ — a SNIPPET ONLY entry from a PRIMARY or NG>=90 source may serve as the second voice, but snippets can NEVER close a claim by themselves, and a single PRIMARY/institutional source alone does not close a claim. UNRATED, UNRELIABLE, republication-flagged entries, and OPINION pieces (except for attribution claims) do not count toward the bar. If the evidence does not clear it, keep the claim "open".
    OLD | Independence rule: a source that merely republishes or mirrors the originating outlet's own reporting (same text, wire-style reprint, or explicitly sourced to that outlet) is NOT independent corroboration — weigh it as the outlet's own voice. The same applies to ANY shared upstream voice: the same wire story, the same media group, or the same press release/statement relayed across domains is ONE voice — count independent voices, not domains.

Hunk 4, new:
    NEW | Closure bar (ENFORCED IN CODE — a close below it is refused and the claim stays open; the open dossier shows its own bar check): "supported"/"refuted" requires one READ RELIABLE source with NG>=90, or two independent voices (PRIMARY/institutional and reliable sources both count) on distinct domains of which at least one is READ — a SNIPPET ONLY entry from a PRIMARY or NG>=90 source may serve as the second voice, but snippets can NEVER close a claim by themselves, and a single PRIMARY/institutional source alone does not close a claim. UNRATED, UNRELIABLE entries, and OPINION pieces (except for attribution claims) do not count toward the bar. If the evidence does not clear it, keep the claim "open".
    NEW | Independence rule: a source that merely republishes or mirrors another outlet's reporting (same text, wire-style reprint, or explicitly sourced to that outlet) is NOT independent corroboration — weigh it as that outlet's own voice. The same applies to ANY shared upstream voice: the same wire story, the same media group, or the same press release/statement relayed across domains is ONE voice — count independent voices, not domains.

Hunk 5, old:
    OLD | For every claim you leave "open", write a one-line "gap": the specific missing evidence the next search should chase (a primary record, a named person or document, a date, an exact figure) — not a restatement of the claim, and never a copy of evidence text already in the dossier.
    OLD | Output strictly valid JSON, no other keys. Emit a row ONLY for a claim whose status you are CHANGING, or an open claim whose gap you are updating in light of this round's evidence. A claim with nothing new — closed and standing, or open with no new evidence and an unchanged gap — gets NO row; the system keeps its state. Output length costs latency; say only what changed:
    OLD | {"ledger": [{"claim_id": <n>, "status": "supported|refuted|conflicting|unsupported|open", "gap": <string, only for status "open"> }]}

Hunk 5, new:
    NEW | If you leave the claim "open", write a one-line "gap": the specific missing evidence the next search should chase (a primary record, a named person or document, a date, an exact figure) — not a restatement of the claim, and never a copy of evidence text already in the dossier.
    NEW | Output strictly valid JSON, no other keys. Emit a row ONLY if you are CHANGING the claim's status, or updating an open claim's gap in light of this round's evidence. A claim with nothing new — closed and standing, or open with no new evidence and an unchanged gap — gets NO row; the system keeps its state. Output length costs latency; say only what changed:
    NEW | {"ledger": [{"claim_id": 1, "status": "supported|refuted|conflicting|unsupported|open", "gap": <string, only for status "open"> }]}

### RESOLVE_FINAL_NOTE

3 lines old, 3 lines new, 1 hunk(s), 1 old line(s) replaced by 1 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | FINAL ROUND: no further searches will run. Resolve what this evidence supports; any claim you leave open will be recorded as unsupported.

Hunk 1, new:
    NEW | FINAL ROUND: no further searches will run. Resolve what this evidence supports; if you leave the claim open it will be recorded as unsupported.

### TRIAGE_SYSTEM

7 lines old, 7 lines new, 3 hunk(s), 4 old line(s) replaced by 4 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | You are the triage step of a fact-checking loop. You receive a social-media POST, its OPEN claims, and up to 10 search results (id, domain, date, snippet). Full pages are read only for the results you pick — reading is the expensive step, snippets are free.

Hunk 1, new:
    NEW | You are the triage step of a fact-checking loop. You receive the CLAIM and up to 10 search results (id, domain, date, snippet). Full pages are read only for the results you pick — reading is the expensive step, snippets are free.

Hunk 2, old:
    OLD | 1. "read": which results to read IN FULL, as an ordered priority list (best first). Pick as many as this round needs: a claim can only be closed as supported by credible FULL sources (prefer 2 independent ones) — snippets never suffice — so pick enough to potentially close the open claims. Prefer primary/institutional sources and reliable outlets; skip results that are clearly the same wire story twice (one voice); skip results whose snippet shows the page is not actually about the claims. If NOTHING is relevant (the query missed), return an empty list — that is a valid, useful answer.
    OLD | 2. "snippet_evidence": results whose snippet ALREADY bears on a specific claim (states, contradicts, or gives a key figure for it), whether or not you also picked them to read.

Hunk 2, new:
    NEW | 1. "read": which results to read IN FULL, as an ordered priority list (best first). Pick as many as this round needs: a claim can only be closed as supported by credible FULL sources (prefer 2 independent ones) — snippets never suffice — so pick enough to potentially close the claim. Prefer primary/institutional sources and reliable outlets; skip results that are clearly the same wire story twice (one voice); skip results whose snippet shows the page is not actually about the claim. If NOTHING is relevant (the query missed), return an empty list — that is a valid, useful answer.
    NEW | 2. "snippet_evidence": results whose snippet ALREADY bears on the claim (states, contradicts, or gives a key figure for it), whether or not you also picked them to read.

Hunk 3, old:
    OLD | {"read": [<result ids, best first>], "snippet_evidence": [{"result": <id>, "claim_id": <n>}], "why": "<one sentence>"}

Hunk 3, new:
    NEW | {"read": [<result ids, best first>], "snippet_evidence": [{"result": <id>, "claim_id": 1}], "why": "<one sentence>"}

### EXA_QUERY_SYSTEM

9 lines old, 9 lines new, 3 hunk(s), 3 old line(s) replaced by 3 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | You compose ONE query for Exa, a NEURAL search engine, as the final escalation of a fact-check whose keyword searches left some claims unresolved. Exa does next-link prediction: it returns the page that would most plausibly be LINKED right after your sentence. So write the sentence a careful researcher would type immediately before pasting the ideal link — a content-rich DECLARATIVE sentence describing the evidence page itself, ending with a colon.

Hunk 1, new:
    NEW | You compose ONE query for Exa, a NEURAL search engine, as the final escalation of a fact-check whose keyword searches left the claim unresolved. Exa does next-link prediction: it returns the page that would most plausibly be LINKED right after your sentence. So write the sentence a careful researcher would type immediately before pasting the ideal link — a content-rich DECLARATIVE sentence describing the evidence page itself, ending with a colon.

Hunk 2, old:
    OLD | - Pack in semantic surface area: the concrete entities, figures, dates, and places from the OPEN claims. Long and specific beats short.

Hunk 2, new:
    NEW | - Pack in semantic surface area: the concrete entities, figures, dates, and places from the CLAIM. Long and specific beats short.

Hunk 3, old:
    OLD | - EXCEPTION: if an open claim hinges on an EXACT string (a verbatim quote, rare proper noun, document ID, exact figure), include that string verbatim in double quotes inside the sentence.

Hunk 3, new:
    NEW | - EXCEPTION: if the claim hinges on an EXACT string (a verbatim quote, rare proper noun, document ID, exact figure), include that string verbatim in double quotes inside the sentence.

### REQUERY_SYSTEM

2 lines old, 2 lines new, 1 hunk(s), 1 old line(s) replaced by 1 new line(s). Unlisted lines are byte-identical.

Hunk 1, old:
    OLD | The previous Google searches (listed) did not resolve this fact-check's open claims, and the next proposed query merely rewords them — running it would waste a round. Write ONE NEW keyword query that attacks the open claims from a DIFFERENT ANGLE than every prior query: change the entities you name, the phrasing, or the KIND of source you chase (switch toward a primary/official record, a named person or document, a specific date, or an exact figure). Do NOT just add or reorder words from the prior queries. Anchor time-sensitive claims to the post's date. Compact keyword query, 4-10 words, no operators, no quotes.

Hunk 1, new:
    NEW | The previous Google searches (listed) did not resolve this claim, and the next proposed query merely rewords them — running it would waste a round. Write ONE NEW keyword query that attacks the claim from a DIFFERENT ANGLE than every prior query: change the entities you name, the phrasing, or the KIND of source you chase (switch toward a primary/official record, a named person or document, a specific date, or an exact figure). Do NOT just add or reorder words from the prior queries. Anchor time-sensitive claims to the claim's date. Compact keyword query, 4-10 words, no operators, no quotes.

## Parity table

| Constant or rule | v7.5 | here | flag |
|---|---|---|---|
| tries_per_claim | 2 | 2 | same |
| serper_k | 10 | 10 | same |
| exa_k | 5 | 5 | same |
| cap_tok (article chars fed to READ) | 2500 | 2500 | same |
| ctx_window (sentences of context around cited segs) | 3 | 3 | same |
| max_segs (cited sentences per entry) | 8 | 8 | same |
| open / read / step / exa_query / requery / triage max_tokens | 300 / 2500 / 3200 / 400 / 120 / 300 | same six values | same |
| dossier_entries_cap | 5 | 5 | same |
| max_rounds | 0 (unlimited) | 0 | same |
| triage_enabled | True | False | changed, arms read a fixed count |
| pages_per_round | 3 (only used with triage off) | 3 (`top3`) or 10 (`all10`) | changed per arm |
| exa_enabled | True | False unless `--exa` | changed |
| date_ceiling | False in production, True in the benchmark parity runner | True | same as the benchmark runner |
| ceiling format | Serper `tbs=cdr:1,cd_min:1/1/1900,cd_max:M/D/YYYY`, unpadded, inclusive | identical string | same |
| code bar | NG_STRONG 90 single read source, or two independent voices on distinct domains with at least one read (PRIMARY and NG>=60 both count), a PRIMARY or NG>=90 snippet may be the second voice, single PRIMARY never closes; SNIPPET_MIN_OVERLAP 0.15; same-publisher families are one voice | same constants and functions (`_qualifying`, `_meets_bar`) | same |
| guards | cid-invalid, cid-coerced, unsupported-untargeted, close-unbacked, close-below-bar, snippet-corroboration, both-directions-qualify, conflict-one-sided, budget-exhausted, open at end coerced to unsupported | same events, same code | same |
| query redundancy | `_content_tokens` (regex, len>3) shadowed the earlier definition and was the one that ran | that tokenizer, defined once | same behaviour |
| per-domain scrape limit per round | 2 for a primary source, else 1; duplicate URLs skipped | same | same |
| snippet evidence | with triage off, only an unreadable hit contributes a snippet row | `snippet_evidence="fallback"` (default) reproduces that; `"all"` adds every hit | same by default |
| origin exclusion | post handle domains plus the linked article domain, excluded server-side (`-site:`) and dropped client-side | `origin_domain(original_claim_url)` (web.archive.org unwrapped, www stripped), same two mechanisms | changed source of the domain, same mechanism |
| social `-site:` suffix | first 16 of `_SOCIAL_PRIORITY` then the rest of the blocklist | identical string (checked programmatically) | same |
| blocklist | `pipeline/config.py SCRAPE_BLOCKLIST` | equal set (checked programmatically) | same |
| scrape timeout / per-domain delay | 10 s / 2.0 to 3.5 s | 10 s / 2.0 to 3.5 s | same |
| temperature | 0.0 (pool default, never overridden) | 0.0 | same |
| model | `VERIFICATION_MODEL` = deepseek-ai/DeepSeek-V4-Flash, json_object, non-thinking | same | same |
| LLM response cache | none | exact-match disk cache, on by default | changed (reproducibility) |
| CONTEXT step, pairs, republication, nudge verdict | present | removed | removed (see above) |
| speaker | post handle shown to the model | never shown | changed |

Four of those rows are switchable per run — `date_ceiling`, `code bar`, `blocklist` and `origin exclusion` are turned off by `--no-ceiling / --no-bar / --no-blocklist / --no-origin`, and `ClaimVerifyConfig` carries all four into `manifest.json` so an arm is reproducible from its config alone.

## Deviations from v7.5 found in smoke

Both were found by reading the traces of the first 10-claim smoke, and both are bugs the
production loop shares. They are fixed here and left untouched in `verify_tweet_claims.py`.

- Evidence segment ids: READ sometimes numbers its pointers "S3" instead of 3, and the
  int-only filter silently dropped the whole evidence entry. String ids matching S then
  digits are now coerced to int and a `seg-coerced` guard event records the doc and the raw
  values; an entry left with no usable segment is still dropped but now records
  `evidence-dropped-no-segs`.
- Final round: v7.5 marks a round final only when it is the Exa round, so with Exa off (the
  AVeriTeC arms) no round was ever final, the RESOLVE final note was never sent and the
  unsupported-untargeted bypass never applied. A round is now final when it is the Exa round
  or, with Exa disabled, when it is the last Serper try. Behaviour with Exa on is unchanged.
- Length cap: `finish_reason="length"` raised `CallFatal` and killed the whole claim (1/491 in
  `top3_noceil`, a READ on a 28-page court PDF). A capped READ is now a failed read — the doc is
  dropped with `read_status="length_cap"`; QUERY / TRIAGE / RESOLVE are re-asked once with a
  brevity nudge and then fall back (previous query / no triage / no close). Every case records a
  `length-cap` guard event with the call label. `max_tokens` is unchanged.

## Known gaps

- `is_republication` (wire byline / syndication-host / reprint phrase) was dropped: every
  branch depends on origin tokens or the origin being a wire agency, neither of which
  exists for an AVeriTeC claim. READ's `republication` flag removed from the prompt;
  `_qualifying` still honours the field (always False).
- Cost: DeepInfra `estimated_cost` is read from the SDK usage object's model_extra when
  present; otherwise PRICE_PER_M (0.20 / 0.02 / 0.80 USD per M, ledger-derived list-price
  ceiling, unverified). Each call's usage carries `cost_source` so the two are
  distinguishable, and `score_averitec` sums cost by source from the traces.
- Serper "top-3" is the first 3 of our credibility-ranked list of the 10 Google hits, not
  Google's top 3.
- `llm_cache` is on by default: a re-run replays cached responses (reproducible, $0). Pass
  a cold `disk_cache.set_dir` or CLAUDE_PIPELINE_CACHE_BYPASS=1 for a fresh measurement of
  latency.
- No live call has been made yet; the loop is exercised end-to-end only through the
  offline fake-pool tests.

## Reproducibility

`manifest.json` in the run directory records: `git_sha`, `git_dirty`, `timestamp_utc`,
`loop_version` (v7.5-claim), `arm`, `n_claims`, `seed` (707), `smoke`, `model`, the full
`config` (ClaimVerifyConfig) and `orchestration` (OrchestrationConfig) dataclasses,
`prompts_sha256` per prompt and `prompts_bundle_sha256`, `dataset` and `dataset_sha256`
(the gold parquet), `full_parquet_sha256` (averitec_full.parquet, the source of
claim_types, reporting_source and fc_url, which are carried on the claim and never shown
to the model), `cache_dir` and `budget_usd`. `config.json` repeats the two configs.

Caches: `src/pipeline/.cache` namespaces `llm` (exact-match responses), `serper_raw` and
`exa_raw` (raw provider JSON under the search cache key, which folds in the blocklist
hash so editing the blocklist rotates the key) plus the pipeline's search and scrape
namespaces. `<run>/pages/` holds every scraped page content-addressed by sha256(url) with
`index.jsonl`; `<run>/trace/<claim_id>.json` holds every LLM call (messages, raw output,
usage, latency, attempts, cache hit), search payload and raw result, scrape record, the
numbered block each READ saw, the dossier each RESOLVE saw and every ledger transition.

Claim order is a seeded shuffle (seed 707) of the ids sorted lexically; `--smoke N` takes
the first N. Temperature is 0 and the LLM cache is exact-match, so a re-run over the same
cache reproduces the verdicts; a cold cache reproduces them up to provider nondeterminism.
