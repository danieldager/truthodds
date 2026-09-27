# Query best practices — Serper (keyword) & Exa (neural)

Reference for tuning the verify-loop query prompts. From two research sweeps (2026-07-09).
Serper = opening query (keyword, non-thinking); Exa = final escalation query (neural, thinking).

## Serper / Google (opening query)

1. **Keyword bag, not a sentence or question.** Load-bearing tokens only: actor + action + object (+ crux number/date). ~4–8 content words.
2. **Include only load-bearing entities**; drop framing, hedges, attribution scaffolding, non-decisive adjectives.
3. **Right-size length.** Each term is an AND that shrinks results. Over-specificity → empty result set is the #1 failure.
4. **Quote only REAL verbatim phrases** (an actually-published/spoken span). Quoting an invented paraphrase = guaranteed zero results. For "X said Y" claims, quote the real words; keyword-bag the rest.
5. **Verbatim for utterances, unquoted keyword-bag for facts/stats** (so semantically-equivalent evidence still ranks).
6. **One query = one proposition.** Decompose compound claims.
7. **Don't resurface the origin.** Exclude the source (we do this server-side via `exclude_domains`; see [[project_verify_exclude_origin_source]]).
8. **On empty/weak results, BROADEN** (drop quotes/dates/terms), never add AND terms.

**Params (search.py):** `gl`/`hl` per claim locale; `num=10`; `autocorrect` on (off when a claim hinges on an exact proper noun); `tbs` time-window only for time-anchored claims.

## Exa (final escalation query)

**Mental model: next-link prediction** — write the sentence a human would type right before pasting the ideal link, NOT the keywords on the target page.

1. **Declarative statement describing the ideal evidence page, ending in a colon.** e.g. `Here is an authoritative primary source that confirms or refutes the claim that <core>, specifically addressing <sub-question>:`
2. **Long & content-rich is GOOD** — Exa rewards semantic surface area. Our compose design (claim core + prior angles + new sub-question) is on-strategy.
3. **Name the source archetype**: primary reporting, official record, peer-reviewed study, fact-check.
4. **Fold prior failed keyword strings in as CONTEXT, not as the operative query.**
5. **Question form and bare keywords both underperform** neural.
6. **EXCEPTION — exact-match claims route to keyword, not neural.** Verbatim quotes, rare proper nouns, IDs, case/statute numbers, exact figures: neural blurs exact strings. For these, escalate with `keyword`/`auto` + the quoted string, not a neural sentence.

**Params (search.py):** `type=neural` (or `auto`); `numResults=10–20`; `category` (`news`/`research paper`); light `includeDomains` authoritative nudge; `startPublishedDate`/`endPublishedDate` window for time-bound claims; `contents.highlights=true` (10× token-efficient excerpts) with full `text` as fallback. Don't stack narrow date + tiny includeDomains + low numResults on the final attempt (starves recall).

## Sources
Serper/Google: Bright Data SERP params, SerpApi tips, Google search-refinement help, FIRE (arXiv 2411.00784), ClaimCheck (arXiv 2510.01226), TACL fact-checking survey.
Exa: docs.exa.ai capabilities-explained + search-api-guide-for-coding-agents + Websets prompting, AI Wiki (next-link prediction), exa.ai/blog, morphllm writeup.
