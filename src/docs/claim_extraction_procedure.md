# Procedure for Selecting News Claims for the Experiment

Parallel to the tweet-selection procedure, but for claims extracted from news articles.

## Overview

We build a pool of check-worthy factual claims from real news outlets, spanning the reliability by
political-lean space. Extraction does not judge truth. It pulls the claims each outlet asserts, and a
separate verification stage (Stage 3) assigns each claim a veracity. The aim is a balanced pool across
four cells: reliable-left, reliable-right, unreliable-left, unreliable-right.

## Step-by-step

### 1. Select outlets (reliability x lean, balanced)

- Reliability comes from the NewsGuard score (0 to 100; 60 and above is reliable, below 60 is
  unreliable). NewsGuard is used internally only and never goes into anything we publish.
- Political lean comes from NewsGuard's own Orientation field (Left or Right). This is what lets us
  place the low-reliability sites, which AllSides does not rate.
- Form a 2x2 grid: reliable / unreliable by left / right.
- In each cell, rank outlets by Tranco traffic and take the most popular ones, keeping only news
  outlets (drop think-tanks, advocacy organizations, and foreign outlets).
- Result: about 15 outlets per cell, 59 in total.

### 2. Sample articles (same procedure for every outlet)

- For each outlet, pull recent articles from its RSS feed. Fall back to a homepage scrape, or to Jina
  for the few outlets that block automated access (paywalls, bot blocks).
- Keep only articles about federal politics or the election, using the keyword lists from the
  tweet-selection doc (candidates and institutions, plus the issue keywords).
- Text only. No claims based on images or video.
- Cap the number of claims taken per outlet (about 18) so no single outlet dominates a cell. Target
  about 250 claims per cell.

### 3. Extract claims (neutral, in the outlet's own voice)

The extractor is one LLM (DeepSeek) reading each article body. The prompt is `src/eval/ace.py`.

- It does not condition on whether a claim looks true, false, or suspicious. It pulls what is central
  and empirically checkable, of every kind, and lets Stage 3 judge truth.
- It extracts only what the outlet states as fact in its own editorial voice. A claim the article
  merely reports or is skeptical of is not flattened into a bare assertion.
- Claims are made self-contained (who, what, when resolved) but are never strengthened beyond what the
  article actually says.
- Bodies only. The headline-only path was dropped: a bare headline has no context, so it produced
  decontextualized claims, and headlines added almost nothing over full articles.

### 4. Filter and de-duplicate

- Semantic de-duplication (embedding similarity) removes near-duplicate claims within and across
  outlets.
- A check-worthiness filter drops the remaining opinion or non-checkable residue.
- The full keep / drop record is saved so the pool is auditable.

### 5. Verify (separate stage)

Stage 3 assigns each claim a veracity from 1 to 5 with a full evidence trace, fact-check-blocked and
date-limited. Truth is judged here, not during extraction.

## Current pull (provisional, 2026-07-06, before filter and dedup)

59 outlets, 674 raw claims: reliable-left 120, reliable-right 162, unreliable-left 220,
unreliable-right 172. Per-source detail (reliability, lean, NewsGuard score, traffic rank, articles
processed, claims extracted) is in `survey_claims/sources_pull_summary.csv`.

## Notes

- Genuinely unreliable left-leaning outlets are scarce. The unreliable-left cell is mostly local
  nonprofit news plus a handful of hyperpartisan sites, nothing like the volume on the right. Most of
  the actually-false pro-Dem claims will therefore need to come from the fact-check route, consistent
  with the tweet-selection doc's note that pro-Dem false claims are much fewer.
- Code: outlet selection and harvest in `src/eval/scripts/claim_sourcing/harvest_shortlist.py`;
  extractor prompt in `src/eval/ace.py`.
