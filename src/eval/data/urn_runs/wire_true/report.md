# wire-true on the new chain, 2026-09-10

Population is the 768 reputable-outlet posts behind `true_urn_assertions.parquet`. Chain is
`key_claim_extract.py` (extract-key-v2, gpt-oss-120b) then `wire_urn_run.py` (query-v3 on
DeepSeek-V4-Flash, Serper top-10, ceiling = post date, origin exclusion = outlet domain plus
`outlet_aliases.server_side()`, reader v7qa on gpt-oss-120b, context OFF). Collapse is
`syndication.py` at containment 0.7, post hoc on the stored documents.

Run actuals. **1,185 claims over 730 posts, $1.039, 11.5 minutes, 102.7 claims/min**, on DeepInfra
through `reader_lab.AdaptiveGate` at 96 claim workers. The gate started at 48, walked to its cap of
96 and **took zero throttles**. 10,908 documents, **10,899 read ok, 1 failed read, 8 empty-doc**,
48 claims with no documents. Extraction earlier the same day was 768/768 posts, 1,185 claims,
$0.182, 19 seconds.

## The superseded Groq pass

The same 1,185 claims were first run on Groq at 40 fixed workers, $1.733 and 21.3 minutes, and
**51.6% of its reads died** (5,558 of 10,804 came back `read_status="failed"`). The failures were
length-sorted, 12.3% under 15 sentences against 88.8% over 100, which is a per-minute token ceiling
refusing the big requests, and `run_claim` retries twice with a 2 second sleep, which is nothing
against a 60 second bucket. Under pad-to-ten an unread slot is a silent slot, so that urn reached
the fit 78.8% silent and produced **w_sil = +0.146** with an interval excluding zero, silence priced
as evidence of truth. Daniel's call was no more Groq. Those files are `*.groq-superseded.*` in this
folder and are the record of what went wrong, nothing else.

One thing survives from it. The read-failure bound computed there, filling each dead slot from the
successful reads in its own length bucket, predicted w_sup +1.145, w_ref −1.583, w_sil −0.179 and
3-voice AUC 0.847. The clean run below gives +1.165, −1.645, −0.189 and 0.847. The imputation was
right to about 0.02 on every weight, which confirms document length was the whole of the selection.

## 1. Collapse at containment 0.7

1,185 claims, 730 posts, 1.62 claims per post, 48 claims with zero documents.
10,908 documents to 10,689 kept, 9.21 to 9.02 per claim.

| cut | claims | docs | kept | origin_copy | syndicated |
|---|---:|---:|---:|---:|---:|
| ALL | 1,185 | 10,908 | 10,689 | 101 (0.93%) | 118 (1.08%) |
| ng90 | 625 | 5,749 | 5,574 | 101 (1.76%) | 74 (1.29%) |
| ng80 | 560 | 5,159 | 5,115 | 0 | 44 (0.85%) |
| apnews.com | 83 | 728 | 656 | 58 (7.97%) | 14 (1.92%) |
| reuters.com | 97 | 898 | 846 | 43 (4.79%) | 9 (1.00%) |
| AP + Reuters | 180 | 1,626 | 1,502 | 101 (6.21%) | 23 (1.41%) |
| non-wire | 1,005 | 9,282 | 9,187 | 0 | 95 (1.02%) |

All 101 origin-by-byline hits sit on AP and Reuters claims, which is rule (a) working as designed,
and all of them sit in ng90 because AP and Reuters are ng90 outlets. The wire cut with aliases is
identical to apnews.com plus reuters.com, so the alias list added no claims here. Near-duplicate
collapse is small and flat, 1.08% overall, 1.41% on wire, 1.02% off it.

Effect on the flag mix, support / refute / silent.

| cut | per read, before | per read, after | pad-to-10, before | pad-to-10, after |
|---|---|---|---|---|
| ALL | 44.8 / 4.2 / 51.1 | 44.3 / 4.2 / 51.6 | 41.2 / 3.8 / 55.0 | 39.9 / 3.8 / 56.4 |
| AP + Reuters | 48.8 / 4.2 / 47.0 | 47.0 / 4.1 / 48.9 | 44.0 / 3.8 / 52.2 | 39.2 / 3.4 / 57.4 |
| non-wire | 44.1 / 4.2 / 51.8 | 43.8 / 4.2 / 52.0 | 40.7 / 3.8 / 55.5 | 40.0 / 3.8 / 56.2 |

The collapse moves wire claims 1.9 points toward silence per read and 4.8 points once the freed
slots pad silent, against 0.2 and 0.7 for non-wire. That is the direction and roughly the size the
design predicted, and it is the whole reason the rule exists.

## 2. Old chain against new chain, 724 shared posts

Old is `true_outlet/scores_collapsed.jsonl` (27 August, extract then normalize then checkworthy,
read-v5 on DeepSeek-V4-Flash). New is this run. Restricted to the 724 posts both chains produced
claims for, out of 762 old and 730 new.

| | old chain | new chain |
|---|---:|---:|
| claims | 1,166 | 1,178 |
| claims per post | 1.61 | 1.63 |
| documents per claim | 9.10 | 9.21 |
| documents read per claim | 9.05 | 9.21 |
| zero-document claims | 68 | 47 |
| read failures | 60 (0.57%) | 9 (0.08%) |
| origin_copy | 84 (0.79%) | 101 (0.93%) |
| syndicated | 91 (0.86%) | 114 (1.05%) |
| sup / ref / sil, per read, before collapse | 47.0 / 6.9 / 46.1 | 44.7 / 4.2 / 51.2 |
| sup / ref / sil, per read, after collapse | 46.7 / 6.9 / 46.5 | 44.2 / 4.2 / 51.6 |
| sup / ref / sil, pad-to-10, after collapse | 41.5 / 6.1 / 52.3 | 39.9 / 3.8 / 56.3 |

The two chains retrieve the same amount of evidence and now read essentially all of it. The new
chain extracts marginally more claims per post, supports 2.5 points less often and **refutes about
40% less often**, 4.2% of reads against 6.9%. It is a quieter urn, which is what the smoke said in
the morning and what the old-chain comparison said before the reads broke.

Seven-flag mix per read, after collapse, on the shared posts, beside the two fitted urns as the fit
sees them (pad-to-ten).

| | n | 5 | 4 | 3 | 2 | 1 | X | I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old chain, per read | 10,379 | 30.8 | 15.8 | 0.6 | 3.3 | 3.6 | 18.7 | 27.2 |
| new chain, per read | 10,629 | 37.9 | 6.3 | 0.4 | 0.6 | 3.5 | 35.3 | 16.0 |
| cn_false, pad-to-10 | 17,390 | 6.6 | 5.9 | 0.6 | 5.5 | 14.0 | 12.5 | 55.0 |
| x_feed, pad-to-10 | 19,700 | 23.1 | 11.0 | 0.3 | 4.0 | 3.3 | 17.8 | 40.5 |
| wire-true, pad-to-10 | 11,850 | 34.2 | 5.6 | 0.3 | 0.6 | 3.2 | 31.8 | 24.2 |

### What is comparable and what is not

Comparable. Both readers emit the same seven flags, both runs aggregate regions with the same
`aggregate_reads`, both map flags to voices the same way (5 and 4 support, 1 and 2 refute, 3 and X
and I silent), and context was OFF in both. So **the three-voice split is comparable across the two
chains**.

Not comparable. The per-flag counts are produced differently. read-v5 asks the model for the flag
itself, with a claim-mode variant for attribution against assertion. v7qa asks two questions, does
the document assert the claim and does it assert the opposite, each answered direct or partial or
no, plus whether the document is on the claim's subject, and `map_qa` turns the answers into a flag.
Both non-no maps to 3, A direct maps to 5, A partial maps to 4, B direct maps to 1, B partial maps
to 2, otherwise X when on subject and I when not.

That mapping is visible in the table. The new reader moves mass from 4 to 5 (15.8 to 6.3 and 30.8 to
37.9) because a partial assertion is a much narrower thing than read-v5's downgraded support, and it
moves mass from I to X (27.2 to 16.0 and 18.7 to 35.3) because it asks about the subject explicitly.
Flag 2 nearly vanishes, 3.3 to 0.6. **A weight fitted on a v7qa flag is not the same object as a
weight applied to a read-v5 flag**, and section 3 shows what that costs.

## 3. Provisional two-urn fit

TRUE side is wire-true collapsed with pad-to-ten, `origin_copy` and `syndicated_of` documents
dropped before counting, eps 0 because reputable-outlet claims are treated as pure true here. FALSE
side is `cn_false.parquet` over `c2_false/scores[_ext].jsonl`, 1,739 claims over 1,739 posts.
Transfer is `fc_gold.parquet`, 3,280 claims, fixed weights, no refit, thresholds nested over the
cluster-disjoint folds. Cluster bootstrap is by post_id within each urn, 2,000 reps, seed 707.
Driver is `eval/scripts/build_eval/wire_true_fit.py`, which imports the model from `fit_two_urn` and
reproduces the pinned cn+feed cell to four decimals, so the plumbing is verified.

| fit | true side | eps | n_true | w_sup | w_ref | w_sil | AUC 3-voice | rec@2% FPR | AUC 7-flag | rec@2% FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| two-urn | **wire-true** | 0.00 | 1,185 | +1.165 | −1.645 | −0.189 | **0.847** | 0.293 | **0.810** | 0.296 |
| two-urn | x_feed | 0.00 | 1,970 | +1.009 | −0.982 | −0.150 | 0.850 | 0.310 | 0.841 | 0.403 |
| two-urn | x_feed, pinned | 0.10 | 1,970 | +1.077 | −1.188 | −0.168 | 0.851 | 0.310 | 0.845 | 0.395 |
| in-corpus | fc-gold refit | n/a | n/a | +2.187 | −1.840 | −0.125 | 0.848 | 0.310 | 0.856 | 0.400 |

The fc-gold row is `populations/refit_results.json`, cells `fit_urn_3voice` and `graded_urn_7flag`.
The x_feed eps 0.10 row is the pinned `two_urn_3voice` and `two_urn_7flag` cell of the same file.

95% cluster intervals on the wire-true weights are w_sup [+1.075, +1.267], w_ref [−1.803, −1.482],
w_sil [−0.232, −0.148]. Design effects are 1.03, 0.97 and 1.19, so clustering by post costs almost
nothing on this urn, against 1.30, 1.16 and 1.76 on x_feed. The reason is that wire-true has 1.62
claims per post where the timeline urn has up to 26. By tier the weights are indistinguishable,
ng90 gives +1.167 / −1.610 / −0.193 over 625 claims and ng80 gives +1.164 / −1.685 / −0.186 over 560,
which is mildly reassuring given the tiers were split on NewsGuard score.

Seven-flag weights at eps 0, beside the pinned feed fit.

| flag | wire-true | x_feed at 0.10 |
|---|---:|---:|
| 5 | +1.651 | +1.334 |
| 4 | **−0.043** | **+0.679** |
| 3 | −0.637 | −0.913 |
| 2 | −2.231 | −0.357 |
| 1 | −1.480 | −1.896 |
| X | +0.935 | +0.389 |
| I | −0.819 | −0.346 |

### Three readings of that table

**The three-voice model transfers, the seven-flag model does not.** Three-voice reaches AUC 0.847
against x_feed's 0.850 and fc-gold's own in-corpus 0.848, which is the result the plan was hoping
for. Seven-flag reaches 0.810 against 0.845 and 0.856, and that is a real loss, not noise. The cause
is the reader gap of section 2. Three-voice survives because it adds 5 and 4 together before
weighting them, so it never has to know which of the two readers produced the flag.

**The w_4 finding holds, in a softer form.** On the Groq pass w_4 came out sign-flipped, −0.153
against x_feed's +0.679. On the clean run it is **−0.043 with an interval of [−0.18, +0.10]**, so it
is no longer negative, it is **indistinguishable from zero**, while x_feed's sits at +0.679 [+0.49,
+0.77] and the intervals are nowhere near each other. So the honest statement is not "the sign
flips", it is **"v7qa's flag 4 carries no evidence about truth at all, where read-v5's flag 4 carries
about half of flag 5's"**. Applied to fc-gold, whose documents were flagged by read-v5, that weight
is simply wrong, and it is the main reason the seven-flag transfer drops.

**Refutation is priced much harder than on the feed.** w_ref is −1.645 against x_feed's −0.982 at the
same eps, and flag 2 is at −2.231 against −0.357. Reputable-outlet claims almost never draw a
refuting document, 4.2% of reads against the feed's 7.3% of slots, so any refutation that does turn
up is strong evidence. Whether that is a property of true claims or of v7qa's reluctance to answer
question B is not separable from this run, and the flag 2 column says it is at least partly the
reader, since v7qa emits 2 on 0.6% of reads where read-v5 emits it on 3.3%.

## 4. Asymmetries to state on any of these numbers

cn-false and fc-gold were extracted by the old chain and read by **read-v5 on DeepSeek-V4-Flash**.
wire-true was extracted by **key_claim_extract (extract-key-v2)** and read by **v7qa on
gpt-oss-120b**. Context was OFF everywhere, which is the one thing that is clean, and both the
reader and the query generator now run on DeepInfra.

So a two-urn fit with wire-true as the true side compares a v7qa-read true urn against a
read-v5-read false urn, then applies the result to a read-v5-read gold corpus. Section 3 shows the
three-voice model tolerating that and the seven-flag model not tolerating it. Re-reading cn-false's
stored documents with v7qa is about 14k reads and roughly $1.30 at the DeepInfra unit cost, and
fc-gold is about 30k reads. Until at least cn-false is re-read, **the seven-flag row of the fit table
is not a measurement of wire-true, it is a measurement of the reader gap**, and the three-voice row
is the only one worth quoting.

## Files

- `results-00.jsonl`, 1,185 scored claims, the DeepInfra pass
- `results-00_collapsed.jsonl`, the same with `origin_copy` and `syndicated_of` marked, nothing
  deleted
- `fit_provisional.json`, every fit number in section 3
- `eval/scripts/build_eval/wire_true_fit.py`, the driver, imports `fit_two_urn`
- `*.groq-superseded.*`, the abandoned 40-worker Groq pass
