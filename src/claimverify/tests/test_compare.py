"""Offline comparison test: two synthetic verifier arms + one urn parquet over 6 gold rows
(one of them a duplicate that must inherit its claim's prediction). Checks the matched-FPR
threshold pick, the McNemar discordant counts and the duplicate inheritance."""
import json

import numpy as np
import pandas as pd
import pytest

from claimverify.compare_averitec import compare, mcnemar, pick_threshold

CONF = "Conflicting Evidence/Cherrypicking"
NEE = "Not Enough Evidence"
RAW = {"supported": "Supported", "refuted": "Refuted", "unsupported": NEE, "conflicting": CONF}

# claim_id, gold, arm-a status, arm-b status, urn score (None = not covered)
ROWS = [
    ("c1", "Supported", "supported", "supported", 3.0),    # gold pass, both right
    ("c2", "Supported", "refuted", "supported", 1.0),      # gold pass, a false-positives
    ("c3", "Refuted", "refuted", "supported", -2.0),       # gold flag, b misses
    ("c4", "Refuted", "unsupported", "refuted", -1.0),     # gold flag, both flag
    ("c5", "Refuted", "refuted", "refuted", -3.0),         # gold flag, both flag
    ("c5", "Refuted", "refuted", "refuted", -3.0),         # duplicate gold row, inherits c5
    ("c6", CONF, "supported", "refuted", 0.5),             # gold flag under A
    ("c7", "Supported", "supported", "refuted", None),     # gold pass, urn not covered
]
LADDER = {"models": {"3-voice": {"threshold_2pct": -2.5, "weights": {}},
                     "7-flag": {"threshold_2pct": -2.5, "weights": {}}}}


def _write_arm(root, name, key):
    d = root / name
    (d / "trace").mkdir(parents=True)
    with (d / "shard-00.jsonl").open("w") as f:
        for i, (cid, _, a, b, _s) in enumerate(ROWS):
            if i == 5:
                continue  # the duplicate gold row has no record of its own
            st = a if key == 0 else b
            f.write(json.dumps({"claim_id": cid, "ok": True, "elapsed_s": 5.0, "result": {
                "claim_id": cid, "status": st, "verdict_raw": RAW[st],
                "verdict_cc": "Refuted" if st == "unsupported" else RAW[st],
                "elapsed_s": 5.0, "llm_calls": 3, "llm_cache_hits": 1, "serper_calls": 2,
                "exa_calls": 1, "scrapes": 4, "cost_usd": 0.001,
                "tokens": {"prompt": 100, "cached": 10, "completion": 20}}}) + "\n")
    (d / "manifest.json").write_text(json.dumps({"orchestration": {"k_claims": 4}}))
    (d / "trace" / "c1.json").write_text(json.dumps({"llm_calls": [
        {"latency_s": 2.0, "cache_hit": False}, {"latency_s": 99.0, "cache_hit": True}],
        "searches": [{"provider": "serper", "n_results": 10, "post_ceiling_hits": 2,
                      "undated_hits": 1, "cache_hit": False},
                     {"provider": "exa", "n_results": 3, "cache_hit": False}]}))
    return d


@pytest.fixture
def inputs(tmp_path):
    from claimverify.score_averitec import score_run
    gold = pd.DataFrame({"claim_id": [r[0] for r in ROWS], "gold_label": [r[1] for r in ROWS],
                         "claim_text": "x", "claim_date": "31-10-2020",
                         "original_claim_url": [None if i % 2 else "http://x" for i in range(len(ROWS))]})
    gold_path = tmp_path / "gold.parquet"
    gold.to_parquet(gold_path, index=False)
    arms = {}
    for key, name in enumerate(("a", "b")):
        d = _write_arm(tmp_path, name, key)
        df, m = score_run(d, gold_path)
        df.to_parquet(d / "scored.parquet", index=False)
        (d / "metrics.json").write_text(json.dumps(m))
        arms[name] = d
    urn = pd.DataFrame([{"claim_id": cid, "s3": s, "s7": s}
                        for cid, _, _, _, s in ROWS if s is not None]).drop_duplicates("claim_id")
    urn_path = tmp_path / "u.scored.parquet"
    urn.to_parquet(urn_path, index=False)
    ladder_path = tmp_path / "ladder.json"
    ladder_path.write_text(json.dumps(LADDER))
    return arms, urn_path, gold_path, ladder_path


def test_pick_threshold_matches_fpr_from_below():
    scores = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    gold_pass = np.array([False, False, False, True, True, True])  # passes score 0, 1, 2
    # flag = score <= thr. FPR budget 0 => no pass row may be flagged => thr just under 0.
    assert pick_threshold(scores, gold_pass, 0.0) == -1.0
    # 1/3 of the pass rows may be flagged: thr 0.0 flags exactly one of them.
    assert pick_threshold(scores, gold_pass, 1 / 3) == 0.0
    # 0.5 budget is not achievable above 1/3 without hitting 2/3, so it stays at 0.0.
    assert pick_threshold(scores, gold_pass, 0.5) == 0.0
    assert pick_threshold(scores, gold_pass, 1.0) == 2.0
    # nothing may be flagged and every score would flag: fall back to -inf.
    assert pick_threshold(np.array([1.0]), np.array([True]), 0.0) == float("-inf")


def test_mcnemar_counts_and_p():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    m = mcnemar(a, b, np.ones(4, bool))
    assert (m["a_only_correct"], m["b_only_correct"], m["n_discordant"]) == (1, 1, 2)
    assert m["p_exact"] == pytest.approx(1.0)
    m2 = mcnemar(a, b, np.array([True, True, False, True]))  # drop the b-only pair
    assert (m2["a_only_correct"], m2["b_only_correct"], m2["n_pairs"]) == (1, 0, 3)
    assert m2["p_exact"] == pytest.approx(1.0)  # 2 * P(X <= 0) with n = 1
    big = mcnemar(np.array([True] * 8 + [False]), np.array([False] * 9), np.ones(9, bool))
    assert big["n_discordant"] == 8 and big["p_exact"] == pytest.approx(2 / 2 ** 8)


def test_compare(inputs):
    arms, urn_path, gold_path, ladder_path = inputs
    c = compare(arms, urn_path, gold_path, ladder_path, reps=100, seed=0)

    assert c["n_gold_rows"] == 8 and c["n_gold_claims"] == 7
    # duplicate inheritance: c5 contributes twice, so 5 of 8 rows are Refuted gold.
    assert c["arms"]["a"]["binary"]["A"]["n_gold_flag"] == 5
    # arm a on map A: flags c3 c4 c5 c5 (recall 4/5, misses c6) and c2 (FPR 1/3).
    ba = c["arms"]["a"]["binary"]["A"]
    assert ba["flag_recall"]["value"] == pytest.approx(0.8)
    assert ba["fpr"]["value"] == pytest.approx(1 / 3)
    # arm b flags c4 c5 c5 c6, missing c3: recall 4/5, and c7 is its FPR 1/3.
    bb = c["arms"]["b"]["binary"]["A"]
    assert bb["flag_recall"]["value"] == pytest.approx(0.8)
    assert bb["fpr"]["value"] == pytest.approx(1 / 3)

    # urn covers 7 of 8 rows (c7 missing); pass rows covered are c1 (3.0) and c2 (1.0).
    assert c["urn"]["n_covered_rows"] == 7 and c["urn"]["n_covered_claims"] == 6
    matched = c["urn"]["models"]["3-voice"]["matched"]["a"]["A"]
    # only two covered pass rows, so the achievable FPRs are 0, 0.5, 1. A 1/3 budget forces
    # 0, and the largest threshold with no pass row flagged is 0.5 (c6's score).
    assert matched["threshold"] == pytest.approx(0.5)
    assert matched["fpr"]["value"] == pytest.approx(0.0)
    assert matched["flag_recall"]["value"] == pytest.approx(1.0)
    fitted = c["urn"]["models"]["3-voice"]["fitted"]["A"]
    assert fitted["threshold"] == pytest.approx(-2.5)
    assert fitted["flag_recall"]["value"] == pytest.approx(0.4)  # only the two c5 rows

    # arm a vs arm b on binary A differ on c3 and c7 (a right) and c2 and c6 (b right).
    mc = c["mcnemar"]["a vs b on binary A"]
    assert (mc["a_only_correct"], mc["b_only_correct"], mc["n_pairs"]) == (2, 2, 8)
    assert "a vs urn 3-voice at matched FPR on binary A" in c["mcnemar"]
    assert c["mcnemar"]["a vs urn 3-voice at matched FPR on binary A"]["n_pairs"] == 7

    # slices are keyed as pre-committed, and the CI is a real interval around the value.
    assert set(c["slices"]) == {"gold_class", "has_original_claim_url", "claim_month"}
    assert set(c["slices"]["claim_month"]) == {"2020-10"}
    assert c["slices"]["gold_class"]["Supported"]["systems"]["a"]["n"] == 3
    lo, hi = ba["accuracy"]["ci"]
    assert lo <= ba["accuracy"]["value"] <= hi

    # speed and cost: uncached latency ignores the cached 99 s call, exa is not instrumented.
    sc = c["arms"]["a"]["speed_cost"]
    assert sc["llm_latency_s_p50_uncached"] == pytest.approx(2.0)
    assert sc["exa_calls_per_claim"] == pytest.approx(1.0)
    assert c["arms"]["a"]["ceiling"]["exa"]["date_instrumented_searches"] == 0
    assert c["arms"]["a"]["ceiling"]["serper"]["share_hits_post_ceiling"] == pytest.approx(0.2)
