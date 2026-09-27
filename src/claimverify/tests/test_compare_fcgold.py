"""Offline fc-gold comparison test: a 42-row synthetic urn table, 40 of them covered by a
synthetic loop run whose A/B/R recall and FPR are hand-computable, with urn scores laid out so
the matched-FPR thresholds and the recall they buy are known exactly."""
import json

import numpy as np
import pandas as pd
import pytest

from claimverify.compare_fcgold import compare, markdown
from claimverify.run_fcgold import claim_id_of

# gold FALSE: 12 refuted, 4 unsupported, 2 conflicting, 2 supported
FALSE_STATUS = ["refuted"] * 12 + ["unsupported"] * 4 + ["conflicting"] * 2 + ["supported"] * 2
# gold TRUE: 12 supported, 4 conflicting, 2 unsupported, 2 refuted
TRUE_STATUS = ["supported"] * 12 + ["conflicting"] * 4 + ["unsupported"] * 2 + ["refuted"] * 2
# true claims score 1..20; false claims score -9..0 and 0.5..9.5, so they interleave
TRUE_SCORE = [float(i) for i in range(1, 21)]
FALSE_SCORE = [float(i) for i in range(-9, 1)] + [i + 0.5 for i in range(10)]
RAW = {"supported": "Supported", "refuted": "Refuted",
       "unsupported": "Not Enough Evidence", "conflicting": "Conflicting Evidence/Cherrypicking"}


def _rows():
    rows = []
    for i, (st, sc) in enumerate(zip(FALSE_STATUS, FALSE_SCORE)):
        rows.append((f"https://fc.test/false/{i}", False, st, sc,
                     "unprovable" if i == 0 else "clear_false"))
    for i, (st, sc) in enumerate(zip(TRUE_STATUS, TRUE_SCORE)):
        rows.append((f"https://fc.test/true/{i}", True, st, sc, "clear_true"))
    return rows


def _build(tmp_path):
    rows = _rows()
    # two extra population rows the loop never covers: one failed record, one absent
    extra = [("https://fc.test/uncovered/failed", False, None, -5.0, "clear_false"),
             ("https://fc.test/uncovered/absent", True, None, 5.0, "clear_true")]
    pop = pd.DataFrame([{"review_url": u, "gold_true": g, "mid": False, "fold": i % 5,
                         "subtype": sub, "n_docs": 5, "score_7flag": s, "score_3voice": s}
                        for i, (u, g, _st, s, sub) in enumerate(rows + extra)])
    urn = tmp_path / "oof_scores.parquet"
    pop.to_parquet(urn)

    run = tmp_path / "run"
    (run / "trace").mkdir(parents=True)
    with (run / "shard-00.jsonl").open("w") as f:
        for i, (u, _g, st, _s, _sub) in enumerate(rows):
            # unsupported rows are i = 12..15 (gold false) and 36, 37 (gold true):
            # 12, 13 refused at the bar; 15, 36 ended with no evidence; 14, 37 are neither
            guards = [{"round": 1, "guard": "close-below-bar", "refused": "refuted"}] \
                if i in (12, 13) else []
            if i == 14:
                guards = [{"round": 1, "guard": "fc-undated-dropped", "url": "x"}]
            ev = [] if i in (15, 36) else [{"src": "D1"}]
            f.write(json.dumps({"claim_id": claim_id_of(u), "ok": True, "result": {
                "claim_id": claim_id_of(u), "status": st, "verdict_raw": RAW[st],
                "guard_events": guards, "evidence": ev,
                "stopped": "exa-final" if i % 4 == 0 else "resolved",
                "llm_calls": 4, "serper_calls": 2, "exa_calls": 1, "cost_usd": 0.002}}) + "\n")
        f.write(json.dumps({"claim_id": claim_id_of(extra[0][0]), "ok": False,
                            "error": "RuntimeError: boom"}) + "\n")
    (run / "trace" / "t.json").write_text(json.dumps({"searches": [
        {"provider": "serper", "n_results": 10, "post_ceiling_hits": 2, "undated_hits": 0},
        {"provider": "serper", "n_results": 10, "post_ceiling_hits": 0, "undated_hits": 0}]}))
    return run, urn, tmp_path / "no-ladder.json"


def test_compare_fcgold(tmp_path):
    run, urn, ladder = _build(tmp_path)
    c = compare(run, urn, ladder, reps=200, seed=0)

    cov = c["coverage"]
    assert (cov["n_population"], cov["n_loop_records"], cov["n_loop_ok"], cov["n_loop_failed"],
            cov["n_joined"], cov["n_missing"]) == (42, 41, 40, 1, 40, 2)
    assert cov["missing_review_urls_first10"] == ["https://fc.test/uncovered/failed",
                                                  "https://fc.test/uncovered/absent"]
    assert (c["gold"]["n_flag_worthy"], c["gold"]["n_pass_worthy"]) == (20, 20)
    assert c["gold"]["n_unprovable_veracity3"] == 1

    # A flags all but supported: 18/20 false, 8/20 true. B lets conflicting pass: 16 and 4.
    # R flags only refuted: 12 and 2.
    expect = {"A": (0.90, 0.40), "B": (0.80, 0.20), "R": (0.60, 0.10)}
    for name, (rec, fpr) in expect.items():
        p = c["loop_points"][name]
        assert p["flag_recall"]["value"] == rec
        assert p["fpr"]["value"] == fpr
        assert p["flag_recall"]["ci"][0] <= rec <= p["flag_recall"]["ci"][1]
    assert c["loop_points"]["A"]["accuracy"]["value"] == (18 + 12) / 40

    # matched thresholds: the largest score whose FPR stays at or below the loop's
    matched = {"A": (8.5, 19 / 20), "B": (4.5, 15 / 20), "R": (2.5, 13 / 20)}
    for model in ("7-flag", "3-voice"):
        m = c["urn"]["models"][model]
        assert m["fitted"]["threshold"] == {"7-flag": -3.9536, "3-voice": -4.6256}[model]
        for name, (thr, rec) in matched.items():
            q = m["matched"][name]
            assert q["threshold"] == thr
            assert q["flag_recall"]["value"] == rec
            assert q["fpr"]["value"] <= expect[name][1]
            assert q["recall_diff_loop_minus_urn"]["value"] == expect[name][0] - rec
            assert q["mcnemar_on_gold_false"]["n_pairs"] == 20
        assert 0.0 <= m["auc"]["value"] <= 1.0
        assert len(m["roc"]["threshold"]) == len(m["roc"]["fpr"]) == len(m["roc"]["recall"])
        assert m["roc"]["fpr"][0] == 0.0 and m["roc"]["recall"][0] == 0.0
        assert m["roc"]["fpr"][-1] == 1.0 and m["roc"]["recall"][-1] == 1.0

    a = c["anatomy"]
    assert a["status_by_gold"]["refuted"] == {"gold_false": 12, "gold_true": 2}
    assert a["status_by_gold"]["unsupported"] == {"gold_false": 4, "gold_true": 2}
    assert a["unsupported_breakdown"] == {"bar_refusal": 2, "retrieval_miss_zero_evidence": 2,
                                          "other": 2}
    assert a["fc_undated_dropped"] == {"events": 1, "claims": 1}
    assert a["per_claim"]["llm_calls"] == 4.0 and a["per_claim"]["cost_usd"] == pytest.approx(0.002)
    assert a["share_stopped_exa_final"] == 10 / 40
    assert a["ceiling"]["serper"]["share_searches_with_post_ceiling_hit"] == 0.5

    sf = c["sign_flips"]
    assert len(sf["gold_true_refuted"]) == 2 and len(sf["gold_false_supported"]) == 2
    assert sf["gold_false_supported"][0]["review_url"].startswith("https://fc.test/false/")

    md = markdown(c)
    assert "| A (pass = supported) |" in md and "ROC AUC for 7-flag" in md
    assert "0.900 [" in md
    json.dumps(c, default=float)


def test_matched_point_is_paired_and_json_clean(tmp_path):
    """The urn's matched FPR never exceeds the loop's, and the ROC has one row per distinct
    score plus the empty-flag endpoint."""
    run, urn, ladder = _build(tmp_path)
    c = compare(run, urn, ladder, reps=50, seed=1)
    scores = pd.read_parquet(urn)["score_7flag"].to_numpy()
    n_distinct = len(np.unique(scores[np.isin(
        pd.read_parquet(urn)["review_url"], [r[0] for r in _rows()])]))
    assert len(c["urn"]["models"]["7-flag"]["roc"]["threshold"]) == n_distinct + 1
    assert all(np.isfinite(c["urn"]["models"]["7-flag"]["roc"]["threshold"]))
