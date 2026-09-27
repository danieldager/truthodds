"""Offline scoring test: a synthetic run dir + gold parquet, with a duplicate gold row
(must inherit its claim's prediction), a failed record and a claim with no record."""
import json

import pandas as pd
import pytest

from claimverify.score_averitec import score_run

CONF = "Conflicting Evidence/Cherrypicking"
NEE = "Not Enough Evidence"
RAW = {"supported": "Supported", "refuted": "Refuted", "unsupported": NEE, "conflicting": CONF}

# claim_id, gold, system status (None = no record, "FAIL" = ok=False record)
ROWS = [
    ("c1", "Supported", "supported"),
    ("c2", "Refuted", "unsupported"),     # raw wrong (NEE), convention right (Refuted)
    ("c3", CONF, "supported"),            # wrong under both; passes in binary A, flag in gold
    ("c4", NEE, "unsupported"),           # raw right, convention wrong
    ("c5", "Refuted", "refuted"),
    ("c5", "Refuted", "refuted"),         # duplicate gold row, inherits c5
    ("c8", "Refuted", "unsupported"),     # raw wrong, convention right
    ("c6", "Supported", None),            # missing
    ("c7", "Supported", "FAIL"),          # failed record
]


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "shard-00.jsonl").open("w") as f:
        for i, (cid, _, st) in enumerate(ROWS):
            if st is None or cid == "c5" and i == 5:
                continue
            if st == "FAIL":
                rec = {"claim_id": cid, "ok": False, "error": "RuntimeError: boom", "elapsed_s": 1.0}
            else:
                rec = {"claim_id": cid, "ok": True, "elapsed_s": 10.0 * (i + 1), "result": {
                    "claim_id": cid, "status": st, "verdict_raw": RAW[st],
                    "verdict_cc": "Refuted" if st == "unsupported" else RAW[st],
                    "elapsed_s": 10.0 * (i + 1), "llm_calls": 3, "llm_cache_hits": 1,
                    "serper_calls": 2, "exa_calls": 0, "scrapes": 4,
                    "tokens": {"prompt": 100, "cached": 10, "completion": 20}, "cost_usd": 0.001}}
            f.write(json.dumps(rec) + "\n")
    gold = pd.DataFrame({"claim_id": [r[0] for r in ROWS], "gold_label": [r[1] for r in ROWS],
                         "claim_text": "x"})
    gold_path = tmp_path / "gold.parquet"
    gold.to_parquet(gold_path, index=False)
    return run_dir, gold_path


def test_score(run):
    df, m = score_run(*run)
    assert len(df) == 9 and m["n_gold_claims"] == 8
    assert m["n_scored_rows"] == 7
    assert m["missing_claims"] == ["c6", "c7"]
    assert list(m["failed_claims"]) == ["c7"]
    # duplicate inherits
    dup = df[df["claim_id"] == "c5"]
    assert len(dup) == 2 and dup["duplicate"].tolist() == [False, True]
    assert dup["verdict_raw"].tolist() == ["Refuted", "Refuted"]
    # raw vs convention
    raw, cc = m["four_class"]["raw"], m["four_class"]["claimcheck_convention"]
    assert raw["n"] == cc["n"] == 7
    assert raw["accuracy"] == pytest.approx(4 / 7)
    assert cc["accuracy"] == pytest.approx(5 / 7)
    assert raw["confusion"]["Refuted"][NEE] == 2 and cc["confusion"]["Refuted"]["Refuted"] == 4
    assert raw["confusion"][NEE][NEE] == 1 and cc["confusion"][NEE]["Refuted"] == 1
    # raw: F1 Sup 2/3, Ref 2/3, NEE 1/2, Conf 0; convention: Sup 2/3, Ref 8/9, NEE 0, Conf 0
    assert raw["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1 / 2) / 4)
    assert cc["macro_f1"] == pytest.approx((2 / 3 + 8 / 9) / 4)
    # binary variants
    a, b, c = (m["binary"][k] for k in "ABC")
    assert a["n"] == 7 and a["n_gold_flag"] == 6 and a["accuracy"] == pytest.approx(6 / 7)
    assert a["flag_recall"] == pytest.approx(5 / 6) and a["fpr"] == 0.0 and a["precision"] == 1.0
    assert b["n"] == 7 and b["n_gold_flag"] == 5 and b["accuracy"] == 1.0
    assert c["n"] == 6 and c["n_gold_flag"] == 5 and c["accuracy"] == 1.0
    assert df["bin_C_gold"].isna().sum() == 1     # the dropped Conflicting row
    # speed / cost over unique claims
    sc = m["speed_cost"]
    assert sc["n_claims"] == 6 and sc["llm_calls"] == 18 and sc["scrapes"] == 24
    assert sc["cost_usd"] == pytest.approx(0.006) and sc["tokens"]["prompt"] == 600
    assert sc["cost_by_source"] == {}                # no trace/ in the synthetic run
