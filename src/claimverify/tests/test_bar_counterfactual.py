"""Offline bar-counterfactual test: a synthetic two-claim run dir with one claim the bar
refused and one it closed, plus a duplicate gold row that must inherit its claim."""
import json

import pandas as pd
import pytest

from claimverify.bar_counterfactual import analyse_arm
from claimverify.compare_averitec import boot_index

# c1: the model proposed supported, the bar refused, the loop recorded unsupported.
# c2: closed refuted on a fact-check page published after the claim.
RECORDS = {
    "c1": {"status": "unsupported", "date": "2020-01-10",
           "guard_events": [{"round": 2, "guard": "close-below-bar", "claim_id": 1,
                             "refused": "supported", "detail": "single reliable voice"}],
           "rounds": [{"round": 1, "docs": [{"id": "D1", "url": "https://news.example.com/a",
                                             "date": "2020-01-02"}]}],
           "evidence": [{"src": "D1", "stance": "supports", "snippet_only": False},
                        {"src": "D1", "stance": "neutral", "snippet_only": False}]},
    "c2": {"status": "refuted", "date": "2020-02-01",
           "guard_events": [{"round": 1, "guard": "budget-exhausted"}],
           "rounds": [{"round": 1, "docs": [{"id": "D1", "url": "https://www.snopes.com/x",
                                             "date": "2020-03-04"}]}],
           "evidence": [{"src": "D1", "stance": "refutes", "snippet_only": False},
                        {"src": "D1", "stance": "refutes", "snippet_only": True}]},
}
LEDGERS = {
    "c1": [{"round": 1, "proposed": [{"claim_id": 1, "status": "open"}]},
           {"round": 2, "proposed": [{"claim_id": 1, "status": "supported"}]}],
    "c2": [{"round": 1, "proposed": [{"claim_id": 1, "status": "refuted"}]}],
}
GOLD = pd.DataFrame({"claim_id": ["c1", "c2", "c2"],
                     "gold_label": ["Supported", "Refuted", "Refuted"]})


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "arm"
    (d / "trace").mkdir(parents=True)
    with (d / "shard-00.jsonl").open("w") as f:
        for cid, res in RECORDS.items():
            f.write(json.dumps({"claim_id": cid, "ok": True, "elapsed_s": 1.0,
                                "result": {"claim_id": cid, **res}}) + "\n")
    for cid, led in LEDGERS.items():
        (d / "trace" / f"{cid}.json").write_text(json.dumps({"claim_id": cid, "ledger": led}))
    return d


def test_analyse_arm(run):
    out = analyse_arm(run, GOLD, boot_index(len(GOLD), 50, 0))

    assert out["n_claims"] == 2
    assert out["actual"]["n_rows"] == 3 and out["actual"]["n_rows_missing"] == 0

    # c1 counts as Refuted under the ClaimCheck convention, so only the two c2 rows are right
    assert out["actual"]["four_class_convention"]["value"] == pytest.approx(2 / 3)
    # with no bar c1 becomes the supported the model proposed, and every row is right
    assert out["cf_a_no_bar"]["four_class_convention"]["value"] == pytest.approx(1.0)
    assert out["cf_a_flips"] == {"unsupported -> supported": 1}
    assert out["n_flipped_by_cf_a"] == 1

    # binary A, c1 is the only gold pass row
    assert out["actual"]["binary_A"]["fpr"]["value"] == pytest.approx(1.0)
    assert out["cf_a_no_bar"]["binary_A"]["fpr"]["value"] == pytest.approx(0.0)
    assert out["cf_a_no_bar"]["binary_A"]["flag_recall"]["value"] == pytest.approx(1.0)
    lo, hi = out["cf_a_no_bar"]["four_class_convention"]["ci"]
    assert lo <= 1.0 <= hi

    an = out["unsupported_anatomy"]
    assert (an["n_unsupported"], an["bar_refusal"], an["retrieval_miss"], an["other"]) == (1, 1, 0, 0)

    # one distinct stance-bearing full-read doc per claim, c2's is a fact-check page dated after
    lk = out["leak"]
    assert lk["stance_docs"] == 2
    assert lk["fact_check_host"] == 1 and lk["fact_check_share"] == pytest.approx(0.5)
    assert lk["post_claim"] == 1 and lk["post_claim_share"] == pytest.approx(0.5)
