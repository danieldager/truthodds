"""Build a full read-v6.1 reads file over the frozen 3,236.

Union (rep1500 u clean1500 = 2,274) genuine v6.1 already lives in
results-v6.1-frozen3280.jsonl. The 962 missing claims get their genuine v6.1
from the reader_lab missing962 run (docs[].new), rebuilt from results-00 so
ranks/metadata are self-consistent. The 44 claim-shape claims (outside 3,236)
keep their frozen3280 (v5-folded) reads; they are never scored.
"""
import argparse, json, sys
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", required=True, help="scratchpad dir with intermediate inputs (missing_ids.json)")
SCR = Path(_ap.parse_args().out)
E1V61 = Path("eval/data/urn_runs/e1_ctx_v61")
FROZEN_V61 = E1V61 / "results-v6.1-frozen3280.jsonl"
RESULTS00 = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
RL_OUT = E1V61 / "missing962/v6.1__DeepSeek-V4-Flash.jsonl"
OUT = E1V61 / "results-v6.1-full3236.jsonl"

missing = set(json.load(open(SCR / "missing_ids.json")))

# reader_lab output: per claim, {rank: new_direction}
rl = {}
for line in open(RL_OUT):
    d = json.loads(line)
    rl[d["claim_id"]] = {doc["rank"]: doc["new"] for doc in d.get("docs", [])}
print("reader_lab claims:", len(rl), "should be 962")
assert missing <= set(rl), f"{len(missing - set(rl))} missing claims absent from reader_lab output"

# results-00 metadata + docs for the missing claims
r00 = {}
for line in open(RESULTS00):
    d = json.loads(line)
    if d.get("review_url") in missing:
        r00[d["review_url"]] = d

# frozen3280 records (metadata + genuine v6.1 for union)
n_written = 0; n_missing_rebuilt = 0; changed_docs = 0; kept_docs = 0
with open(OUT, "w") as out:
    for line in open(FROZEN_V61):
        rec = json.loads(line)
        url = rec["review_url"]
        if url in missing:
            src = r00[url]
            newres = []
            rlmap = rl[url]
            for doc in src.get("results") or []:
                dr = (doc.get("read") or {}).get("direction")
                if dr is None:
                    continue
                nd = rlmap.get(doc["rank"], dr)   # v6.1 read, else preserve old (unreadable doc)
                if nd != dr: changed_docs += 1
                else: kept_docs += 1
                newres.append({"rank": doc["rank"], "domain": doc.get("domain"),
                               "read": {"direction": nd}})
            rec = {k: src.get(k) for k in ("review_url","veracity","publisher_site","ceiling_src",
                                           "screen_verdict","claim_resolved","claim_text","post_id",
                                           "frame","topic")}
            rec["results"] = newres
            n_missing_rebuilt += 1
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_written += 1
print(f"wrote {n_written} records to {OUT}")
print(f"missing rebuilt: {n_missing_rebuilt}  (docs changed vs v5 {changed_docs}, preserved {kept_docs})")
