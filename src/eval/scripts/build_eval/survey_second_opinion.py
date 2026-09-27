"""Second opinion on every survey-pool claim (Daniel 2026-09-22).

One call per claim on DeepSeek-V4-Pro: the post, the claim, and the evidence the instrument
retrieved (each source's domain, date and the sentences the reader cited), asking for a
verdict and a short written justification. The model is BLIND to the instrument's score and to
the reader's per-source codes, so agreement with the instrument is measured, not induced.
This is guidance for the collaborators (the page shows it next to the instrument's verdict),
not a replacement for the instrument. Abstract principles only in the prompt.

  uv run python -m eval.scripts.build_eval.survey_second_opinion --scored <parquet> --dossiers <jsonl> --out <json>
"""
from __future__ import annotations
import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from deepinfra_runner import run  # noqa: E402
from eval.scripts.build_eval.survey_claim_check import _parse_json, prompt_hash, PROVIDER  # noqa: E402

SECOND_SYS = """\
You give an independent second opinion on whether a factual claim taken from a social media post
is true. You are given the post, the claim, the date the post was made, and the evidence an
automated system retrieved for the claim: for each source, its domain, its publication date where
known, and the sentences a reader singled out as relevant. You have no other knowledge of what the
system concluded.

Judge the claim AS STATED. A source that confirms an adjacent or weaker fact does not confirm the
claim. A quotation or restatement of the claim in a source is not evidence for it. Weigh sources
by how directly they bear on the claim and how reliable they are; several sources echoing one
another count once. A claim that presents an old event as current is judged on what it asserts
about timing. When the evidence is thin or off topic, say so rather than guessing.

Return JSON only:
{"verdict": "true" | "false" | "unclear",
 "confidence": "high" | "medium" | "low",
 "justification": "<two to four plain sentences a non-specialist can follow, naming the decisive source or the gap>"}
"""
TAG = "second-opinion-v1"
MAX_CITED, CITE_CHARS = 6, 300


def evidence_block(docs: list[dict]) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        rd = d.get("read") or {}
        if d.get("read_status") != "prepped":
            continue
        pos = {sid: k for k, sid in enumerate(d.get("sent_ids") or [])}
        sents = d.get("sents") or []
        cited = [html.unescape(sents[pos[s]])[:CITE_CHARS] for s in (rd.get("evidence") or [])[:MAX_CITED]
                 if s in pos and pos[s] < len(sents)]
        if not cited:
            continue
        out.append(f"[{i}] {d.get('domain') or ''}" + (f", {d['date']}" if d.get("date") else "") + "\n" + "\n".join(f"  - {c}" for c in cited))
    return "\n".join(out) if out else "(no source had relevant sentences)"


def build_request(rec, post_text):
    user = (f"POST DATE: {rec.get('post_date') or 'unknown'}\nPOST:\n{html.unescape(post_text)}\n\n"
            f"CLAIM: {html.unescape(rec['claim'])}\n\nEVIDENCE:\n{evidence_block(rec.get('docs') or [])}")
    return {"temperature": 0, "max_tokens": 500, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SECOND_SYS}, {"role": "user", "content": user}]}


def parse_response(j):
    obj = _parse_json(j["choices"][0]["message"].get("content"))
    return {"verdict": obj.get("verdict") if obj.get("verdict") in ("true", "false", "unclear") else "unclear",
            "confidence": obj.get("confidence") if obj.get("confidence") in ("high", "medium", "low") else "low",
            "justification": str(obj.get("justification") or "")[:900]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True, help="scored pool parquet (claim_id, post_id, text, flag)")
    ap.add_argument("--dossiers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--provider", default="deepinfra")
    ap.add_argument("--workers", type=int, default=None, help="None = the runner's per-model default")
    ap.add_argument("--cap", type=float, default=8.0)
    a = ap.parse_args()
    PROVIDER[0] = a.provider
    p = pd.read_parquet(a.scored)
    p["post_id"] = p.post_id.astype(str)
    posts = p.drop_duplicates("post_id").set_index("post_id")["text"].to_dict()
    flags = p.set_index("claim_id")["flag"].to_dict()
    recs = {}
    for line in a.dossiers.open():
        r = json.loads(line)
        if r["claim_id"] in flags:
            recs[r["claim_id"]] = r
    print(f"claims {len(recs)}  prompt {TAG} ({prompt_hash(SECOND_SYS)})  model {a.model}", flush=True)
    rr = run(list(recs), lambda cid: build_request(recs[cid], posts.get(str(recs[cid]["post_id"]), "")),
             model=a.model, workers=a.workers, cache_path=str(a.out) + ".cache.jsonl",
             item_id=lambda cid: f"{prompt_hash(SECOND_SYS)}|{cid}", parse=lambda j, cid: parse_response(j),
             max_spend=a.cap)
    print(rr.summary(), flush=True)
    spent = [rr.spent]
    results = {}
    for cid in recs:
        res = rr.results.get(f"{prompt_hash(SECOND_SYS)}|{cid}")
        results[cid] = dict(res) if res else {"verdict": "unclear", "confidence": "low",
                                              "justification": f"failed {str(rr.errors.get(f'{prompt_hash(SECOND_SYS)}|{cid}', ''))[:80]}"}
        results[cid]["agrees"] = (None if res is None or res["verdict"] == "unclear"
                                  else res["verdict"] == ("false" if flags[cid] else "true"))
    json.dump({"prompt": TAG, "prompt_hash": prompt_hash(SECOND_SYS), "model": a.model,
               "system_prompt": SECOND_SYS, "claims": results}, open(a.out, "w"), indent=0, ensure_ascii=False)
    n = len(results)
    v = pd.Series({c: r["verdict"] for c, r in results.items()})
    ag = pd.Series({c: r["agrees"] for c, r in results.items()})
    print(f"spent ${spent[0]:.3f}  verdicts {v.value_counts().to_dict()}  agree {int((ag == True).sum())}  "
          f"disagree {int((ag == False).sum())}  unclear {int(ag.isna().sum())}  of {n}  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
