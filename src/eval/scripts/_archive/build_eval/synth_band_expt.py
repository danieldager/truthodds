"""Banded synthesis experiment: 4 populations x 3 s7 bands, 4 synthesis arms.

Expansion of the precision band audit (Daniel 2026-08-27). The flags place each
claim in a band; the experiment measures how much each escalation step (better
prompt, smarter model, Exa retrieval, unsure->Exa escalation) improves the
verdicts per band, on a fixed stratified sample with labels where they exist.

    uv run python -m eval.scripts.build_eval.synth_band_expt --sample
    uv run python -m eval.scripts.build_eval.synth_band_expt --arm serper_v1
    uv run python -m eval.scripts.build_eval.synth_band_expt --arm exa_v1 --smoke 10
    uv run python -m eval.scripts.build_eval.synth_band_expt --report
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval.evidence_urn_run import llm, _clean, select_sentences  # noqa: E402
from eval.scripts.build_eval.tl_band_audit import SYNTH_SYS  # noqa: E402
from eval.scripts.build_eval.fit_urn import load_judged_axis, MEDIA_AXIS  # noqa: E402
from eval.scripts.build_eval.fit_two_urn import load_urn  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/synth_expt"
SAMPLE = OUT_DIR / "sample.jsonl"
FIT = SRC / "eval/data/urn_runs/true_timeline/two_urn_fit.json"
C2 = SRC / "eval/data/urn_runs/c2_false"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
GOLD_SYNTH = SRC / "eval/data/urn_runs/e1_ctx/synth_verdicts.jsonl"
TL_SYNTH = SRC / "eval/data/urn_runs/true_timeline/band_audit.jsonl"
SEED = 20260827
PER_CELL = 40
EPS = 0.10
PRO_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

# v2 = v1 + retrieval-failure guard + attribution guard. Nothing else moved, so
# any delta is attributable to these two lines.
SYNTH_SYS_V2 = SYNTH_SYS.replace(
    "- Absence of coverage is weak evidence either way for niche topics, but telling for claims "
    "that would be major news if true.",
    "- Before judging, ask whether the dossiers actually engage the claim's topic. If most are "
    "off-topic or merely adjacent, the search failed; return \"unsure\", not \"false\". Treat "
    "absence of coverage as evidence of falsity ONLY when the dossiers show the topic itself was "
    "well covered and the specific claimed event is missing from that coverage.\n"
    "- If the claim reports that a person or outlet SAID something, the question is whether they "
    "said it, not whether what they said is true.")
assert SYNTH_SYS_V2 != SYNTH_SYS

ARMS = {
    "serper_v1": {"docs": "serper", "sys": SYNTH_SYS, "model": None},
    "serper_v2": {"docs": "serper", "sys": SYNTH_SYS_V2, "model": None},
    "serper_pro": {"docs": "serper", "sys": SYNTH_SYS, "model": PRO_MODEL},
    "exa_v1": {"docs": "exa", "sys": SYNTH_SYS, "model": None},
    "serper_qwen": {"docs": "serper", "sys": SYNTH_SYS,
                    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507"},
    "exa_pro": {"docs": "exa", "sys": SYNTH_SYS, "model": PRO_MODEL},
}


def band_of(s: float) -> str:
    return "flag" if s <= -4.05 else ("cliff" if s <= -2.0 else "pass")


def w7() -> dict:
    for f in json.loads(FIT.read_text())["fits7"]:
        if abs(f["eps"] - EPS) < 1e-9 and f.get("weights"):
            return f["weights"]
    raise SystemExit("no eps=0.10 fit")


def doss_of(rec: dict) -> list[dict]:
    return [{"domain": d.get("domain"), "date": d.get("date"),
             "sents": (d.get("sents") or [])[:12]}
            for d in rec.get("results") or []
            if d.get("read") and d["read"].get("direction")]


def build_sample() -> None:
    w = w7()
    pools = collections.defaultdict(list)

    def add(pop, rec, veracity=None):
        flags = [d["read"]["direction"] for d in rec.get("results") or []
                 if d.get("read") and d["read"].get("direction")]
        if not flags:
            return
        s = sum(w.get(f, 0.0) for f in flags)
        pools[(pop, band_of(s))].append({
            "review_url": rec["review_url"], "population": pop,
            "band": band_of(s), "s7": round(s, 4), "veracity": veracity,
            "claim": rec.get("claim_resolved") or rec.get("claim_text") or "",
            "ceiling": rec.get("ceiling"), "dossiers": doss_of(rec)})

    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            if (r.get("claim_text") or "")[:80] in excl:
                continue
            add("cn_false", r)
    axis = load_judged_axis()
    for line in E1.open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        if (axis.get(r["review_url"]) or "untagged") == MEDIA_AXIS:
            continue
        add("gold_false" if v <= 3 else "gold_true", r, veracity=v)
    for line in TL.open():
        add("timeline", json.loads(line))

    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for (pop, band), rows in sorted(pools.items()):
            rng.shuffle(rows)
            take = rows[:PER_CELL]
            print(f"  {pop:<11} {band:<6} pool {len(rows):>5} -> {len(take)}")
            for r in take:
                f.write(json.dumps(r) + "\n")


def dossier_text(dossiers: list[dict]) -> str:
    parts = []
    for i, d in enumerate(dossiers, 1):
        sents = " ".join(d.get("sents") or [])[:900]
        parts.append(f"[{i}] {d.get('domain')} ({d.get('date') or 'undated'})\n{sents}")
    return "\n\n".join(parts)


def synth_call(row: dict, dossiers: list[dict], sys_prompt: str, model: str | None) -> dict:
    user = (f"CLAIM: {row['claim']}\n"
            f"CLAIM DATE: {row.get('ceiling') or 'unknown'}\n\n"
            f"DOSSIERS:\n{dossier_text(dossiers)}")
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
    if model is None:
        obj, cost, _, _ = llm(msgs, cache_key="synth-expt", max_tokens=300)
    else:
        import requests
        from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
        r = requests.post(f"{EXTRACTION_BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
                          json={"model": model, "temperature": 0, "max_tokens": 900,
                                "response_format": {"type": "json_object"}, "messages": msgs},
                          timeout=120)
        r.raise_for_status()
        j = r.json()
        obj = json.loads(j["choices"][0]["message"]["content"] or "{}")
        cost = (j.get("usage") or {}).get("estimated_cost") or 0.0
    v = obj.get("verdict")
    return {"verdict": v if v in ("true", "false", "unsure") else "unsure",
            "confidence": obj.get("confidence"), "reason": obj.get("reason", ""),
            "key_evidence": obj.get("key_evidence", []), "cost": cost}


def exa_dossiers(row: dict) -> list[dict]:
    from pipeline.search import _fetch_exa
    res = _fetch_exa(row["claim"], 10, row.get("ceiling"), ["x.com"], 1)
    out = []
    for r in res:
        text = _clean(r.get("content") or r.get("snippet") or "")
        if not text:
            continue
        ids, sents, _ = select_sentences(text, row["claim"])
        if sents:
            dom = r["url"].split("/")[2].removeprefix("www.") if "://" in r["url"] else r["url"]
            out.append({"domain": dom, "date": r.get("date"), "sents": sents[:12]})
    return out


def reuse_verdicts() -> dict:
    out = {}
    if GOLD_SYNTH.exists():
        for line in GOLD_SYNTH.open():
            r = json.loads(line)
            out[r["review_url"]] = r
    if TL_SYNTH.exists():
        for line in TL_SYNTH.open():
            r = json.loads(line)
            out[r["review_url"]] = r
    return out


def run_arm(arm: str, smoke: int, workers: int, only_unsure_of: str | None = None) -> None:
    cfg = ARMS[arm]
    rows = [json.loads(l) for l in SAMPLE.open()]
    out_path = OUT_DIR / f"arm_{arm}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["review_url"] for l in out_path.open()}
    todo = [r for r in rows if r["review_url"] not in done]
    if only_unsure_of:
        src = {json.loads(l)["review_url"]: json.loads(l)["verdict"]
               for l in (OUT_DIR / f"arm_{only_unsure_of}.jsonl").open()}
        todo = [r for r in todo if src.get(r["review_url"]) == "unsure"]

    reused = 0
    if arm == "serper_v1":          # verdicts already computed by earlier runs
        prior = reuse_verdicts()
        with out_path.open("a") as f:
            for r in list(todo):
                p = prior.get(r["review_url"])
                if p:
                    f.write(json.dumps({"review_url": r["review_url"],
                                        "verdict": p["verdict"], "confidence": p.get("confidence"),
                                        "reason": p.get("reason", ""), "cost": 0.0,
                                        "reused": True}) + "\n")
                    todo.remove(r)
                    reused += 1
        print(f"reused {reused} prior verdicts")
    if smoke:
        todo = todo[:smoke]
    print(f"{arm}: todo {len(todo)}", flush=True)

    lock = threading.Lock()
    state = {"n": 0, "cost": 0.0}
    t0 = time.time()

    def work(row):
        doss = row["dossiers"] if cfg["docs"] == "serper" else exa_dossiers(row)
        if not doss:
            return {"review_url": row["review_url"], "verdict": "unsure",
                    "confidence": None, "reason": "no dossiers retrieved",
                    "n_docs": 0, "cost": 0.0}
        out = synth_call(row, doss, cfg["sys"], cfg["model"])
        out["review_url"] = row["review_url"]
        out["n_docs"] = len(doss)
        if cfg["docs"] == "exa":
            out["exa_domains"] = [d["domain"] for d in doss]
            out["dossiers"] = doss
        return out

    n_workers = workers
    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in as_completed(futs):
            o = fut.result()
            with lock:
                state["n"] += 1
                state["cost"] += o["cost"]
                f.write(json.dumps(o) + "\n")
                f.flush()
                if state["n"] % 40 == 0 or smoke:
                    el = (time.time() - t0) / 60
                    print(f"  {state['n']}/{len(todo)} | ${state['cost']:.3f} | {el:.1f}m",
                          flush=True)
    print(f"{arm} done {state['n']} | ${state['cost']:.3f}", flush=True)


def report() -> None:
    rows = {json.loads(l)["review_url"]: json.loads(l) for l in SAMPLE.open()}
    arms = {}
    for arm in ARMS:
        p = OUT_DIR / f"arm_{arm}.jsonl"
        if p.exists():
            arms[arm] = {json.loads(l)["review_url"]: json.loads(l) for l in p.open()}
    esc = {}
    if "serper_v1" in arms and "exa_v1" in arms:
        for u, s in arms["serper_v1"].items():
            e = arms["exa_v1"].get(u)
            esc[u] = e if (s["verdict"] == "unsure" and e) else s
        arms["escalation"] = esc

    for arm, verd in arms.items():
        print(f"\n== {arm} (n={len(verd)}) ==")
        print(f"  {'population':<11} {'band':<6} {'n':>3}  {'false':>6} {'unsure':>7} {'true':>6}")
        agg = collections.defaultdict(collections.Counter)
        for u, v in verd.items():
            r = rows.get(u)
            if r:
                agg[(r["population"], r["band"])][v["verdict"]] += 1
        for (pop, band) in sorted(agg):
            c = agg[(pop, band)]
            n = sum(c.values())
            print(f"  {pop:<11} {band:<6} {n:>3}  {c['false']/n:>6.0%} {c['unsure']/n:>7.0%} "
                  f"{c['true']/n:>6.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only-unsure-of", default=None,
                    help="restrict to claims this arm judged unsure")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.sample:
        build_sample()
    elif args.arm:
        run_arm(args.arm, args.smoke, args.workers, args.only_unsure_of)
    elif args.report:
        report()


if __name__ == "__main__":
    main()
