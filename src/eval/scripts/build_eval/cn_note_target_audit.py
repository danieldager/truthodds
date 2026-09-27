"""CN-false urn note-target audit (Daniel 2026-08-28) — independent check of the label.

The FALSE label on every CN-false urn claim comes from a Community Note, and it only
transfers to OUR extracted claim if that claim is the proposition the note disputes.
The build's own matcher (c2_audit.MATCH_SYS) asserted this. This script re-asks the
question BLIND: three judges from three model families see the post, the note and one
extracted claim, and nothing else — no matcher output, no confidence, no gate band.

  uv run python -m eval.scripts.build_eval.cn_note_target_audit --sample   # $0
  uv run python -m eval.scripts.build_eval.cn_note_target_audit --judge --smoke
  uv run python -m eval.scripts.build_eval.cn_note_target_audit --judge
  uv run python -m eval.scripts.build_eval.cn_note_target_audit --report

Outputs: eval/data/cn_target_audit/ (nothing under urn_runs is written).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402

CN = SRC / "eval/data/community_notes"
CORP = SRC / "eval/data/tweet_corpus"
C2 = SRC / "eval/data/urn_runs/c2_false"          # READ ONLY
OUT = SRC / "eval/data/cn_target_audit"
BASE = "https://api.deepinfra.com/v1/openai"
_KEY = None


def _key() -> str:
    """Read the API key on first use, never at import time."""
    global _KEY
    if _KEY is None:
        _KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
                    if l.startswith("DEEPINFRA_API_KEY="))
    return _KEY


SEED = 20260828
N_SAMPLE = 120
N_SMOKE = 20
BUDGET_CAP = 0.75

# three families; the third is the matcher's own model, kept as a same-model control
JUDGES = ["openai/gpt-oss-120b",
          "Qwen/Qwen3-235B-A22B-Instruct-2507",
          "deepseek-ai/DeepSeek-V4-Flash"]
LABELS = ["YES", "RELATED", "NO"]

JUDGE_SYS = """You are auditing a labelling step. A post on X had a Community Note attached to it. Separately, an automatic system extracted one claim from that post. Decide whether that extracted claim is the proposition the note disputes.

You are given the post text, the note text, and one extracted claim. Answer with exactly one label.

- "YES": the note's correction is aimed at this claim's own proposition. The note argues that what this claim asserts is not so. Wording may differ; what matters is that the thing the note contradicts and the thing the claim states are the same proposition.
- "RELATED": the claim comes from the same post and shares its subject matter, but the note disputes a different proposition — another assertion in the post, the post's framing or implication, context the post omitted, the authenticity or provenance of an image or video, who posted it, or a detail this claim does not state. The note's correction, taken as accurate, would not by itself establish that this claim is untrue.
- "NO": the claim and the note concern different matters; the note has no bearing on this claim at all.

Discipline:
- Judge the claim exactly as written, in isolation from the rest of the post. If the post as a whole was misleading but this particular claim is a narrower statement the note never contradicts, that is RELATED, not YES.
- If what the note targets is something the claim does not state — a visual, a superlative or exclusivity, a causal link, a quantity, missing context — that is RELATED.
- If the claim asserts that some party said, reported or announced something, and the note disputes the truth of what was said rather than whether that party said it, that is RELATED.
- Do not use outside knowledge about whether the claim is true, and do not judge whether the note is correct. The only question is whether this claim is the proposition the note argues against.

Return only JSON:
{"label": "YES"|"RELATED"|"NO", "failure_mode": null if YES else one of "media_content"|"framing_or_context"|"different_claim_in_post"|"about_the_account"|"attribution_shell"|"superlative_or_scope"|"other", "reason": "<one sentence>"}"""


# ---------------------------------------------------------------- sample

JUDGE_HASH = prompt_hash(JUDGE_SYS)


def urn_rows() -> list[dict]:
    """Exact CN-false urn membership, as fit_two_urn.load_urn defines it."""
    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    rows = []
    for p in (C2 / "scores.jsonl", C2 / "scores_ext.jsonl"):
        for line in p.open():
            r = json.loads(line)
            claim = r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl:
                continue
            if not any((d.get("read") or {}).get("direction") in
                       ("5", "4", "3", "2", "1", "X", "I") for d in r["results"]):
                continue
            rows.append({"claim_id": r["review_url"], "claim": claim,
                         "post_id": r["post_id"], "noteId": r["noteId"],
                         "note_class": r["note_class"], "topic": r.get("topic"),
                         "attribution_form": r.get("attribution_form"),
                         "media_locus": r.get("media_locus"),
                         "mixed_signal": r.get("mixed_signal")})
    return rows


def sample(stratum: str | None = None, n: int = N_SAMPLE) -> None:
    """stratum: None = the headline seeded 120; else a note_class booster draw.

    The booster EXCLUDES anything already in sample.jsonl and is reported
    SEPARATELY, never pooled with the headline sample (it is not representative)."""
    rows = urn_rows()
    print(f"urn: {len(rows)} claims | class {dict(Counter(r['note_class'] for r in rows))}",
          flush=True)
    if stratum:
        already = {json.loads(l)["claim_id"] for l in open(OUT / "sample.jsonl")}
        rows = [r for r in rows if r["note_class"] == stratum
                and r["claim_id"] not in already]
        print(f"stratum {stratum}: {len(rows)} eligible (excl. {len(already)} headline)",
              flush=True)
    posts = pl.concat([pl.read_parquet(CORP / "cn_false_urn_posts.parquet"),
                       pl.read_parquet(CORP / "cn_false_urn_ext_posts.parquet")])
    ptxt = {r["post_id"]: r for r in posts.select(
        ["post_id", "handle", "text", "n_images", "n_videos"]).iter_rows(named=True)}
    notes = pl.read_parquet(CN / "cn_gold.parquet").select(["noteId", "summary"])
    ntxt = dict(zip(notes["noteId"].to_list(), notes["summary"].to_list()))

    rows.sort(key=lambda r: r["claim_id"])
    draw = random.Random(SEED if not stratum else SEED + 1).sample(rows, min(n, len(rows)))
    for i, r in enumerate(draw):
        p = ptxt.get(r["post_id"], {})
        r["order"] = i
        r["handle"] = p.get("handle")
        r["post_text"] = p.get("text")
        r["n_images"] = p.get("n_images")
        r["n_videos"] = p.get("n_videos")
        r["note"] = ntxt.get(r["noteId"])
    miss = [r["claim_id"] for r in draw if not r["post_text"] or not r["note"]]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"stratum_{stratum}.jsonl" if stratum else "sample.jsonl")
    with open(path, "w") as fh:
        for r in draw:
            fh.write(json.dumps(r) + "\n")
    print(f"{path.name} locked: {len(draw)} claims | missing post/note text: {len(miss)}",
          flush=True)


# ---------------------------------------------------------------- judge

def _chat(model: str, sys_prompt: str, user: str, retries: int = 4) -> dict:
    # 900 not 400: gpt-oss-120b is a reasoning model and silently returned an empty
    # object on ~5% of items at 400 (reasoning ate the budget). Raised for the RETRY
    # of items that produced no measurement at all — never for an already-judged item.
    payload = {"model": model, "temperature": 0, "max_tokens": 900,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            m = re.search(r"\{.*\}", d["choices"][0]["message"]["content"], re.S)
            out = json.loads(m.group(0)) if m else {}
            out["_cost"] = (d.get("usage") or {}).get("estimated_cost") or 0
            return out
        except Exception as e:
            last = e
            time.sleep(3 * 2 ** a)
    raise RuntimeError(f"{type(last).__name__}: {last}")


def _payload(r: dict) -> str:
    media = []
    if r.get("n_images"):
        media.append(f"{r['n_images']} image(s)")
    if r.get("n_videos"):
        media.append(f"{r['n_videos']} video(s)")
    mline = f"\n[the post also carries {' and '.join(media)}]" if media else ""
    return (f"POST (@{r['handle']}):\n{r['post_text']}{mline}\n\n"
            f"COMMUNITY NOTE:\n{r['note']}\n\n"
            f"EXTRACTED CLAIM:\n{r['claim']}")


def judge(smoke: bool, workers: int, stratum: str | None = None) -> None:
    src = OUT / (f"stratum_{stratum}.jsonl" if stratum else "sample.jsonl")
    rows = [json.loads(l) for l in open(src)]
    rows = rows[:N_SMOKE] if smoke else rows
    out_path = OUT / (f"judged_stratum_{stratum}.jsonl" if stratum else "judged.jsonl")
    done = {(j["claim_id"], j["model"]) for j in
            (json.loads(l) for l in open(out_path))} if out_path.exists() else set()
    todo = [(r, m) for r in rows for m in JUDGES if (r["claim_id"], m) not in done]
    print(f"judge{' [SMOKE]' if smoke else ''}: {len(todo)} calls "
          f"({len(rows)} claims x {len(JUDGES)} judges) | cap ${BUDGET_CAP}", flush=True)
    lock, st, t0 = threading.Lock(), {"cost": 0.0, "n": 0, "fail": 0}, time.time()
    fh = open(out_path, "a")

    def work(job):
        r, model = job
        try:
            o = _chat(model, JUDGE_SYS, _payload(r))
        except Exception as e:
            with lock:
                st["n"] += 1
                st["fail"] += 1
                print(f"  [failed] {r['claim_id']} {model}: {e}", flush=True)
            return
        lab = str(o.get("label", "")).strip().upper()
        with lock:
            st["cost"] += o.pop("_cost", 0)
            st["n"] += 1
            fh.write(json.dumps({
                "claim_id": r["claim_id"], "model": model,
                "label": lab if lab in LABELS else None,
                "failure_mode": o.get("failure_mode"),
                "reason": o.get("reason"), "prompt_hash": JUDGE_HASH}) + "\n")
            n, el = st["n"], (time.time() - t0) / 60
            if n % 20 == 0 or n == len(todo):
                fh.flush()
                eta = el / n * (len(todo) - n)
                print(f"  {n}/{len(todo)} | ${st['cost']:.4f} | {el:.1f}m "
                      f"{n/max(el,.01):.0f}/min | ETA {eta:.1f}m | fail {st['fail']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"judge done {st['n']} | ${st['cost']:.4f} | {(time.time()-t0)/60:.1f}m | "
          f"failed {st['fail']}", flush=True)


# ---------------------------------------------------------------- report

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fleiss(votes: list[list[str]]) -> float:
    """Fleiss kappa over items with the same number of raters."""
    votes = [v for v in votes if len(v) == len(JUDGES)]
    n, k = len(votes), len(JUDGES)
    if n == 0:
        return float("nan")
    cnt = [Counter(v) for v in votes]
    p_i = [(sum(c[l] ** 2 for l in LABELS) - k) / (k * (k - 1)) for c in cnt]
    p_j = [sum(c[l] for c in cnt) / (n * k) for l in LABELS]
    pbar, pe = sum(p_i) / n, sum(p * p for p in p_j)
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def cohen(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in LABELS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(smoke: bool, show: int, stratum: str | None = None) -> None:
    src = OUT / (f"stratum_{stratum}.jsonl" if stratum else "sample.jsonl")
    jpath = OUT / (f"judged_stratum_{stratum}.jsonl" if stratum else "judged.jsonl")
    rows = {r["claim_id"]: r for r in (json.loads(l) for l in open(src))}
    order = [r["claim_id"] for r in sorted(rows.values(), key=lambda r: r["order"])]
    if smoke:
        order = order[:N_SMOKE]
    keep = set(order)
    by: dict[str, dict[str, dict]] = {}
    for l in open(jpath):
        j = json.loads(l)
        if j["claim_id"] in keep and j["label"]:
            by.setdefault(j["claim_id"], {})[j["model"]] = j
    full = [c for c in order if len(by.get(c, {})) == len(JUDGES)]
    print(f"== note-target audit ({'SMOKE ' if smoke else ''}n={len(full)} "
          f"of {len(order)} sampled, all {len(JUDGES)} judges) ==\n")

    print("per-judge label rates:")
    for m in JUDGES:
        c = Counter(by[cid][m]["label"] for cid in full)
        print(f"  {m:<42} " + "  ".join(
            f"{l} {c[l]:>3} ({c[l]/max(len(full),1):5.1%})" for l in LABELS))

    maj, ties = {}, 0
    for cid in full:
        c = Counter(by[cid][m]["label"] for m in JUDGES)
        top, n_top = c.most_common(1)[0]
        if n_top == 1:            # 3-way split: no majority, take the strictest
            ties += 1
            top = "RELATED"
        maj[cid] = top
    cm = Counter(maj.values())
    print(f"\nMAJORITY VOTE (3-way splits {ties}, resolved to RELATED):")
    for l in LABELS:
        lo, hi = wilson(cm[l], len(full))
        print(f"  {l:<8} {cm[l]:>4}/{len(full)}  {cm[l]/max(len(full),1):6.1%}  "
              f"95% CI [{lo:.1%}, {hi:.1%}]")
    bad = cm["RELATED"] + cm["NO"]
    lo, hi = wilson(bad, len(full))
    print(f"  {'RELATED+NO':<8} {bad:>4}/{len(full)}  {bad/max(len(full),1):6.1%}  "
          f"95% CI [{lo:.1%}, {hi:.1%}]   <- mislabelled share")

    print("\nagreement:")
    print(f"  unanimous 3/3: {sum(1 for c in full if len(set(by[c][m]['label'] for m in JUDGES))==1)}"
          f"/{len(full)}")
    print(f"  Fleiss kappa (3 labels): "
          f"{fleiss([[by[c][m]['label'] for m in JUDGES] for c in full]):.3f}")
    bin_ = [[("YES" if by[c][m]["label"] == "YES" else "NO") for m in JUDGES] for c in full]
    print(f"  Fleiss kappa (YES vs not): {fleiss(bin_):.3f}")
    for i in range(len(JUDGES)):
        for k in range(i + 1, len(JUDGES)):
            a = [by[c][JUDGES[i]]["label"] for c in full]
            b = [by[c][JUDGES[k]]["label"] for c in full]
            po = sum(x == y for x, y in zip(a, b)) / max(len(a), 1)
            print(f"  {JUDGES[i].split('/')[-1]} vs {JUDGES[k].split('/')[-1]}: "
                  f"raw {po:.1%} kappa {cohen(a,b):.3f}")

    print("\nfailure modes (majority RELATED/NO, modal judge code):")
    fm = Counter()
    for cid in full:
        if maj[cid] == "YES":
            continue
        codes = [by[cid][m].get("failure_mode") for m in JUDGES
                 if by[cid][m]["label"] != "YES" and by[cid][m].get("failure_mode")]
        fm[Counter(codes).most_common(1)[0][0] if codes else "unstated"] += 1
    for k, v in fm.most_common():
        print(f"  {k:<26} {v:>3} ({v/max(bad,1):.0%} of mislabelled)")

    def strat(name: str, sub: list[str]) -> None:
        if not sub:
            return
        b = sum(maj[c] != "YES" for c in sub)
        lo, hi = wilson(b, len(sub))
        print(f"  {name:<26} n={len(sub):>3}  YES {1-b/len(sub):6.1%}  "
              f"RELATED+NO {b/len(sub):6.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

    print("\nby note class (majority vote, RELATED+NO = mislabelled):")
    for cls in ("factual_error", "unverified_as_fact", "mixed_true_core", "other"):
        strat(cls, [c for c in full if rows[c]["note_class"] == cls])
    print("  --")
    strat("mixed_true_core", [c for c in full if rows[c]["note_class"] == "mixed_true_core"])
    strat("all other classes", [c for c in full if rows[c]["note_class"] != "mixed_true_core"])
    for flag in ("media_locus", "attribution_form", "mixed_signal"):
        sub = [c for c in full if rows[c].get(flag)]
        if sub:
            y = sum(maj[c] == "YES" for c in sub)
            print(f"  [{flag}=True] n={len(sub)} YES {y/len(sub):.0%}")

    print(f"\n== cases (all RELATED/NO, first {show} YES) ==")
    for cid in order:
        if cid not in maj:
            continue
        if maj[cid] == "YES" and show <= 0:
            continue
        if maj[cid] == "YES":
            show -= 1
        r = rows[cid]
        print(f"\n--- {cid} [{maj[cid]}] {r['note_class']} "
              f"(img {r['n_images']} vid {r['n_videos']})")
        print(f"POST @{r['handle']}: {r['post_text']}")
        print(f"NOTE: {r['note']}")
        print(f"OUR CLAIM: {r['claim']}")
        for m in JUDGES:
            j = by[cid][m]
            print(f"   {m.split('/')[-1]:<32} {j['label']:<8} "
                  f"{j.get('failure_mode') or '-':<24} {j.get('reason')}")


# ---------------------------------------------------------------- controls

def controls(workers: int) -> None:
    """Validity check: can the prompt emit RELATED/NO at all?

    Two synthetic arms over the smoke claims, judged by the same three judges:
      shuffled — the claim paired with ANOTHER post's note   (expect NO)
      sibling  — a different claim extracted from the SAME post, paired with the
                 real note; the matcher did not pick it      (expect RELATED/NO)
    """
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl")][:N_SMOKE]
    claims = pl.concat([pl.read_parquet(CORP / "cn_false_urn_claims.parquet"),
                        pl.read_parquet(CORP / "cn_false_urn_ext_claims.parquet")])
    by_post: dict[str, list[str]] = {}
    for c in claims.select(["post_id", "claim_id", "claim"]).iter_rows(named=True):
        by_post.setdefault(c["post_id"], []).append((c["claim_id"], c["claim"]))

    jobs = []
    for i, r in enumerate(rows):
        other = rows[(i + 7) % len(rows)]
        jobs.append({**r, "arm": "shuffled", "note": other["note"],
                     "key": f"{r['claim_id']}|shuffled"})
        sibs = [c for cid, c in by_post.get(r["post_id"], []) if cid != r["claim_id"]]
        if sibs:
            jobs.append({**r, "arm": "sibling", "claim": sibs[0],
                         "key": f"{r['claim_id']}|sibling"})
    out_path = OUT / "controls.jsonl"
    done = {(j["key"], j["model"]) for j in
            (json.loads(l) for l in open(out_path))} if out_path.exists() else set()
    todo = [(j, m) for j in jobs for m in JUDGES if (j["key"], m) not in done]
    print(f"controls: {len(todo)} calls "
          f"({dict(Counter(j['arm'] for j in jobs))} x {len(JUDGES)} judges)", flush=True)
    lock, st = threading.Lock(), {"cost": 0.0, "n": 0}
    fh = open(out_path, "a")

    def work(job):
        j, model = job
        try:
            o = _chat(model, JUDGE_SYS, _payload(j))
        except Exception as e:
            print(f"  [failed] {j['key']} {model}: {e}", flush=True)
            return
        lab = str(o.get("label", "")).strip().upper()
        with lock:
            st["cost"] += o.pop("_cost", 0)
            st["n"] += 1
            fh.write(json.dumps({"key": j["key"], "arm": j["arm"], "model": model,
                                 "label": lab if lab in LABELS else None,
                                 "failure_mode": o.get("failure_mode"),
                                 "reason": o.get("reason"),
                                 "prompt_hash": JUDGE_HASH}) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    recs = [json.loads(l) for l in open(out_path)]
    print(f"controls done {st['n']} | ${st['cost']:.4f}\n")
    for arm in ("shuffled", "sibling"):
        sub = [r for r in recs if r["arm"] == arm]
        c = Counter(r["label"] for r in sub)
        print(f"  {arm:<9} n={len(sub):>3}  " + "  ".join(
            f"{l} {c[l]:>3} ({c[l]/max(len(sub),1):5.1%})" for l in LABELS))
        print(f"            modes {dict(Counter(r['failure_mode'] for r in sub if r['label']!='YES'))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    for s in ("sample", "judge", "report", "controls"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--show-yes", type=int, default=6)
    ap.add_argument("--stratum", help="note_class booster draw, reported separately")
    ap.add_argument("-n", type=int, default=N_SAMPLE)
    a = ap.parse_args()
    if a.sample:
        sample(a.stratum, a.n)
    if a.judge:
        judge(a.smoke, a.workers, a.stratum)
    if a.report:
        report(a.smoke, a.show_yes, a.stratum)
    if a.controls:
        controls(a.workers)


if __name__ == "__main__":
    main()
