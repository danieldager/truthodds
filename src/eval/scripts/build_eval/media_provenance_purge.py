"""Two label-validity screens for the CN-false urn, as exclusion files.

SCREEN 1 — MEDIA-PROVENANCE (LLM, 3 judges). SCREEN 2 — MIXED-FRAMING ($0,
deterministic; see mixed_ids()). Separate exclusion files so either can be applied
alone. Neither deletes or rewrites anything.

THE CLASS SCREEN 1 TARGETS (Daniel 2026-08-28, found by cn_note_target_audit.py):

    The reviewer disputes the PROVENANCE or AUTHENTICITY of the media attached to
    the post — what the image shows, when or where it was taken, whether it was
    edited or synthesised, who is in it — while OUR claim asserts the underlying
    event or fact. Granting the reviewer's correction in full leaves our claim's
    truth untouched, yet the claim inherits the review's FALSE verdict.

It is invisible to the existing screens for two structural reasons. Hydration was
TEXT-ONLY everywhere, so no judge in any build ever saw the media. And the media
screens that exist fire on claims that TALK ABOUT media (`cn_false_stratum.MEDIA_RE`
over claim text; the note-box rule requiring manipulated-media to be the SOLE box),
whereas these claims are bare event assertions that never mention media at all.

THIS SCRIPT IS ABOUT LABEL VALIDITY, NOT GATING. It does not touch pipeline/, the
scope gate, or extraction. Media posts stay in scope and stay routed by locus.

ONE detector, ONE prompt, ONE model set, applied identically to every at-risk
corpus. The verdict is COMPOSED IN CODE from three independent judgements so the
decision is auditable rather than a single opaque label:

    purge  <=>  review_targets_media AND (not claim_is_about_media)
                AND residual == "DOES_NOT_ESTABLISH"

LIMITATION, stated up front: we do not have the media. Hydration was text-only in
every build (`cn_hydrate.py` fxtwitter text path), and in the CN corpus even the
media COUNTS are unpopulated — n_images/n_videos are 0 and image_urls/video_urls
empty for all 6,500 posts. The detector therefore works from text alone and infers
media presence from the review's own language. It cannot see a post that carried
media the reviewer never mentioned; that direction is unmeasurable here.

    uv run python -m eval.scripts.build_eval.media_provenance_purge --inventory
    uv run python -m eval.scripts.build_eval.media_provenance_purge --controls
    uv run python -m eval.scripts.build_eval.media_provenance_purge --judge --corpus cn_false --smoke
    uv run python -m eval.scripts.build_eval.media_provenance_purge --judge --corpus cn_false
    uv run python -m eval.scripts.build_eval.media_provenance_purge --report --corpus cn_false
    uv run python -m eval.scripts.build_eval.media_provenance_purge --emit --corpus cn_false
    uv run python -m eval.scripts.build_eval.media_provenance_purge --emit_mixed
    uv run python -m eval.scripts.build_eval.media_provenance_purge --consequences

Outputs live in eval/data/media_purge/ (NEW dir). The only thing ever written under
eval/data/urn_runs/ is a NEW exclusion file (--emit); no parquet or jsonl is mutated.
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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402

CN = SRC / "eval/data/community_notes"
CORP = SRC / "eval/data/tweet_corpus"
URNS = SRC / "eval/data/urn_runs"
OUT = SRC / "eval/data/media_purge"
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
N_SMOKE = 40
BUDGET_CAP = 3.00

# Same three families as cn_note_target_audit.py, so the two measurements are
# comparable and no single model's blind spot decides a purge.
JUDGES = ["openai/gpt-oss-120b",
          "Qwen/Qwen3-235B-A22B-Instruct-2507",
          "deepseek-ai/DeepSeek-V4-Flash"]
RESIDUAL = ["ESTABLISHES", "DOES_NOT_ESTABLISH"]


# ---------------------------------------------------------------- the detector
# No dataset-derived examples. Abstract principles only.

DETECT_SYS = """You are auditing whether a truth label is valid. A post was published with some text and, usually, an attached image or video. A reviewer wrote a correction about that post. Separately, an automatic system extracted one claim from the post's text, and that claim was then given the reviewer's FALSE verdict.

Your job is to decide whether the reviewer's correction actually bears on the extracted claim, or whether it only disputes the attached media.

You are given the post text, the reviewer's correction, and the extracted claim. You cannot see the media itself; nobody in this pipeline could. Judge from the text.

Answer three separate questions.

1. review_targets_media — true if the reviewer's correction is, in whole or in part, about an image, video, screenshot, recording or other media artifact accompanying the post: what it depicts, when or where it was captured, whether it is old or recycled from another occasion, whether it was staged, edited, mislabelled or machine-generated, or who actually appears in it. False if the correction never concerns such an artifact.

2. claim_is_about_media — true if the extracted claim ITSELF asserts something about a media artifact: that a picture or recording shows a particular thing, that it is genuine, that it was taken at a certain place or time. False if the claim is a bare assertion about the world — that an event happened, that a person holds a role, that a quantity is so — with no reference to any image or recording.

3. residual — grant every factual correction the reviewer makes, in full, as true. Take nothing else on trust. Now: does the reviewer's correction state, or logically require, that the extracted claim's proposition is untrue?
   - "ESTABLISHES": yes. Either the reviewer directly contradicts what the claim asserts, or the claim's proposition cannot survive the correction being true.
   - "DOES_NOT_ESTABLISH": no. After the correction is granted, the claim's proposition is left undetermined — it might be true, it might be false, the correction simply does not reach it.

Discipline, all of it load-bearing:

- Question 3 is about ENTAILMENT, not plausibility. Do not reason from your own knowledge about whether the claim happens to be true. Do not reason from the post being untrustworthy overall. Ask only what the correction, taken as accurate, forces.

- A correction that establishes only that the attached media is recycled, altered, synthetic, or from a different occasion tells you about the media. It does not by itself tell you whether the event the claim asserts occurred. A person can describe a real event and illustrate it with the wrong picture. Unless the reviewer separately asserts that the event did not occur, that is DOES_NOT_ESTABLISH.

- The exception is a claim whose whole content is a description of what the media shows. If the claim asserts nothing beyond the depicted scene, then correcting the depiction does reach the claim, and that is ESTABLISHES.

- A correction that adds missing context, disputes the post's framing or emphasis, disputes a superlative, a comparison or a characterisation, or disputes a different assertion in the post, does not establish that THIS claim is untrue.

- A correction that fixes a date or location the extracted claim does not itself state does not establish that the claim is untrue. Judge the claim exactly as written, in isolation from the rest of the post.

- If the claim asserts that a party said, reported or announced something, and the correction disputes the truth of what was said rather than whether it was said, that is DOES_NOT_ESTABLISH.

- Do not judge whether the reviewer is right. Assume they are.

Return only JSON:
{"review_targets_media": true|false, "claim_is_about_media": true|false, "residual": "ESTABLISHES"|"DOES_NOT_ESTABLISH", "reason": "<one sentence>"}"""


DETECT_HASH = prompt_hash(DETECT_SYS)


def purge_vote(v: dict) -> bool:
    """The composed decision. All three conditions must hold."""
    return bool(v.get("review_targets_media")) and not bool(v.get("claim_is_about_media")) \
        and v.get("residual") == "DOES_NOT_ESTABLISH"


# ---------------------------------------------------------------- corpora

NOTE_MEDIA_RE = re.compile(
    r"\b(photo|photograph|image|picture|pic|video|clip|footage|screenshot|screen shot|"
    r"reel|gif|thumbnail|ai[- ]generated|ai[- ]created|deepfake|deep ?fake|doctored|"
    r"photoshop\w*|manipulat\w*|altered|edited|staged|cgi)\b", re.I)


def _cn_notes() -> dict[str, str]:
    n = pl.read_parquet(CN / "cn_gold.parquet").select(["noteId", "summary"])
    return dict(zip(n["noteId"].to_list(), n["summary"].to_list()))


def load_cn_false() -> list[dict]:
    """The CN-false urn, exactly as fit_two_urn.load_urn defines membership."""
    d = URNS / "c2_false"
    excl = {e["claim"][:80] for e in json.loads((d / "fit_exclusions.json").read_text())}
    posts = pl.concat([pl.read_parquet(CORP / "cn_false_urn_posts.parquet"),
                       pl.read_parquet(CORP / "cn_false_urn_ext_posts.parquet")])
    pmeta = {r["post_id"]: r for r in posts.iter_rows(named=True)}
    ntxt = _cn_notes()
    rows = []
    for p in (d / "scores.jsonl", d / "scores_ext.jsonl"):
        for line in p.open():
            r = json.loads(line)
            claim = r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl:
                continue
            if not any((x.get("read") or {}).get("direction") in
                       ("5", "4", "3", "2", "1", "X", "I") for x in r["results"]):
                continue
            m = pmeta.get(r["post_id"], {})
            rows.append({
                "corpus": "cn_false", "claim_id": r["review_url"], "claim": claim,
                "post_text": m.get("text"), "handle": m.get("handle"),
                "review": ntxt.get(r["noteId"]), "post_id": r["post_id"],
                "note_class": r.get("note_class"),
                "mm_box": m.get("misleadingManipulatedMedia") == "1",
                "n_images": m.get("n_images"), "n_videos": m.get("n_videos")})
    return rows


def _fc_gold(side: str) -> list[dict]:
    """fc-gold, split by the axis exclusion `fit_urn.MEDIA_AXIS` already applies.

    Two sides, and we need BOTH to test the exclusion rather than assume it:
      kept  — the pinned n=3,274 population model_ladder scores. If the detector
              purges ~0 here, the exclusion left no residue.
      media_axis — the 336 rows the exclusion drops. If the detector purges a HIGH
              share here, it is measuring the same construct the exclusion targets,
              which is what makes the low `kept` rate meaningful rather than vacuous.

    fc-gold has no post: the claim was harvested from a fact-check article, not
    extracted from a social post. The reviewer's correction is the fact-check's
    headline plus its rating — exactly the input `derive_judged_axis` was given.
    The SYSTEM prompt is byte-identical to the CN run; only the payload differs,
    because the corpora genuinely differ in what they hold.
    """
    from eval.scripts.build_eval import fit_urn
    axis = fit_urn.load_judged_axis()
    meta = pl.read_parquet(SRC / "eval/data/fc_gold_v3.parquet").select(
        ["review_url", "review_title", "original_rating", "publisher_name", "claimant"])
    m = {r["review_url"]: r for r in meta.iter_rows(named=True)}
    rows = []
    for l in (URNS / "e1_ctx/results-00.jsonl").open():
        r = json.loads(l)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        if not any((d.get("read") or {}).get("direction") in
                   ("5", "4", "3", "2", "1", "X", "I") for d in r.get("results", [])):
            continue
        a = axis.get(r["review_url"]) or "untagged"
        is_media = a == fit_urn.MEDIA_AXIS
        if side == "kept" and (is_media or r.get("rating_subtype") == "mixed"):
            continue
        if side == "media_axis" and not is_media:
            continue
        d = m.get(r["review_url"], {})
        rows.append({
            "corpus": f"fc_gold_{side}", "claim_id": r["review_url"],
            "claim": r.get("claim_resolved") or r.get("claim_text") or "",
            "post_text": None, "handle": d.get("publisher_name"),
            "review": f"{d.get('review_title')}\n[verdict: {d.get('original_rating')}]",
            "judged_axis": a, "veracity": r.get("veracity"),
            "rating_subtype": r.get("rating_subtype")})
    return rows


CORPORA = {"cn_false": load_cn_false,
           "fc_gold_kept": lambda: _fc_gold("kept"),
           "fc_gold_media_axis": lambda: _fc_gold("media_axis"),
           # The FALSE side of the kept cut, swept WHOLE (2026-08-28). Under the
           # Layer-0 reversal this is the fit corpus, so a mislabel here is not a
           # lost claim, it is a corrupted weight. Same rows as fc_gold_kept, same
           # payload, same prompt; only the veracity filter differs.
           "fc_gold_false": lambda: [r for r in _fc_gold("kept")
                                     if (r["veracity"] or 5) <= 3]}

# Sub-sample sizes for corpora we probe rather than sweep whole.
PROBE_N = {"fc_gold_kept": 300}


# ---------------------------------------------------------------- llm

def _chat(model: str, sys_prompt: str, user: str, retries: int = 4) -> dict:
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


def payload(r: dict) -> str:
    post = (f"POST (@{r.get('handle')}):\n{r['post_text']}" if r.get("post_text")
            else "POST:\n[the original post is not available; only the claim as "
                 "harvested and the reviewer's correction are known]")
    return (f"{post}\n\nREVIEWER'S CORRECTION:\n{r['review']}\n\n"
            f"EXTRACTED CLAIM:\n{r['claim']}")


def _norm(o: dict) -> dict | None:
    res = str(o.get("residual", "")).strip().upper()
    if res not in RESIDUAL:
        return None
    return {"review_targets_media": bool(o.get("review_targets_media")),
            "claim_is_about_media": bool(o.get("claim_is_about_media")),
            "residual": res, "reason": o.get("reason")}


def _run(jobs: list[tuple[dict, str]], out_path: Path, key: str, tag: str,
         workers: int) -> None:
    print(f"{tag}: {len(jobs)} calls | cap ${BUDGET_CAP}", flush=True)
    lock, st, t0 = threading.Lock(), {"cost": 0.0, "n": 0, "fail": 0}, time.time()
    fh = open(out_path, "a")

    def work(job):
        r, model = job
        try:
            o = _chat(model, DETECT_SYS, payload(r))
        except Exception as e:
            with lock:
                st["n"] += 1
                st["fail"] += 1
                print(f"  [failed] {r[key]} {model}: {e}", flush=True)
            return
        cost = o.pop("_cost", 0)
        v = _norm(o)
        with lock:
            st["cost"] += cost
            st["n"] += 1
            if v is not None:
                fh.write(json.dumps({key: r[key], "model": model,
                                     "prompt_hash": DETECT_HASH, **v}) + "\n")
            else:
                st["fail"] += 1
            n, el = st["n"], (time.time() - t0) / 60
            if n % 50 == 0 or n == len(jobs):
                fh.flush()
                print(f"  {n}/{len(jobs)} | ${st['cost']:.4f} | {el:.1f}m "
                      f"{n/max(el,.01):.0f}/min | ETA {el/n*(len(jobs)-n):.1f}m | "
                      f"fail {st['fail']}", flush=True)
            if st["cost"] > BUDGET_CAP:
                raise SystemExit(f"BUDGET CAP ${BUDGET_CAP} HIT at {n} calls")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    fh.close()
    print(f"{tag} done {st['n']} | ${st['cost']:.4f} | {(time.time()-t0)/60:.1f}m | "
          f"failed {st['fail']}", flush=True)


def agg(votes: dict[str, dict], rule: str) -> bool:
    """Aggregation is applied HERE, at report/emit time — never baked into the sweep.

    judged_*.jsonl holds one UNAGGREGATED row per (claim_id, model) with all three
    elicited fields, so the rule can be changed without re-spending a cent.
    """
    k = sum(purge_vote(votes[m]) for m in JUDGES)
    return k == len(JUDGES) if rule == "unanimous" else k >= 2


def judge(corpus: str, smoke: bool, workers: int) -> None:
    rows = CORPORA[corpus]()
    rows.sort(key=lambda r: r["claim_id"])
    if corpus in PROBE_N and not smoke:
        n = PROBE_N[corpus]
        rows = random.Random(SEED).sample(rows, min(n, len(rows)))
        print(f"PROBE: seeded {len(rows)} of {len(CORPORA[corpus]())}", flush=True)
    if smoke:
        # Stratified smoke: half from the note-mentions-media candidate pool (where
        # the class lives), half from the rest (where a yes-machine would show).
        cand = [r for r in rows if NOTE_MEDIA_RE.search(r["review"] or "")]
        rest = [r for r in rows if not NOTE_MEDIA_RE.search(r["review"] or "")]
        rnd = random.Random(SEED)
        rows = rnd.sample(cand, N_SMOKE // 2) + rnd.sample(rest, N_SMOKE // 2)
        print(f"SMOKE: {len(rows)} claims ({N_SMOKE//2} media-mentioning note, "
              f"{N_SMOKE//2} not) of {len(cand)}+{len(rest)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / f"pool_{corpus}{'_smoke' if smoke else ''}.jsonl"
    with open(src, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    out_path = OUT / f"judged_{corpus}{'_smoke' if smoke else ''}.jsonl"
    done = {(j["claim_id"], j["model"]) for j in
            (json.loads(l) for l in open(out_path))} if out_path.exists() else set()
    jobs = [(r, m) for r in rows for m in JUDGES if (r["claim_id"], m) not in done]
    _run(jobs, out_path, "claim_id", f"judge[{corpus}]{' SMOKE' if smoke else ''}",
         workers)


# ---------------------------------------------------------------- controls

def controls(workers: int) -> None:
    """Four arms. A detector that purges everything, or nothing, fails these.

      no_media   the note contains no media language at all      expect ~0% purge
      claim_media the CLAIM itself is about the media            expect ~0% purge
                  (this is the shape the EXISTING screens catch — the detector
                   must separate it from the new shape, not merge them)
      known_pos  claims the 28/08 blind judges flagged media_content
                                                                 expect high purge
      shuffled   claim paired with a DIFFERENT post's note        expect ~0% ESTABLISHES
      shuf_media claim paired with a different post's note that DOES mention media.
                 THE DECISIVE ARM. If the detector is really a "note contains media
                 words" machine, this purges at ~100% and the real rate is
                 uninterpretable. The real rate must sit well BELOW this.
    """
    rows = load_cn_false()
    rows.sort(key=lambda r: r["claim_id"])
    rnd = random.Random(SEED + 3)
    CLAIM_MEDIA = re.compile(r"\b(photo|image|picture|video|clip|footage|screenshot)\b", re.I)

    known = set()
    for f in ("judged.jsonl", "judged_stratum_mixed_true_core.jsonl"):
        p = SRC / "eval/data/cn_target_audit" / f
        if not p.exists():
            continue
        agg = defaultdict(list)
        for l in open(p):
            j = json.loads(l)
            agg[j["claim_id"]].append(j)
        for cid, js in agg.items():
            if any(j.get("failure_mode") == "media_content" for j in js):
                known.add(cid)
    print(f"known media_content positives from the 28/08 blind audit: {len(known)}")

    by_id = {r["claim_id"]: r for r in rows}
    no_media = [r for r in rows if not NOTE_MEDIA_RE.search(r["review"] or "")]
    claim_media = [r for r in rows if CLAIM_MEDIA.search(r["claim"])]
    jobs_rows = []
    for r in rnd.sample(no_media, 40):
        jobs_rows.append({**r, "arm": "no_media", "key": f"{r['claim_id']}|no_media"})
    for r in rnd.sample(claim_media, min(30, len(claim_media))):
        jobs_rows.append({**r, "arm": "claim_media", "key": f"{r['claim_id']}|claim_media"})
    for cid in sorted(known):
        if cid in by_id:
            jobs_rows.append({**by_id[cid], "arm": "known_pos", "key": f"{cid}|known_pos"})
    shuf = rnd.sample(rows, 30)
    for i, r in enumerate(shuf):
        other = shuf[(i + 7) % len(shuf)]
        jobs_rows.append({**r, "arm": "shuffled", "review": other["review"],
                          "key": f"{r['claim_id']}|shuffled"})
    # THE DECISIVE ARM. Media-mentioning notes, deliberately mispaired.
    med = [r for r in rows if NOTE_MEDIA_RE.search(r["review"] or "")]
    sm = rnd.sample(med, 40)
    for i, r in enumerate(sm):
        other = sm[(i + 11) % len(sm)]
        jobs_rows.append({**r, "arm": "shuf_media", "review": other["review"],
                          "key": f"{r['claim_id']}|shuf_media"})

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "controls.jsonl"
    done = {(j["key"], j["model"]) for j in
            (json.loads(l) for l in open(out_path))} if out_path.exists() else set()
    jobs = [(r, m) for r in jobs_rows for m in JUDGES if (r["key"], m) not in done]
    print(f"arms: {dict(Counter(r['arm'] for r in jobs_rows))}", flush=True)
    _run(jobs, out_path, "key", "controls", workers)
    arm_of = {r["key"]: r["arm"] for r in jobs_rows}
    control_report(arm_of)


def control_report(arm_of: dict[str, str] | None = None) -> None:
    recs = [json.loads(l) for l in open(OUT / "controls.jsonl")]
    if arm_of is None:
        arm_of = {r["key"]: r["key"].split("|")[-1] for r in recs}
    by = defaultdict(dict)
    for r in recs:
        by[r["key"]][r["model"]] = r
    print("\n== control arms (majority of 3) ==")
    for arm in ("no_media", "claim_media", "known_pos", "shuffled", "shuf_media"):
        keys = [k for k in by if arm_of.get(k) == arm and len(by[k]) == len(JUDGES)]
        if not keys:
            continue
        p = sum(sum(purge_vote(by[k][m]) for m in JUDGES) >= 2 for k in keys)
        rtm = sum(sum(by[k][m]["review_targets_media"] for m in JUDGES) >= 2 for k in keys)
        cam = sum(sum(by[k][m]["claim_is_about_media"] for m in JUDGES) >= 2 for k in keys)
        est = sum(sum(by[k][m]["residual"] == "ESTABLISHES" for m in JUDGES) >= 2 for k in keys)
        n = len(keys)
        lo, hi = wilson(p, n)
        print(f"  {arm:<12} n={n:>3}  PURGE {p:>3} ({p/n:5.1%}) [{lo:.0%},{hi:.0%}]   "
              f"review_targets_media {rtm/n:5.1%}  claim_is_about_media {cam/n:5.1%}  "
              f"ESTABLISHES {est/n:5.1%}")


# ---------------------------------------------------------------- report

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fleiss(votes: list[list[bool]]) -> float:
    votes = [v for v in votes if len(v) == len(JUDGES)]
    n, k = len(votes), len(JUDGES)
    if n == 0:
        return float("nan")
    cnt = [Counter(v) for v in votes]
    p_i = [(sum(c[l] ** 2 for l in (True, False)) - k) / (k * (k - 1)) for c in cnt]
    p_j = [sum(c[l] for c in cnt) / (n * k) for l in (True, False)]
    pbar, pe = sum(p_i) / n, sum(p * p for p in p_j)
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def load_judged(corpus: str, smoke: bool):
    src = OUT / f"pool_{corpus}{'_smoke' if smoke else ''}.jsonl"
    jp = OUT / f"judged_{corpus}{'_smoke' if smoke else ''}.jsonl"
    rows = {r["claim_id"]: r for r in (json.loads(l) for l in open(src))}
    by = defaultdict(dict)
    for l in open(jp):
        j = json.loads(l)
        if j["claim_id"] in rows:
            by[j["claim_id"]][j["model"]] = j
    full = [c for c in rows if len(by.get(c, {})) == len(JUDGES)]
    return rows, by, full


def report(corpus: str, smoke: bool, show: int, rule: str = "majority") -> None:
    rows, by, full = load_judged(corpus, smoke)
    n = len(full)
    print(f"== media-provenance purge [{corpus}]{' SMOKE' if smoke else ''}: "
          f"n={n} of {len(rows)} pooled, all {len(JUDGES)} judges ==\n")

    print("per-judge purge rate:")
    for m in JUDGES:
        k = sum(purge_vote(by[c][m]) for c in full)
        print(f"  {m:<42} {k:>4}/{n} {k/max(n,1):6.1%}")

    print("\nBOTH AGGREGATION RULES (votes stored unaggregated; rule set at emit):")
    for rl in ("majority", "unanimous"):
        k = sum(agg(by[c], rl) for c in full)
        lo, hi = wilson(k, n)
        mark = " <-- selected" if rl == rule else ""
        print(f"  {rl:<10} {k:>4}/{n} = {k/max(n,1):6.1%}  95% CI "
              f"[{lo:.1%}, {hi:.1%}]{mark}")
    vd = Counter(sum(purge_vote(by[c][m]) for m in JUDGES) for c in full)
    print(f"  vote distribution 0/1/2/3 judges: "
          f"{vd[0]}/{vd[1]}/{vd[2]}/{vd[3]}   (2-1 splits: {vd[2]})")
    maj = {c: agg(by[c], rule) for c in full}
    unan = sum(1 for c in full if len({purge_vote(by[c][m]) for m in JUDGES}) == 1)
    print(f"  unanimous (either way) {unan}/{n}   Fleiss kappa (purge vs not): "
          f"{fleiss([[purge_vote(by[c][m]) for m in JUDGES] for c in full]):.3f}")

    print("\ncomponent rates (majority):")
    for f in ("review_targets_media", "claim_is_about_media"):
        v = sum(sum(bool(by[c][m][f]) for m in JUDGES) >= 2 for c in full)
        print(f"  {f:<24} {v:>4}/{n} {v/max(n,1):6.1%}")
    e = sum(sum(by[c][m]["residual"] == "ESTABLISHES" for m in JUDGES) >= 2 for c in full)
    print(f"  {'residual=ESTABLISHES':<24} {e:>4}/{n} {e/max(n,1):6.1%}")

    if any(r.get("note_class") for r in rows.values()):
        print("\nby note class — RATE within class, and COMPOSITION of the purge set.")
        print("  The composition column answers whether this screen and the")
        print("  mixed_true_core exclusion compose cleanly or fight each other.")
        tot_p = sum(maj.values())
        for cls in ("factual_error", "unverified_as_fact", "mixed_true_core", "other"):
            sub = [c for c in full if rows[c].get("note_class") == cls]
            if not sub:
                continue
            b = sum(maj[c] for c in sub)
            lo2, hi2 = wilson(b, len(sub))
            print(f"  {cls:<20} n={len(sub):>4} ({len(sub)/n:5.1%} of urn)  "
                  f"purge {b:>4} {b/len(sub):6.1%} [{lo2:.1%},{hi2:.1%}]  "
                  f"= {b/max(tot_p,1):5.1%} of the purge set")
        mtc = [c for c in full if rows[c].get("note_class") == "mixed_true_core"]
        both = sum(maj[c] for c in mtc)
        print(f"\n  OVERLAP: purging mixed_true_core wholesale would remove "
              f"{len(mtc)} claims,\n  of which {both} are also media-provenance "
              f"purges. The two screens together\n  remove "
              f"{tot_p + len(mtc) - both} claims ({(tot_p+len(mtc)-both)/n:.1%}); "
              f"media-provenance alone removes {tot_p} ({tot_p/n:.1%}),\n"
              f"  and {tot_p - both} of those ({(tot_p-both)/max(tot_p,1):.0%}) sit "
              f"OUTSIDE mixed_true_core and would survive it.")

    for f in ("mm_box",):
        sub = [c for c in full if rows[c].get(f)]
        if sub:
            b = sum(maj[c] for c in sub)
            inpurge = sum(1 for c in full if maj[c] and rows[c].get(f))
            print(f"  [manipulatedMedia box ticked] n={len(sub)} purge {b} "
                  f"({b/len(sub):.1%}); {inpurge}/{tot_p} "
                  f"({inpurge/max(tot_p,1):.0%}) of the purge set has it ticked")

    nm = [c for c in full if NOTE_MEDIA_RE.search(rows[c]["review"] or "")]
    nn = [c for c in full if c not in set(nm)]
    for name, sub in (("note mentions media", nm), ("note does NOT", nn)):
        if sub:
            b = sum(maj[c] for c in sub)
            print(f"  [{name}] n={len(sub)} purge {b} ({b/len(sub):.1%})")

    print(f"\n== purged cases (first {show}) ==")
    for c in [x for x in full if maj[x]][:show]:
        r = rows[c]
        print(f"\n--- {c}  ({r.get('note_class')}, manipulatedMedia box "
              f"{'ticked' if r.get('mm_box') else 'not ticked'})")
        print(f"POST @{r.get('handle')}: {r['post_text']}")
        print(f"REVIEW: {r['review']}")
        print(f"OUR CLAIM: {r['claim']}")
        for m in JUDGES:
            j = by[c][m]
            print(f"   {m.split('/')[-1]:<32} purge={str(purge_vote(j)):<5} "
                  f"media={str(j['review_targets_media'])[:1]} "
                  f"claimmedia={str(j['claim_is_about_media'])[:1]} "
                  f"{j['residual']:<20} {j['reason']}")


def emit(corpus: str, rule: str) -> None:
    """Write the exclusion file. NEW file only; nothing existing is touched.

    Every judged claim is written with its THREE raw votes, so flipping the rule
    later is a filter over this file, not another run. `excluded` is the boolean
    the chosen rule produced; consumers filter on it.
    """
    rows, by, full = load_judged(corpus, smoke=False)
    dest = {"cn_false": URNS / "c2_false/media_provenance_exclusions.json",
            "fc_gold_false": URNS / "e1_ctx/media_provenance_exclusions.json"}[corpus]
    out = []
    for c in sorted(full):
        r = rows[c]
        votes = {m: purge_vote(by[c][m]) for m in JUDGES}
        k = sum(votes.values())
        if k == 0:
            continue                      # nothing to record, keeps the file small
        why = [by[c][m]["reason"] for m in JUDGES if votes[m]]
        out.append({
            "claim_id": c, "claim": r["claim"],
            "reason": "media_provenance_only",
            "excluded": agg(by[c], rule), "rule": rule, "n_votes": k,
            "votes": {m.split("/")[-1]: votes[m] for m in JUDGES},
            "fields": {m.split("/")[-1]: {k2: by[c][m][k2] for k2 in
                                          ("review_targets_media",
                                           "claim_is_about_media", "residual")}
                       for m in JUDGES},
            "detail": why[0] if why else "",
            "note_class": r.get("note_class"),
            "judged_axis": r.get("judged_axis"), "veracity": r.get("veracity"),
            "detector": "media_provenance_purge.py (3 judges, 3 families, temp 0)"})
    n_ex = sum(o["excluded"] for o in out)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} — {len(out)} claims with >=1 purge vote, "
          f"{n_ex} EXCLUDED under rule={rule}, of {len(full)} judged")


# ---------------------------------------------------------------- inventory

def mixed_ids() -> tuple[set[str], dict]:
    """SCREEN 2 — mixed-framing. Costs nothing: the build already computed it twice.

    `cn_false_stratum.py:66-68` writes a `mixed_signal` column
        missingImportantContext == 1 AND factualError == 0 AND unverifiedAsFact == 0
    and `c2_audit._note_class` independently derives `mixed_true_core` from the same
    three boxes under a priority order that reaches it only when fe==0 and uv==0.
    The two are the SAME PREDICATE written twice. Verified, not assumed: on all
    1,969 urn claims both select exactly the same 241 rows, zero divergence, and
    the stored note_class matches a fresh recomputation on every row.

    So "which is the better screen" has no content — there is one screen with two
    names. No LLM is needed and none is used. We key on the stored tag.
    """
    rows = load_cn_false()
    ids = {r["claim_id"] for r in rows if r.get("note_class") == "mixed_true_core"}
    return ids, {r["claim_id"]: r for r in rows}


def emit_mixed() -> None:
    ids, by = mixed_ids()
    dest = URNS / "c2_false/mixed_framing_exclusions.json"
    out = [{"claim_id": c, "claim": by[c]["claim"],
            "reason": "mixed_framing_missing_context_only",
            "excluded": True,
            "basis": "misleadingMissingImportantContext=1, misleadingFactualError=0, "
                     "misleadingUnverifiedClaimAsFact=0",
            "note_class": "mixed_true_core",
            "detector": "deterministic note-box predicate; no model call"}
           for c in sorted(ids)]
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} — {len(out)} exclusions (deterministic, $0)")


def consequences(rule: str) -> None:
    """Every headline the purge touches, before vs after. Reads only; refits in RAM.

    TWO distinct effects, kept apart because they move different numbers:
      1. R_meas ON the CN urn — the share of urn claims the frozen E1 threshold
         already flags. Dropping claims the evidence never reached RAISES this.
      2. The REFIT — the CN urn is the FALSE side of the two-urn fit, so purging it
         moves the 7-flag weights, and through them AUC and recall@2%FPR measured
         on fc-gold.
    """
    import os
    os.environ["EXTRA_EXCLUSIONS"] = "none"   # BEFORE arm must be the pre-purge urn;
    from eval.scripts.build_eval import fit_two_urn as F   # the screens are applied here

    rows, by, full = load_judged("cn_false", smoke=False)
    media = {c for c in full if agg(by[c], rule)}
    mixed, _ = mixed_ids()

    false_all = F.load_urn([F.C2 / "scores.jsonl", F.C2 / "scores_ext.jsonl"],
                           F.C2 / "fit_exclusions.json")
    urn_ids = {r["review_url"] for r in false_all}
    media &= urn_ids
    mixed &= urn_ids
    n0 = len(false_all)
    print(f"== consequences [media rule={rule}] ==\n")
    print("TWO SCREENS, OVERLAP:")
    print(f"  media-provenance only   {len(media - mixed):>5}")
    print(f"  mixed-framing only      {len(mixed - media):>5}")
    print(f"  BOTH                    {len(media & mixed):>5}")
    print(f"  union                   {len(media | mixed):>5}  "
          f"({len(media | mixed)/n0:.1%} of {n0})")
    print(f"  remaining after both    {n0 - len(media | mixed):>5}")
    ov = len(media & mixed) / max(len(media), 1)
    print(f"  share of the media purge that ALSO falls in mixed_true_core: {ov:.1%}")

    true_rows = F.load_urn([F.TRUE_SCORES])
    gold = F.load_headline(F.E1_RESULTS)
    print(f"\nfc-gold headline population: n={len(gold)}  (UNCHANGED by both "
          f"screens — they touch only the CN urn)\n")

    arms = (("BEFORE", false_all),
            ("MEDIA", [r for r in false_all if r["review_url"] not in media]),
            ("MIXED", [r for r in false_all if r["review_url"] not in mixed]),
            ("BOTH", [r for r in false_all if r["review_url"] not in (media | mixed)]))
    out = {}
    for name, fr in arms:
        p_mix, p_false = F.rates(true_rows), F.rates(fr)
        fm, ff = F.flag_rates(true_rows), F.flag_rates(fr)
        docs_m = sum(sum(r["flags"].values()) for r in true_rows)
        docs_f = sum(sum(r["flags"].values()) for r in fr)
        w7 = F.demix_flag_weights(fm, ff, 0.10, (docs_m, docs_f))
        ev = F.transfer_eval7(w7, gold)
        s = sorted(F.score7(r["flags"], w7) for r in fr)
        thr = ev["threshold"]
        rmeas = sum(1 for x in s if x <= thr) / len(s)
        out[name] = {"w7": w7, "ev": ev, "rmeas": rmeas, "n": len(fr)}
        print(f"{name:<7} n_false={len(fr):>5}  AUC {ev['auc']:.4f}  "
              f"recall@2%FPR {ev['recall_at_2pct_fpr']:.4f}  thr {thr:+.3f}  "
              f"R_meas(on CN urn) {rmeas:.3%}")
    b = out["BEFORE"]
    print()
    for name in ("MEDIA", "MIXED", "BOTH"):
        a = out[name]
        print(f"DELTA vs BEFORE [{name:<6}]  AUC {a['ev']['auc']-b['ev']['auc']:+.4f}   "
              f"recall@2%FPR "
              f"{a['ev']['recall_at_2pct_fpr']-b['ev']['recall_at_2pct_fpr']:+.4f}"
              f"   R_meas {a['rmeas']-b['rmeas']:+.3%}")
    print("\n7-flag weights, BEFORE -> BOTH:")
    a = out["BOTH"]
    for f in F.FLAGS7:
        print(f"  {f}  {b['w7'][f]:+.4f} -> {a['w7'][f]:+.4f} "
              f"({a['w7'][f]-b['w7'][f]:+.4f})")

    # Flag rate of each purged block vs the rest: WHY recall moves.
    print()
    for nm, sub in (("MEDIA-purged", [r for r in false_all if r["review_url"] in media]),
                    ("MIXED-purged", [r for r in false_all if r["review_url"] in mixed]),
                    ("KEPT after both",
                     [r for r in false_all if r["review_url"] not in (media | mixed)])):
        if not sub:
            continue
        tot = sum(r["n_t"] + r["n_f"] + r["n_e"] for r in sub)
        sil = sum(r["n_e"] for r in sub) / max(tot, 1)
        allirr = sum(1 for r in sub if r["n_f"] == 0 and r["n_t"] == 0) / len(sub)
        print(f"  {nm:<7} n={len(sub):>5}  silent-doc share {sil:.1%}  "
              f"all-irrelevant claims {allirr:.1%}")


def inventory() -> None:
    print("== corpora sweep: can the media-provenance class occur? ==\n")
    print("The class requires: a claim EXTRACTED from a post, inheriting a label")
    print("from a REVIEW OF THAT POST, where the post carried media.\n")
    rows = load_cn_false()
    cand = sum(1 for r in rows if NOTE_MEDIA_RE.search(r["review"] or ""))
    cm = sum(1 for r in rows
             if re.search(r"\b(photo|image|picture|video|clip|footage|screenshot)\b",
                          r["claim"], re.I))
    print(f"cn_false   n={len(rows)}  AT RISK  note mentions media {cand} "
          f"({cand/len(rows):.1%})  claim mentions media {cm} ({cm/len(rows):.1%})")
    print(f"           media counts in corpus: n_images/n_videos all "
          f"{set(r['n_images'] for r in rows)} / {set(r['n_videos'] for r in rows)}"
          f"  <- unpopulated, media presence must be inferred from note text")


def main() -> None:
    ap = argparse.ArgumentParser()
    for s in ("inventory", "judge", "report", "controls", "emit", "control_report",
              "emit_mixed",
              "consequences"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--corpus", default="cn_false")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--rule", choices=("majority", "unanimous"), default="majority",
                    help="aggregation, applied at report/emit time only")
    a = ap.parse_args()
    if a.inventory:
        inventory()
    if a.controls:
        controls(a.workers)
    if a.control_report:
        control_report()
    if a.judge:
        judge(a.corpus, a.smoke, a.workers)
    if a.report:
        report(a.corpus, a.smoke, a.show, a.rule)
    if a.emit:
        emit(a.corpus, a.rule)
    if a.emit_mixed:
        emit_mixed()
    if a.consequences:
        consequences(a.rule)


if __name__ == "__main__":
    main()
