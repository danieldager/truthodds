"""Quick VLM bake-off for the Stage-1 gate: same v8 prompt, 3 models, on the CLEAN audit subset
(confident labels only — drops everything under review). Records verdict + latency + token usage per call,
then reports specificity / recall / latency (p50/p95) / $-per-1k / JSON-parse-fail per model.

  uv run python -m eval.scripts.model_bakeoff [--max N]   ->  eval/data/bakeoff_results.parquet

Clean set: positives = audit current=positive AND in_scope AND has_claim (706); negatives = audit
current=negative AND NOT(in_scope AND has_claim) (456). 250 sampled per class (deterministic md5 order).
"""
from __future__ import annotations

import argparse
import base64
import glob
import time
from pathlib import Path

import polars as pl
import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
from eval.scripts._pool import pooled_checkpointed
from eval.scripts.gate_eval import GATE_SYSTEM, MAX_TOKENS, MIN_TOKENS, _obj
from eval.scripts.build_gate_splits import h2

URL = f"{EXTRACTION_BASE_URL}/chat/completions"
HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}
# model -> (slug, $in/M, $out/M)
MODELS = {
    "qwen30b": ("Qwen/Qwen3-VL-30B-A3B-Instruct", 0.15, 0.60),
    "gemma4": ("google/gemma-4-26B-A4B-it", 0.07, 0.34),
    "seed20mini": ("ByteDance/Seed-2.0-mini", 0.10, 0.40),
}


def build_clean_set(n_per: int) -> list[dict]:
    # source lookups with FULL text + image lists (audit stored truncated text + first image only)
    pos_by_img, neg_by_txt = {}, {}
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        d = pl.read_parquet(f)
        if "image_paths" not in d.columns:
            continue
        src = Path(f).stem.replace("_harvest", "")
        for r in d.to_dicts():
            ip = (r.get("image_paths") or [None])[0]
            if ip and ip not in pos_by_img:
                pos_by_img[ip] = {"text": r.get("raw_context") or "", "image_paths": r.get("image_paths"),
                                  "image_est_tokens": r.get("image_est_tokens"), "strat": src}
    for r in pl.read_parquet("eval/data/synthetic_negatives.parquet").to_dicts():
        neg_by_txt[r.get("text") or ""] = {"text": r.get("text") or "", "image_paths": r.get("image_paths"),
                                            "image_est_tokens": r.get("image_est_tokens"), "strat": r.get("category")}
    au = pl.read_parquet("eval/data/dataset_audit.parquet").filter(pl.col("recommend").is_not_null()).to_dicts()
    pos, neg = [], []
    for r in au:
        clean_pos = r["current_label"] == "positive" and r.get("in_scope") is True and r.get("has_claim") is True
        clean_neg = r["current_label"] == "negative" and not (r.get("in_scope") is True and r.get("has_claim") is True)
        if clean_pos and r.get("image_path") in pos_by_img:
            pos.append({"uid": r["uid"], "label": "positive", **pos_by_img[r["image_path"]]})
        elif clean_neg:
            t = r["uid"][5:] if r["uid"].startswith("neg::") else r["text"]
            if t in neg_by_txt:
                neg.append({"uid": r["uid"], "label": "negative", **neg_by_txt[t]})
    # deterministic sample n_per each (md5 order)
    pos.sort(key=lambda x: h2(x["uid"]))
    neg.sort(key=lambda x: h2(x["uid"]))
    return pos[:n_per] + neg[:n_per]


def content_for(row: dict) -> list:
    content = []
    paths = row.get("image_paths") or []
    toks = row.get("image_est_tokens") or [None] * len(paths)
    for i, (p, t) in enumerate(zip(paths, toks)):
        if t is not None and t < MIN_TOKENS:
            continue
        if not p or not Path(p).exists():
            continue
        b = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "text", "text": f"Image {i+1} (post image):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    content.append({"type": "text", "text": f"POST TEXT: {row.get('text') or '(none)'}"})
    return content


def call(slug: str, content: list, use_json: bool):
    body = {"model": slug, "messages": [{"role": "system", "content": GATE_SYSTEM},
            {"role": "user", "content": content}], "temperature": 0, "max_tokens": MAX_TOKENS}
    if use_json:
        body["response_format"] = {"type": "json_object"}
    t0 = time.time()
    r = requests.post(URL, headers=HDR, json=body, timeout=180)
    lat = time.time() - t0
    r.raise_for_status()
    j = r.json()
    g = _obj(j["choices"][0]["message"]["content"])
    u = j.get("usage", {})
    return g, lat, u.get("prompt_tokens"), u.get("completion_tokens")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=250)
    args = ap.parse_args()
    data = build_clean_set(args.max)
    npos = sum(1 for r in data if r["label"] == "positive")
    print(f"clean set: {npos} pos / {len(data)-npos} neg", flush=True)

    out = Path("eval/data/bakeoff_results.parquet")
    for key, (slug, _pin, _pout) in MODELS.items():
        # pre-flight: does this model accept response_format=json_object?
        try:
            call(slug, content_for(data[0]), True)
            use_json = True
        except Exception:
            use_json = False
        print(f"\n=== {key} ({slug}) | json_object={use_json} ===", flush=True)
        res: dict = {}

        def flush():
            if res:
                new = pl.DataFrame(list(res.values()))
                comb = new if not out.exists() else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique(["model", "uid"], keep="last")
                comb.write_parquet(out)

        def work(i, _slug=slug, _key=key, _uj=use_json):
            r = data[i]
            try:
                g, lat, ptok, ctok = call(_slug, content_for(r), _uj)
                kept = bool(g.get("check_worthy")) and g.get("claim_locus") != "video"
                return i, {"model": _key, "uid": r["uid"], "label": r["label"], "kept": kept,
                           "parse_ok": bool(g), "latency": lat, "ptok": ptok, "ctok": ctok}
            except Exception as e:
                return i, {"model": _key, "uid": r["uid"], "label": r["label"], "kept": None,
                           "parse_ok": False, "latency": None, "note": f"ERR {e}"[:120]}

        pooled_checkpointed(list(range(len(data))), work, lambda i, p: res.__setitem__(p["uid"], p), flush, 4, key, checkpoint_every=100)
        flush()
    report()


def report() -> None:
    df = pl.read_parquet("eval/data/bakeoff_results.parquet")
    print("\n\n================== BAKE-OFF ==================")
    print(f"{'model':12} {'spec':>6} {'recall':>7} {'lat_p50':>8} {'lat_p95':>8} {'$/1k':>7} {'parsefail':>10} {'errs':>5}")
    for key, (_slug, pin, pout) in MODELS.items():
        m = df.filter(pl.col("model") == key)
        ev = m.filter(pl.col("kept").is_not_null())
        neg = ev.filter(pl.col("label") == "negative"); pos = ev.filter(pl.col("label") == "positive")
        spec = (1 - neg.filter(pl.col("kept") == True).height / neg.height) * 100 if neg.height else 0  # noqa: E712
        rec = pos.filter(pl.col("kept") == True).height / pos.height * 100 if pos.height else 0  # noqa: E712
        lats = sorted(x for x in m["latency"].to_list() if x is not None)
        p50 = lats[len(lats)//2] if lats else 0
        p95 = lats[int(len(lats)*0.95)] if lats else 0
        pt = [x for x in m["ptok"].to_list() if x]; ct = [x for x in m["ctok"].to_list() if x]
        cost1k = ((sum(pt)/len(pt))*pin + (sum(ct)/len(ct))*pout) / 1e6 * 1000 if pt and ct else 0
        pf = m.filter(pl.col("parse_ok") == False).height  # noqa: E712
        errs = m.filter(pl.col("kept").is_null()).height
        print(f"{key:12} {spec:5.1f}% {rec:6.1f}% {p50:7.2f}s {p95:7.2f}s {cost1k:6.3f} {pf:10} {errs:5}")


if __name__ == "__main__":
    main()
