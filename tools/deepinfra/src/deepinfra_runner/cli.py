"""deepinfra-run: preflight, status, and the common per-item case.

    deepinfra-run --preflight [--model M]
    deepinfra-run --pool-status
    deepinfra-run status
    deepinfra-run --items x.jsonl --prompt p.txt --model M --out y.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .gate import DEFAULT_OBS_PATH, default_obs_path
from .pool import default_pool
from .prices import price_of
from .runner import PreflightAbort, preflight_probe, resolve_workers, run
from .transport import base_url

FALLBACK_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def _load_env():
    """Pick up DEEPINFRA_API_KEY from a .env in the cwd or above, if python-dotenv is there."""
    if os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("LLM_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / ".env").exists():
            load_dotenv(d / ".env")
            return


def cmd_status(a):
    obs = Path(a.obs).expanduser() if a.obs else default_obs_path()
    print(f"deepinfra_runner  base_url {base_url(a.base)}")
    key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("LLM_API_KEY")
    print(f"api key: {'set (' + str(len(key)) + ' chars)' if key else 'NOT SET'}")
    print(f"workers: default {resolve_workers('x')}, {a.model} -> {resolve_workers(a.model)}")
    print(f"price {a.model}: ${price_of(a.model)[0]}/${price_of(a.model)[1]} per 1M tok")
    print(default_pool().status_line())
    print(f"observations: {obs}")
    if not obs.exists():
        print("  (no observations yet)")
        return 0
    lines = [json.loads(x) for x in obs.read_text().splitlines() if x.strip()]
    print(f"  {len(lines)} lines; last {min(5, len(lines))}:")
    for r in lines[-5:]:
        print(f"   {r['ts']}  {r['model']}  target {r['target']}  inflight {r['inflight']}  "
              f"{r['reads_s']}/s  429 {r['rate_429']:.1%}  p50 {r['p50_latency']}s  "
              f"p95 {r['p95_latency']}s  failed {r['failed']}  cuts {r['cuts']}")
    best = max(lines, key=lambda r: r.get("reads_s", 0))
    print(f"  peak: {best['reads_s']}/s at target {best['target']} on {best['model']} ({best['ts']})")
    return 0


def cmd_pool_status(a):
    """Ceiling, who holds what, and slot files left above the ceiling."""
    pool = default_pool()
    print(pool.status_line())
    for i, pid in pool.holders():
        print(f"  slot_{i:03d}  pid {pid if pid is not None else '?'}")
    return 0


def cmd_preflight(a):
    print(default_pool().status_line())
    print(f"preflight: {a.preflight_n} tiny calls on {a.model} at up to 96 in-flight")
    try:
        res = preflight_probe(a.model, n=a.preflight_n, base=a.base, obs_path=a.obs)
    except PreflightAbort as e:
        print(f"ABORT: {e}")
        return 2
    print(f"preflight OK: {res.summary()}")
    return 0 if not [e for e in res.errors.values() if e != "spend-cap"] else 1


def cmd_items(a):
    sys_prompt = Path(a.prompt).read_text() if a.prompt else None
    items = [json.loads(x) for x in Path(a.items).read_text().splitlines() if x.strip()]
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    cache = Path(a.cache) if a.cache else out.with_suffix(out.suffix + ".cache.jsonl")

    def make_request(it):
        msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + \
               [{"role": "user", "content": str(it[a.text_field])}]
        body = {"messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens}
        if a.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def parse(j, it):
        return {"content": j["choices"][0]["message"].get("content"),
                "usage": j.get("usage")}

    try:
        res = run(items, make_request, model=a.model, cache_path=cache, parse=parse,
                  workers=a.workers,
                  gate_target=a.gate_target, gate_ceiling=a.gate_ceiling, gate_floor=a.gate_floor,
                  obs_path=a.obs, max_spend=a.max_spend, base=a.base, timeout=a.timeout,
                  preflight=a.preflight, preflight_n=a.preflight_n)
    except PreflightAbort as e:
        print(f"ABORT: {e}")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            iid = it.get("id", it.get("item_id"))
            r = res.results.get(iid)
            f.write(json.dumps({**it, "output": (r or {}).get("content"),
                                "error": res.errors.get(iid)}, ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(res.results)} ok, {len(res.errors)} failed)")
    return 1 if res.errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deepinfra-run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["status", "run"], default="run")
    ap.add_argument("--items", help="input jsonl, one object per item (needs an id field)")
    ap.add_argument("--prompt", help="file holding the system prompt")
    ap.add_argument("--text-field", default="text", help="item field used as the user message")
    ap.add_argument("--out", help="output jsonl")
    ap.add_argument("--cache", help="cache jsonl (default: <out>.cache.jsonl)")
    ap.add_argument("--model", default=None, help=f"default: $LLM_MODEL, else {FALLBACK_MODEL}")
    ap.add_argument("--workers", type=int, default=None, help="default: per-model (200; 8 for Qwen3-235B)")
    ap.add_argument("--gate-target", type=int, default=160)
    ap.add_argument("--gate-ceiling", type=int, default=200)
    ap.add_argument("--gate-floor", type=int, default=32)
    ap.add_argument("--max-spend", type=float, default=None, help="hard cap in $")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json-mode", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base", default=None, help="override base URL")
    ap.add_argument("--obs", default=None, help=f"observations jsonl (default {DEFAULT_OBS_PATH})")
    ap.add_argument("--preflight", action="store_true",
                    help="60s probe at 96 in-flight first; abort if the 429 rate > 20%%")
    ap.add_argument("--preflight-n", type=int, default=96)
    ap.add_argument("--pool-status", action="store_true",
                    help="print the machine-wide slot pool: ceiling, holders, stale files")
    a = ap.parse_args(argv)
    _load_env()
    a.model = a.model or os.environ.get("LLM_MODEL") or FALLBACK_MODEL
    if a.pool_status:
        return cmd_pool_status(a)
    if a.command == "status":
        return cmd_status(a)
    if a.items:
        if not a.out:
            ap.error("--items needs --out")
        return cmd_items(a)
    if a.preflight:
        return cmd_preflight(a)
    ap.error("nothing to do: pass --preflight, --pool-status, --items/--out, or 'status'")


if __name__ == "__main__":
    sys.exit(main())
