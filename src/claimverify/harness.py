"""K-claims-in-flight runner with sharded resumable checkpoints, per-claim traces, a
flushed progress line every 10 claims and every `progress_every` seconds, a stall alarm,
a USD budget abort, and a run manifest (git SHA, config, prompt hashes, dataset sha256).

Output layout under out_dir:
  shard-XX.jsonl   {claim_id, ts, ok, result | error, elapsed_s}   (resume key: claim_id)
  trace/<claim_id>.json, pages/<sha>.txt + pages/index.jsonl       (claimverify.trace)
  manifest.json, config.json
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from claimverify.config import SRC, ClaimVerifyConfig, OrchestrationConfig
from claimverify.loop import verify_claim
from claimverify.pools import Pools
from claimverify.prompts import ALL_PROMPTS
from claimverify.trace import PageStore, Trace

log = logging.getLogger("claimverify.harness")


def _shard_path(out_dir: Path, claim_id: str, n_shards: int) -> Path:
    h = int(hashlib.sha1(str(claim_id).encode()).hexdigest(), 16) % n_shards
    return out_dir / f"shard-{h:02d}.jsonl"


def load_records(out_dir: Path, ok_only: bool = True) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for shard in sorted(Path(out_dir).glob("shard-*.jsonl")):
        with shard.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("ok") or not ok_only:
                    recs[str(rec["claim_id"])] = rec
    return recs


def _git(*a) -> str:
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=SRC).stdout.strip()
    except Exception:
        return ""


def write_manifest(out_dir: Path, *, cfg: ClaimVerifyConfig, ocfg: OrchestrationConfig,
                   dataset_path: Path | None, n: int, arm: str, seed: int | None,
                   smoke: int | None, extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loop_version": cfg and __import__("claimverify.config", fromlist=["LOOP_VERSION"]).LOOP_VERSION,
        "arm": arm, "n_claims": n, "seed": seed, "smoke": smoke,
        "model": cfg.model, "config": cfg.to_dict(), "orchestration": ocfg.to_dict(),
        "prompts_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ALL_PROMPTS.items()},
        "prompts_bundle_sha256": hashlib.sha256(
            json.dumps(ALL_PROMPTS, sort_keys=True).encode()).hexdigest(),
        "dataset": str(dataset_path.relative_to(SRC)) if dataset_path else None,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path else None,
        "cache_dir": str(__import__("claimverify.disk_cache", fromlist=["get_dir"]).get_dir()),
    }
    if extra:
        m.update(extra)
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=1))
    (out_dir / "config.json").write_text(json.dumps(
        {"loop": cfg.to_dict(), "orchestration": ocfg.to_dict()}, indent=1))
    return m


class _Progress:
    def __init__(self, total: int, already: int, pools: Pools, name: str):
        self.total, self.done, self.errors = total, already, 0
        self.pools, self.name = pools, name
        self.t0 = time.monotonic()
        self.done_at_start = already
        self.last_complete = time.monotonic()
        self.stall_warned = False
        self.cost = 0.0
        self.llm_calls = 0
        self.llm_hits = 0
        self.scrapes = 0

    def on_complete(self, rec: dict):
        self.done += 1
        if not rec.get("ok"):
            self.errors += 1
        else:
            r = rec["result"]
            self.cost += r.get("cost_usd") or 0.0
            self.llm_calls += r.get("llm_calls") or 0
            self.llm_hits += r.get("llm_cache_hits") or 0
            self.scrapes += r.get("scrapes") or 0
        self.last_complete = time.monotonic()
        self.stall_warned = False

    def projected(self) -> float:
        run_done = self.done - self.done_at_start
        return self.cost / run_done * (self.total - self.done_at_start) if run_done else 0.0

    def line(self) -> str:
        elapsed = time.monotonic() - self.t0
        run_done = self.done - self.done_at_start
        rate = run_done / elapsed * 60 if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = f"{remaining / rate:.1f}m" if rate > 0 else "?"
        hms = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        err = f" err {self.errors}" if self.errors else ""
        p = self.pools
        hit = self.llm_hits / self.llm_calls if self.llm_calls else 0.0
        return (f"[{self.name} {hms}] {self.done}/{self.total} done{err} ({rate:.1f}/min, ETA {eta}) "
                f"| ${self.cost:.3f} proj ${self.projected():.2f} | llm {self.llm_calls} (cache {hit:.0%}) "
                f"| serper credits {p.serper.credits} (cache {p.serper.metrics.cache_hits}) "
                f"| exa {p.exa.calls} | scrapes {self.scrapes} | {p.status_line()} "
                f"| last-done {time.monotonic() - self.last_complete:.0f}s ago")


async def _progress_loop(prog: _Progress, ocfg: OrchestrationConfig):
    while True:
        await asyncio.sleep(ocfg.progress_every)
        print(prog.line(), flush=True)
        stalled = time.monotonic() - prog.last_complete
        if stalled > ocfg.stall_alarm_s and prog.done < prog.total and not prog.stall_warned:
            prog.stall_warned = True
            print(f"\n{'!' * 80}\n!!! STALL ALARM: no claim completed in {stalled / 60:.1f} min "
                  f"({prog.done}/{prog.total} done). Pools: {prog.pools.status_line()}\n{'!' * 80}\n",
                  flush=True)


async def run_claims(claims: list[dict], pools: Pools, cfg: ClaimVerifyConfig,
                     out_dir: str | Path, *, name: str = "run", budget_usd: float = 0.0,
                     min_for_projection: int = 10) -> dict:
    """Run verify_claim for every claim not already checkpointed. Each claim dict carries
    `claim_id`. A budget abort stops LAUNCHING new claims once the projected spend (after
    `min_for_projection` completions) exceeds `budget_usd`; in-flight claims finish."""
    ocfg = pools.cfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir, pages = out_dir / "trace", PageStore(out_dir / "pages")
    done = load_records(out_dir)
    todo = [c for c in claims if str(c["claim_id"]) not in done]
    print(f"[{name}] {len(claims)} claims, {len(done)} already done, {len(todo)} to run "
          f"(K={ocfg.k_claims}, budget ${budget_usd or 'none'}) -> {out_dir}", flush=True)
    if not todo:
        return {"total": len(claims), "ran": 0, "errors": 0, "wall_s": 0.0, "cost_usd": 0.0}
    prog = _Progress(len(claims), len(done), pools, name)
    sem = asyncio.Semaphore(ocfg.k_claims)
    shard_locks = [asyncio.Lock() for _ in range(ocfg.n_shards)]
    stop = {"budget": False}

    async def _checkpoint(rec: dict):
        path = _shard_path(out_dir, rec["claim_id"], ocfg.n_shards)
        idx = int(path.stem.split("-")[1])
        line = json.dumps(rec, ensure_ascii=False)
        async with shard_locks[idx]:
            await asyncio.to_thread(_append_line, path, line)

    async def _one(claim: dict):
        async with sem:
            if stop["budget"]:
                return
            t0 = time.monotonic()
            cid = str(claim["claim_id"])
            rec = {"claim_id": cid, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            trace = Trace(cid, pages)
            try:
                result = await verify_claim(claim, pools, cfg, trace)
                rec.update(ok=True, result=result)
            except Exception as e:
                log.warning("[%s] claim %s failed: %s: %s", name, cid, type(e).__name__, e)
                rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            rec["elapsed_s"] = round(time.monotonic() - t0, 1)
            await asyncio.to_thread(trace.dump, trace_dir)
            await _checkpoint(rec)
            prog.on_complete(rec)
            n_run = prog.done - prog.done_at_start
            if n_run % 10 == 0:
                print(prog.line(), flush=True)
            if budget_usd and n_run >= min_for_projection and prog.projected() > budget_usd \
                    and not stop["budget"]:
                stop["budget"] = True
                print(f"\n!!! BUDGET ABORT: projected ${prog.projected():.2f} > cap ${budget_usd:.2f} "
                      f"after {n_run} claims — no new claims launched\n", flush=True)

    reporter = asyncio.create_task(_progress_loop(prog, ocfg))
    t0 = time.monotonic()
    try:
        await asyncio.gather(*(_one(c) for c in todo))
    finally:
        reporter.cancel()
    wall = time.monotonic() - t0
    print(prog.line(), flush=True)
    print(f"[{name}] finished: {prog.done - prog.done_at_start} ran, {prog.errors} errors, "
          f"{wall:.0f}s wall, ${prog.cost:.4f}, budget_abort={stop['budget']}", flush=True)
    return {"total": len(claims), "ran": prog.done - prog.done_at_start, "errors": prog.errors,
            "wall_s": round(wall, 1), "cost_usd": round(prog.cost, 4),
            "budget_abort": stop["budget"]}


def _append_line(path: Path, line: str):
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
