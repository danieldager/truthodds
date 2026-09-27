"""K-posts-in-flight runner over the pooled clients (pipeline/pools.py).

The per-post pipeline is any `async def fn(post: dict, pools: Pools) -> dict`. This harness:
  - runs K posts concurrently (posts mostly wait on pools, so K is cheap; pool semaphores
    are the real concurrency governors),
  - checkpoints each post as it completes to jsonl shards (crash/resume safe: a re-run
    skips post_ids already on disk),
  - prints a flushed status line every few seconds (done/total, throughput, ETA, per-pool
    in-flight/p50/retries/hangs) and a LOUD stall alarm if nothing completes for
    cfg.stall_alarm_s (house rule: never be in the dark on a long run).

Pipeline exceptions (CallFatal data errors or bugs) are caught, recorded to the shard with
ok=false, and never take the run down. Transient failures never reach here — the pools
retry them forever.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.pools import OrchestrationConfig, Pools

log = logging.getLogger("harness")


def _shard_path(out_dir: Path, post_id: str, n_shards: int) -> Path:
    h = int(hashlib.sha1(str(post_id).encode()).hexdigest(), 16) % n_shards
    return out_dir / f"results-{h:02d}.jsonl"


def load_done(out_dir: Path) -> set[str]:
    """post_ids already checkpointed (any ok status — errored posts are re-run only if the
    caller filters them out of `done` explicitly)."""
    done: set[str] = set()
    for shard in sorted(out_dir.glob("results-*.jsonl")):
        with shard.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue  # torn tail line from a crash mid-write; that post re-runs
                if rec.get("ok"):
                    done.add(str(rec["post_id"]))
    return done


class _Progress:
    def __init__(self, total: int, already_done: int, pools: Pools, name: str):
        self.total = total
        self.done = already_done
        self.errors = 0
        self.pools = pools
        self.name = name
        self.t0 = time.monotonic()
        self.done_at_start = already_done
        self.last_complete = time.monotonic()
        self.stall_warned = False

    def on_complete(self, ok: bool):
        self.done += 1
        if not ok:
            self.errors += 1
        self.last_complete = time.monotonic()
        self.stall_warned = False

    def line(self) -> str:
        elapsed = time.monotonic() - self.t0
        run_done = self.done - self.done_at_start
        rate = run_done / elapsed * 60 if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = f"{remaining / (rate / 60) / 3600:.1f}h" if rate > 0 else "?"
        hms = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        err = f" err {self.errors}" if self.errors else ""
        return (f"[{self.name} {hms}] {self.done}/{self.total} done{err} "
                f"({rate:.1f}/min, ETA {eta}) | {self.pools.status_line()} "
                f"| last-done {time.monotonic() - self.last_complete:.0f}s ago")


async def _progress_loop(prog: _Progress, cfg: OrchestrationConfig):
    while True:
        await asyncio.sleep(cfg.progress_every)
        print(prog.line(), flush=True)
        stalled_for = time.monotonic() - prog.last_complete
        if stalled_for > cfg.stall_alarm_s and prog.done < prog.total and not prog.stall_warned:
            prog.stall_warned = True
            print(f"\n{'!' * 80}\n!!! STALL ALARM: no post completed in {stalled_for / 60:.1f} min "
                  f"({prog.done}/{prog.total} done). Pools: {prog.pools.status_line()}\n{'!' * 80}\n",
                  flush=True)


async def run_posts(posts: list[dict], pipeline_fn, pools: Pools, out_dir: str | Path,
                    *, name: str = "run") -> dict:
    """Run `pipeline_fn(post, pools)` for every post not already checkpointed in out_dir.

    Each post dict must carry a "post_id". Returns summary stats.
    """
    cfg = pools.cfg
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_ids = load_done(out_dir)
    todo = [p for p in posts if str(p["post_id"]) not in done_ids]
    print(f"[{name}] {len(posts)} posts, {len(done_ids)} already done, {len(todo)} to run "
          f"(K={cfg.k_posts}) -> {out_dir}", flush=True)
    if not todo:
        return {"total": len(posts), "ran": 0, "errors": 0, "wall_s": 0.0}

    prog = _Progress(len(posts), len(done_ids), pools, name)
    post_sem = asyncio.Semaphore(cfg.k_posts)
    shard_locks = [asyncio.Lock() for _ in range(cfg.n_shards)]

    async def _checkpoint(rec: dict):
        path = _shard_path(out_dir, rec["post_id"], cfg.n_shards)
        idx = int(path.stem.split("-")[1])
        line = json.dumps(rec, ensure_ascii=False)
        async with shard_locks[idx]:
            # open/append/close per record: crash-safe, and 10k writes is nothing
            await asyncio.to_thread(_append_line, path, line)

    async def _one(post: dict):
        async with post_sem:
            t0 = time.monotonic()
            rec = {"post_id": str(post["post_id"]),
                   "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                result = await pipeline_fn(post, pools)
                rec.update(ok=True, result=result)
            except Exception as e:
                log.warning("[%s] post %s failed: %s: %s", name, post["post_id"],
                            type(e).__name__, e)
                rec.update(ok=False, error=f"{type(e).__name__}: {e}")
            rec["elapsed_s"] = round(time.monotonic() - t0, 1)
            await _checkpoint(rec)
            prog.on_complete(rec["ok"])

    reporter = asyncio.create_task(_progress_loop(prog, cfg))
    t0 = time.monotonic()
    try:
        await asyncio.gather(*(_one(p) for p in todo))
    finally:
        reporter.cancel()
    wall = time.monotonic() - t0
    print(prog.line(), flush=True)
    print(f"[{name}] finished: {len(todo)} ran, {prog.errors} errors, {wall:.0f}s wall", flush=True)
    return {"total": len(posts), "ran": len(todo), "errors": prog.errors, "wall_s": round(wall, 1)}


def _append_line(path: Path, line: str):
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
