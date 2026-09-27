"""Crash-safe, HIGH-VISIBILITY thread-pool driver shared by the resumable eval passes
(fetch_sources, enrich_eval, quality_gate).

Two jobs:
1. Crash safety — apply results OUT-OF-ORDER (one hung row can't stall the pass), CHECKPOINT via a
   caller flush() every `checkpoint_every` rows, ABANDON stragglers after `idle_timeout` of no
   completions. A single hung request once stalled a finished run whose only write was at the end,
   costing 97% of the work (clog 250626).
2. Visibility — print a flushed progress line every LOG_INTERVAL seconds with throughput + ETA, a
   loud STALLED line if it idles out, and a final summary. So a background job is never a black box:
   you can always see done/total, rows/s, ETA, or that it hung (w/ Daniel, clog 250626). The tqdm
   bar auto-disables in non-TTY (background) so it doesn't clutter the log; the printed lines carry
   the signal.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from tqdm import tqdm

LOG_INTERVAL = 30  # seconds between flushed progress lines


def pooled_checkpointed(todo: list, work: Callable, apply: Callable, flush: Callable,
                        workers: int, label: str,
                        checkpoint_every: int = 200, idle_timeout: int = 60) -> bool:
    """Run work(item) over `todo` in a pool of `workers`, checkpointing + logging as it goes.

    - work(item)          -> (key, payload)   # runs in a worker thread
    - apply(key, payload)  -> None            # mutates caller state, main thread
    - flush()             -> None             # persist a checkpoint, main thread

    Returns True if any rows were ABANDONED (hit the idle timeout). The caller should then hard-exit
    (os._exit), because a hung non-daemon worker thread would otherwise keep the process alive.
    """
    total = len(todo)
    t0 = time.time()
    print(f"{label}: starting {total} items on {workers} workers", flush=True)
    ex = ThreadPoolExecutor(max_workers=workers)
    pending = {ex.submit(work, it) for it in todo}
    done = 0
    abandoned = False
    last_log = t0
    bar = tqdm(total=total, desc=label, disable=None)  # disable=None → auto-off in non-TTY (background)

    def _progress() -> None:
        el = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate / 60 if rate > 0 else 0.0
        pct = done * 100 // total if total else 100
        print(f"{label}: {done}/{total} ({pct}%) · {rate:.1f}/s · ETA {eta:.1f}m", flush=True)

    try:
        while pending:
            finished, pending = wait(pending, timeout=idle_timeout, return_when=FIRST_COMPLETED)
            if not finished:
                print(f"{label}: STALLED — no row completed in {idle_timeout}s; abandoning "
                      f"{len(pending)} hung item(s) (re-run to retry)", flush=True)
                abandoned = True
                break
            for fut in finished:
                key, payload = fut.result()
                apply(key, payload)
                done += 1
                bar.update(1)
                if done % checkpoint_every == 0:
                    flush()
            if time.time() - last_log >= LOG_INTERVAL:
                last_log = time.time()
                _progress()
    finally:
        bar.close()
        ex.shutdown(wait=False, cancel_futures=True)
    flush()
    el = (time.time() - t0) / 60
    rate = done / (el * 60) if el > 0 else 0.0
    print(f"{label}: DONE {done}/{total} in {el:.1f}m ({rate:.1f}/s)"
          + (f" — {len(pending)} ABANDONED" if abandoned else ""), flush=True)
    return abandoned
