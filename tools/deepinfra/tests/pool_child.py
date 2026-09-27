"""Child process for the two-process pool tests. Not collected by pytest (no test_ prefix).

    python pool_child.py run    <out.json> <start_epoch> <n_items> <workers> <delay_s>
    python pool_child.py hold   <ready_file> <n_slots>

`run` does a full deepinfra_runner.run() against a fake transport that sleeps `delay_s`
per request and samples the machine-wide held count while in flight. `hold` grabs
n_slots and blocks forever, so the parent can kill -9 it and watch the locks drop.
"""
import json
import os
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepinfra_runner import run                      # noqa: E402
from deepinfra_runner.pool import default_pool        # noqa: E402


class Resp:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def raise_for_status(self):
        pass


class SlowTransport:
    """Sleeps `delay` per call and, while in flight, samples how many slots the WHOLE
    machine holds -- the direct check that the shared ceiling is never exceeded."""

    def __init__(self, pool, delay):
        self.pool, self.delay = pool, delay
        self.lock = threading.Lock()
        self.max_total = 0
        self.n = 0

    def post(self, url, headers, body, timeout):
        t = self.pool.total_held()
        with self.lock:
            self.max_total = max(self.max_total, t)
            self.n += 1
        time.sleep(self.delay)
        return Resp()


def cmd_run(out, start_epoch, n_items, workers, delay):
    pool = default_pool()
    tr = SlowTransport(pool, delay)
    items = [{"id": f"{os.getpid()}-{i}", "text": f"t{i}"} for i in range(n_items)]
    while time.time() < start_epoch:                  # common start, so neither gets a head start
        time.sleep(0.002)
    t0 = time.time()
    res = run(items, lambda it: {"messages": [{"role": "user", "content": it["text"]}]},
              model="deepseek-ai/DeepSeek-V4-Flash", transport=tr, workers=workers,
              gate_target=workers, gate_floor=workers, pool=pool,
              log=lambda *a, **k: None, sweeps=0, obs_path=str(Path(out).with_suffix(".obs")))
    t1 = time.time()
    Path(out).write_text(json.dumps({
        "pid": os.getpid(), "done": len(res.results), "errors": len(res.errors),
        "t0": t0, "t1": t1, "elapsed": t1 - t0, "calls": tr.n,
        "max_total_held": tr.max_total, "peak_held": pool.peak_held,
        "n_waits": pool.n_waits, "gate_peak": res.gate.peak,
        "mean_inflight": tr.n * delay / max(1e-9, t1 - t0),
    }))


def cmd_hold(ready, n_slots):
    pool = default_pool()
    with ExitStack() as st:
        for _ in range(n_slots):
            st.enter_context(pool.slot())
        Path(ready).write_text(f"{os.getpid()} {pool.held}\n")
        while True:
            time.sleep(0.5)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                float(sys.argv[6]))
    else:
        cmd_hold(sys.argv[2], int(sys.argv[3]))
