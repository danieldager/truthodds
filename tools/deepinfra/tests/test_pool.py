"""The machine-wide slot pool: ceiling, gate composition, fairness, death of a holder.

Everything here is offline -- fake transport, fake clock, real flock. The two-process
tests spawn `tests/pool_child.py` with the pool directory pointed at tmp_path.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import FakeTransport, ok_reply

from deepinfra_runner import RateGate, run
from deepinfra_runner.pool import DEFAULT_CEILING, Pool, ceiling, enabled

CHILD = str(Path(__file__).parent / "pool_child.py")
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
QUIET = lambda *a, **k: None        # noqa: E731
NOSLEEP = lambda s: None            # noqa: E731


def items(n):
    return [{"id": f"i{k}", "text": f"item {k}"} for k in range(n)]


def req(it):
    return {"messages": [{"role": "user", "content": it["text"]}], "max_tokens": 8}


# --- config --------------------------------------------------------------

def test_defaults(monkeypatch):
    assert DEFAULT_CEILING == 190
    monkeypatch.delenv("DEEPINFRA_POOL_CEILING", raising=False)
    assert ceiling() == 190 and enabled()
    monkeypatch.setenv("DEEPINFRA_POOL_CEILING", "42")
    assert ceiling() == 42
    monkeypatch.setenv("DEEPINFRA_POOL", "0")
    assert not enabled()


def test_disabled_pool_is_a_passthrough(tmp_path):
    p = Pool(size=1, directory=tmp_path / "p", on=False)
    g = RateGate(target=4, log=QUIET)
    with p.slot(g):
        with p.slot(g):                       # size 1, but off: no capping at all
            assert g.inflight == 2
    assert not (tmp_path / "p").exists()      # nothing on disk when off


# --- the ceiling, in one process -----------------------------------------

def test_slot_count_is_the_ceiling(tmp_path):
    p = Pool(size=3, directory=tmp_path / "p", poll=0.001, poll_max=0.005)
    held = []
    for _ in range(3):
        cm = p.slot()
        cm.__enter__()
        held.append(cm)
    assert p.held == 3 and p.total_held() == 3
    got = []
    t = threading.Thread(target=lambda: got.append(_take_briefly(p)))
    t.start()
    time.sleep(0.05)
    assert p.waiters == 1 and not got         # the 4th worker polls, it does not take
    held.pop().__exit__(None, None, None)     # free one
    t.join(5)
    assert got == [True] and p.waiters == 0
    for cm in held:
        cm.__exit__(None, None, None)
    assert p.total_held() == 0


def _take_briefly(p):
    with p.slot():
        return True


def test_waiting_for_a_slot_does_not_count_as_in_flight(tmp_path):
    """The gate seat is handed back between polls, so a pool-starved worker is invisible
    to the gate -- and cannot hoard a slot behind a gate it has cut."""
    p = Pool(size=1, directory=tmp_path / "p", poll=0.001, poll_max=0.005)
    g = RateGate(target=8, log=QUIET)
    with p.slot(g):
        assert g.inflight == 1
        t = threading.Thread(target=lambda: _take_briefly_gated(p, g))
        t.start()
        time.sleep(0.05)
        assert p.waiters == 1
        assert g.inflight == 1                # the waiter is NOT in flight
    t.join(5)
    assert g.inflight == 0 and g.peak == 1


def _take_briefly_gated(p, g):
    with p.slot(g):
        pass


def test_slot_is_released_on_an_exception(tmp_path):
    p = Pool(size=2, directory=tmp_path / "p")
    g = RateGate(target=2, log=QUIET)
    with pytest.raises(ValueError):
        with p.slot(g):
            raise ValueError("boom")
    assert p.held == 0 and p.total_held() == 0 and g.inflight == 0


# --- one process alone behaves exactly as before --------------------------

def test_single_process_is_unchanged_by_the_pool(tmp_path):
    """Pool on vs pool off, one process: same calls, same results, and the same 8 requests
    genuinely in flight at once -- the transport holds every request on a barrier of 8, so
    a pool that throttled a lone run would break the barrier and fail the items."""
    def go(on):
        bar = threading.Barrier(8, timeout=20)
        tr = FakeTransport(lambda body, n: (bar.wait(), ok_reply(body))[1])
        p = Pool(size=190, directory=tmp_path / "p", on=on)
        r = run(items(40), req, model=MODEL, transport=tr, pool=p, workers=8,
                gate_target=8, gate_floor=4, log=QUIET, sleep=NOSLEEP)
        return tr.n, len(r.results), r.gate.peak, p.n_waits

    on = go(True)
    off = go(False)
    assert on[:3] == off[:3] == (40, 40, 8)   # same calls, same results, 8 in flight either way
    assert on[3] == 0                         # 190 slots, 8 workers: nobody ever waited


def test_pool_caps_the_gate_when_it_is_the_smaller_of_the_two(tmp_path):
    """A 3-slot pool under a target-8 gate: the effective target is 3, and every item
    still completes -- no deadlock, no spinning."""
    tr = FakeTransport()
    p = Pool(size=3, directory=tmp_path / "p", poll=0.001, poll_max=0.005)
    r = run(items(40), req, model=MODEL, transport=tr, pool=p, workers=8,
            gate_target=8, gate_floor=8, log=QUIET, sleep=NOSLEEP)
    assert len(r.results) == 40 and not r.errors
    assert p.peak_held <= 3 and r.gate.peak <= 3


def test_gate_still_caps_below_the_pool(tmp_path):
    tr = FakeTransport()
    p = Pool(size=190, directory=tmp_path / "p")
    r = run(items(30), req, model=MODEL, transport=tr, pool=p, workers=16,
            gate_target=4, gate_floor=4, log=QUIET, sleep=NOSLEEP)
    assert r.gate.peak <= 4 and p.peak_held <= 4


def test_observation_line_carries_pool_fields(tmp_path):
    obs = tmp_path / "obs.jsonl"
    p = Pool(size=190, directory=tmp_path / "p")
    run(items(3), req, model=MODEL, transport=FakeTransport(), pool=p, workers=2,
        gate_target=2, gate_floor=2, obs_path=obs, log=QUIET, sleep=NOSLEEP)
    rows = [json.loads(x) for x in obs.read_text().splitlines()]
    assert set(rows[-1]) >= {"pool_slots_held", "pool_waiters", "pool_total_held"}
    assert rows[-1]["pool_total_held"] == 0   # every request finished


def test_status_line_names_the_holder(tmp_path):
    p = Pool(size=4, directory=tmp_path / "p")
    with p.slot():
        line = p.status_line()
    assert "1/4 slots held" in line and f"pid {os.getpid()}: 1" in line
    assert "0/4" in p.status_line()


def test_slot_files_above_the_ceiling_are_reported(tmp_path):
    for q in Pool(size=6, directory=tmp_path / "p").paths():   # lay down slot_000..005
        q.touch()
    small = Pool(size=3, directory=tmp_path / "p")
    stale = small.stale_files()
    assert [x.name for x in stale] == ["slot_003", "slot_004", "slot_005"]
    assert "3 slot files above the ceiling" in small.status_line()


# --- two processes --------------------------------------------------------

def _env(tmp_path, ceiling):
    e = dict(os.environ)
    e.update(DEEPINFRA_POOL_DIR=str(tmp_path / "pool"), DEEPINFRA_POOL_CEILING=str(ceiling),
             DEEPINFRA_API_KEY="test-key-not-real", PYTHONPATH=str(Path(CHILD).parents[1] / "src"))
    e.pop("DEEPINFRA_POOL", None)
    return e


def test_two_processes_share_the_ceiling_fairly(tmp_path):
    """Two runs, one machine. Neither may exceed the shared ceiling, and over the run each
    should get roughly half of it -- no starvation."""
    # delay is the fake request duration: 150ms, so the pool's <=50ms wake-up poll is a
    # small fraction of it, as it is against a real multi-second DeepInfra call.
    cap, n, workers, delay = 8, 70, 10, 0.15
    outs = [tmp_path / "a.json", tmp_path / "b.json"]
    start = time.time() + 1.0
    procs = [subprocess.Popen([sys.executable, CHILD, "run", str(o), f"{start}",
                               str(n), str(workers), str(delay)], env=_env(tmp_path, cap))
             for o in outs]
    for pr in procs:
        assert pr.wait(120) == 0
    a, b = (json.loads(o.read_text()) for o in outs)

    assert a["done"] == b["done"] == n and a["errors"] == b["errors"] == 0
    # 1. the shared ceiling is never exceeded, as sampled from inside live requests
    assert max(a["max_total_held"], b["max_total_held"]) <= cap
    # 2. each process kept its own workers under the ceiling too
    assert a["peak_held"] <= cap and b["peak_held"] <= cap
    # 3. both really overlapped (otherwise "fair" is meaningless)
    overlap = min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])
    assert overlap > 0.5 * max(a["elapsed"], b["elapsed"])
    # 4. fair split: each held ~cap/2 on average, neither starved
    for r in (a, b):
        assert 0.25 * cap <= r["mean_inflight"] <= 0.75 * cap, (a, b)
    # 5. and the two finished within a factor of ~1.5 of each other
    lo, hi = sorted([a["elapsed"], b["elapsed"]])
    assert hi <= 1.6 * lo, (a, b)
    print(f"\n  A: {a['mean_inflight']:.1f} mean in flight, {a['elapsed']:.1f}s, "
          f"{a['n_waits']} waits;  B: {b['mean_inflight']:.1f}, {b['elapsed']:.1f}s, "
          f"{b['n_waits']} waits;  max total held {max(a['max_total_held'], b['max_total_held'])}/{cap}")


def test_a_killed_process_leaks_no_slots(tmp_path):
    """flock is dropped by the kernel when the process dies, so there is no reaper and no
    stale-entry problem: kill -9 a holder and its slots are free immediately."""
    ready = tmp_path / "ready"
    cap = 4
    pr = subprocess.Popen([sys.executable, CHILD, "hold", str(ready), "3"],
                          env=_env(tmp_path, cap))
    for _ in range(400):
        if ready.exists():
            break
        time.sleep(0.02)
    assert ready.exists(), "holder never started"
    p = Pool(size=cap, directory=tmp_path / "pool")
    assert p.total_held() == 3
    assert [pid for _, pid in p.holders()] == [pr.pid] * 3

    pr.send_signal(signal.SIGKILL)
    pr.wait(10)
    for _ in range(200):                      # the kernel drops the locks; poll briefly
        if p.total_held() == 0:
            break
        time.sleep(0.01)
    assert p.total_held() == 0
    with p.slot():                            # and the slots are usable again
        assert p.total_held() == 1
