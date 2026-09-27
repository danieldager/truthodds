"""Machine-wide concurrency pool: one flock'd slot per in-flight HTTP request.

RateGate governs ONE process. The DeepInfra limit is per *account*, so when two
experiments run at once each gate independently targets 160-190 in flight, the
account is overshot, both see a 429 storm, and the unlucky one is cut to its floor
(observed 2026-09-23: two runs, one finished, the other ended at target 32 with 23
failed reads). This module is the missing coordination: a directory of N lock files
under `~/.cache/deepinfra_runner/pool/`, one taken for the duration of one in-flight
request, so N concurrent runs share one ceiling instead of each taking it.

The two controllers compose instead of fighting, which is what the earlier
"pool vs gate" objection was about:

  - a worker takes a SLOT first, then asks the gate non-blockingly; if the gate is full
    it hands the slot straight back. So a worker waiting on the pool never touches the
    gate (it is not counted in flight, and the gate's effective target becomes
    min(own target, slots it can get)), and a run whose gate has cut itself cannot hoard
    the machine's slots behind that cut;
  - the gate still owns the 429 reaction; the pool only caps the machine.

Slots are `fcntl.flock`'d fds. The kernel drops the lock when the fd closes *or when
the process dies*, so a killed run leaks nothing and there is no reaper.

Config (env):
    DEEPINFRA_POOL_CEILING   pool size, default 190
    DEEPINFRA_POOL_DIR       default ~/.cache/deepinfra_runner/pool
    DEEPINFRA_POOL=0         opt out entirely

Per machine, not per account: if the laptop and the mini both run jobs, split the
budget between them by env. All processes on one machine should agree on the ceiling
-- the effective cap is the largest value any live process was started with.

Status:  deepinfra-run --pool-status
"""
from __future__ import annotations

import fcntl
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CEILING = 190
DEFAULT_DIR = "~/.cache/deepinfra_runner/pool"


def enabled() -> bool:
    return os.environ.get("DEEPINFRA_POOL", "1").lower() not in ("0", "false", "no")


def ceiling() -> int:
    return max(1, int(os.environ.get("DEEPINFRA_POOL_CEILING") or DEFAULT_CEILING))


def pool_dir() -> Path:
    return Path(os.environ.get("DEEPINFRA_POOL_DIR") or DEFAULT_DIR).expanduser()


class Pool:
    """A machine-wide slot semaphore over flock'd files. One instance per process is
    enough (`default_pool()`); the counters it keeps -- held, waiters -- are this
    process's share, and `total_held()` counts every process's."""

    def __init__(self, size=None, directory=None, on=None, poll=0.005, poll_max=0.05,
                 sleep=time.sleep):
        self.size = int(size) if size else ceiling()
        self.dir = Path(directory).expanduser() if directory else pool_dir()
        self.on = enabled() if on is None else bool(on)
        self.poll = poll
        self.poll_max = poll_max
        self._sleep = sleep
        self._paths = None
        self._lock = threading.Lock()
        self.held = 0          # slots this process holds right now
        self.waiters = 0       # workers of this process polling for a slot
        self.peak_held = 0
        self.n_waits = 0       # acquisitions that had to poll at least once

    def paths(self):
        if self._paths is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._paths = [self.dir / f"slot_{i:03d}" for i in range(self.size)]
        return self._paths

    def _take(self, paths):
        """One non-blocking sweep over the slot files from a random start (so a crowd of
        waiters does not queue on the same file). Returns a locked fd, or None."""
        start = random.randrange(len(paths))
        for k in range(len(paths)):
            p = paths[(start + k) % len(paths)]
            fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            try:
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()}\n".encode())
            except OSError:
                pass
            return fd
        return None

    @contextmanager
    def slot(self, gate=None):
        """Hold one machine-wide slot (and, if given, one gate seat) for the block.

        Order is slot-then-gate, both non-blocking: a worker waiting for the pool never
        touches the gate, so it is not counted in flight (the gate's effective target
        becomes min(own target, slots it can get)); and a worker that gets a slot the gate
        has no room for gives the slot straight back instead of hoarding it. Retries are
        jittered, which is what splits the pool evenly between competing processes.
        No-op when the pool is off."""
        if not self.on:
            if gate is None:
                yield
            else:
                with gate:
                    yield
            return
        paths = self.paths()
        fd, wait, waiting = None, self.poll, False
        try:
            while True:
                fd = self._take(paths)
                if fd is not None:
                    if gate is None or gate.try_acquire():
                        break
                    os.close(fd)               # gate full: hand the slot back at once
                    fd = None
                if not waiting:
                    waiting = True
                    with self._lock:
                        self.waiters += 1
                        self.n_waits += 1
                self._sleep(wait * (0.5 + random.random()))
                wait = min(self.poll_max, wait * 1.4)
        finally:
            if waiting:
                with self._lock:
                    self.waiters -= 1
        with self._lock:
            self.held += 1
            self.peak_held = max(self.peak_held, self.held)
        try:
            yield
        finally:
            os.close(fd)                       # releases the flock
            with self._lock:
                self.held -= 1
            if gate is not None:
                gate.release()

    def total_held(self) -> int:
        """Slots held right now by every process on this machine."""
        return len(self.holders())

    def holders(self):
        """[(slot index, holder pid or None)] over the slots locked right now."""
        if not self.on:
            return []
        out = []
        for i, p in enumerate(self.paths()):
            try:
                fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o644)
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:                          # someone holds it
                try:
                    pid = int(os.pread(fd, 32, 0).split()[0])
                except (ValueError, IndexError):
                    pid = None
                out.append((i, pid))
            finally:
                os.close(fd)
        return out

    def stale_files(self):
        """Slot files on disk beyond the current ceiling -- left by a process started with
        a bigger DEEPINFRA_POOL_CEILING. They still govern that process, so they are worth
        seeing; they are not leaked locks (flock has none)."""
        if not self.dir.exists():
            return []
        return sorted(p for p in self.dir.glob("slot_*")
                      if p.name not in {q.name for q in self.paths()})

    def line(self):
        """Cheap, in-memory, for the progress line: no syscalls."""
        return f"pool {self.held}/{self.size}" + (f" +{self.waiters} waiting" if self.waiters else "")

    def status_line(self):
        if not self.on:
            return f"pool: DISABLED (DEEPINFRA_POOL=0); nominal ceiling {self.size}"
        held = self.holders()
        by_pid = {}
        for _, pid in held:
            by_pid[pid] = by_pid.get(pid, 0) + 1
        who = ", ".join(f"pid {pid}: {n}" for pid, n in sorted(by_pid.items(), key=lambda x: -x[1]))
        stale = self.stale_files()
        return (f"pool: {len(held)}/{self.size} slots held"
                + (f" ({who})" if who else "")
                + f"  dir {self.dir}"
                + (f"  [{len(stale)} slot files above the ceiling: {stale[0].name}..{stale[-1].name}]"
                   if stale else ""))


_default = {}
_default_lock = threading.Lock()


def default_pool() -> Pool:
    """The process-wide Pool, one per (dir, ceiling, on) so the held/waiters counters
    aggregate across concurrent run()s in the same process."""
    key = (str(pool_dir()), ceiling(), enabled())
    with _default_lock:
        if key not in _default:
            _default[key] = Pool()
        return _default[key]


def status_line() -> str:
    return default_pool().status_line()


if __name__ == "__main__":
    print(status_line())
