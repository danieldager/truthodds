# deepinfra_runner

**The** runner for every DeepInfra job in every project. One gated, cached, retried, swept
pass over a list of items. If you are about to write another `ThreadPoolExecutor` loop around
`chat/completions`, use this instead.

It generalises `factchecking_with_LLMs/src/eval/scripts/build_eval/reader_lab.py`, whose
`RateGate` is carried over verbatim — same defaults, same arithmetic, same observation schema,
same preflight.

## What it does

- **RateGate** — holds a target in flight and reacts only to the *rate* of 429s over a trailing
  60s window, never to a single 429. Defaults: `--workers 200`, target **160**, ceiling **200**
  (DeepInfra's documented per-model account limit), floor **32**. Cut ×0.8 when the windowed
  429 rate exceeds **5%** with a sample of ≥20; recover +10% every clean 30s back to the ceiling.
- **Per-request retries** — 429 / 5xx / transport failure is retried on the *same* request with
  jittered exponential backoff (1, 2, 4, 8, 16, capped 30s), 6 attempts, then the item is failed.
- **Per-item JSONL cache** — keyed by `sha256(item id | model | canonical request body)`. A rerun
  costs $0; a killed run resumes. Failures are never cached.
  The cache stores the *parsed* result and the key does not cover `parse`, so a changed
  `parse` wants a new cache file.
- **End-of-run sweep** — every failed item is re-run, up to 2 sweeps, so no run ends with
  silently missing items.
- **Hard spend cap** — `max_spend` in $. Cost from `usage.estimated_cost` when the provider
  gives one, else a price table with an override hook (`set_price` / `prices=`).
- **Progress + observations** — a progress line with done/total, %, $ so far, elapsed, rate and
  ETA; and a per-minute observation line appended to
  `~/.cache/deepinfra_runner/gate_observations.jsonl` (override with `obs_path=` or
  `$DEEPINFRA_RUNNER_OBS`), so every run doubles as a concurrency probe.
- **Per-model worker override** — `MODEL_WORKERS`: Qwen3-235B runs at 8, GLM-5 / Thinking at 16,
  everything else at 200.

## Install into a project

```bash
uv add --editable path/to/tools/deepinfra
```

or as a path dependency in `pyproject.toml`:

```toml
dependencies = ["deepinfra-runner"]

[tool.uv.sources]
deepinfra-runner = { path = "../tools/deepinfra", editable = true }
```

Key from `$DEEPINFRA_API_KEY` (or `$LLM_API_KEY`); base URL from `$DEEPINFRA_BASE_URL` /
`$LLM_BASE_URL`, default `https://api.deepinfra.com/v1/openai`.

## Usage

```python
from deepinfra_runner import run

items = [{"id": r["uid"], "text": r["passage"]} for r in rows]

def make_request(it):                       # one request body per item
    return {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": it["text"]}],
            "temperature": 0, "max_tokens": 1000,
            "response_format": {"type": "json_object"}}

res = run(items, make_request,
          model="deepseek-ai/DeepSeek-V4-Flash",
          cache_path="results/extract.cache.jsonl",
          parse=lambda j, it: json.loads(j["choices"][0]["message"]["content"]),
          max_spend=5.0)                    # workers/gate default per model

print(res.summary())                        # "1842 ok, 0 failed, 310 cached, $1.9031 in 41s"
res.results["uid-7"]                        # parsed output per item id
```

## CLI

```bash
deepinfra-run --preflight                       # 96 tiny calls at 96 in flight, abort if 429 > 20%
deepinfra-run status                            # key/base/workers/prices + last observations
deepinfra-run --items x.jsonl --prompt p.txt --model M --out y.jsonl
```

`--items` is a JSONL with an `id` and a text field (`--text-field`, default `text`); `--prompt`
is a file holding the system prompt. `--max-spend`, `--workers`, `--json-mode`, `--limit`,
`--preflight` all apply.

## Measured facts

- On a quiet pool **DeepSeek-V4-Flash sustains ~105 reads/s at 200 in flight, 0 throttles**
  (factchecking, 2026-09-14). The old "64 workers" cap was self-imposed storm avoidance, not a
  ceiling.
- DeepInfra returned **zero 429s up to 240 concurrent** on Mistral-Small with 5-token and with
  300-token completions (policytrace probe, 2026-09-12, `results/deepinfra_probe/`). The binding
  constraint is throughput, which peaks around **~128 concurrent** under real generation.
- DeepInfra's 429 `Model busy` is transient pool contention — it fires "even if you're under the
  limit", is time-varying and per-model, and carries **no rate-limit headers**. So: retry each
  request, control on the rate, never cut on one 429.
- Run `--preflight` before a session's first paid leg. Read the observations file before changing
  any gate default.

## Per-model caveat (Qwen3-235B and friends)

Big MoE models have a per-model **throughput** budget, not a request-rate one, and get *worse*
under concurrency: 12 concurrent Qwen3-235B extraction calls (~1.9k in / 8k max out) ran at a
**148s median and timed out 3–4 of 12** at a 240s timeout, where the same 12 at 4 concurrent had
**0 errors**. `MODEL_WORKERS` therefore caps Qwen3-235B at **8** workers (GLM-5 / Thinking at 16)
and the gate target follows the worker count. A slow run on a big model is almost never the pool
— check the model first. Also: a high `max_tokens` costs concurrency, because DeepInfra admits a
request against the tokens it *reserves*, not the tokens it returns.

## Several experiments at once

RateGate governs **one process**. The DeepInfra limit is per **account**, so two runs at once
each target 160-190 in flight, the account is overshot, both meet a 429 storm and the unlucky
one is cut to its floor (2026-09-23: two experiments, one finished, the other ended at target 32
with 23 failed reads). So the runner also holds a **machine-wide slot pool**: a directory of
`flock`'d slot files under `~/.cache/deepinfra_runner/pool/`, one slot taken for the duration of
one in-flight HTTP request. N concurrent runs share one ceiling instead of each taking it.

```bash
deepinfra-run --pool-status         # ceiling, slots held by pid, slot files above the ceiling
DEEPINFRA_POOL_CEILING=190          # pool size (default)
DEEPINFRA_POOL_DIR=~/.cache/deepinfra_runner/pool
DEEPINFRA_POOL=0                    # opt out entirely
```

**It composes with the gate rather than fighting it** — which is what the earlier objection to
folding in policytrace's pool was about. A worker takes a *slot first*, then asks the gate for a
seat *non-blockingly*; if the gate is full it hands the slot straight back, and if no slot is free
it never touches the gate at all. Therefore:

- a worker waiting on the pool is **not** counted in flight, so the gate's effective target is
  `min(own target, slots it can actually get)` and `peak` stays honest;
- a run whose gate has cut itself to 32 cannot sit on 200 slots behind that cut;
- retries are jittered, which is what splits the pool evenly between competing processes;
- the gate still owns the 429 reaction; the pool only caps the machine.

**One run alone behaves exactly as today.** 190 slots against at most 200 workers (and a gate
target of 160) means the first sweep always finds a free slot and nobody ever waits — asserted in
the tests, which run with the pool **on**.

**Death is handled by the kernel.** flock is dropped when the holding process dies, so a killed
run leaks nothing, there is no reaper and there is no stale state to clean up (there is a test
that `kill -9`s a holder and watches the slots free).

**Cost:** a waiter wakes within <=50 ms of a slot freeing, ~1 % of a multi-second DeepInfra call.
The pool is not meant for millisecond requests.

**Observability.** The per-minute observation line carries `pool_slots_held` and `pool_waiters`
(this process) and `pool_total_held` (every process), and the progress line ends with
`pool 87/190 +3 waiting` -- so a slow run says whether the cause is machine contention.

**Per machine, not per account.** The laptop and the mini each keep their own directory: when both
run jobs, split the budget by env (e.g. `DEEPINFRA_POOL_CEILING=140` and `=50`). Cross-machine
coordination would need a server and is deliberately not built. All processes on one machine
should agree on the ceiling -- the effective cap is the largest value any live process started with.

**The pool is not a per-model knob.** It stops two runs from oversubscribing the account; it does
nothing about a big MoE model's throughput budget. Heavy-model runs still need their own low
`--workers` (see above).

## Tests

```bash
cd tools/deepinfra && uv run pytest
```

37 offline tests against a fake transport: cache hit, key stability, 429 retry, transport-error
retry, bounded attempts, spend cap (and that capped items are not swept), price table + override,
failed-item sweep and its give-up path, per-model worker override, observation schema, parse hook,
the RateGate arithmetic on an injected clock, and the CLI end to end. Plus the slot pool: the
ceiling, gate composition both ways round, release on exception, one-process-is-unchanged (a
transport barrier proves 8 really are in flight), two subprocesses contending for an 8-slot pool
(49 % / 51 % of the pool each, total held never above 8), and a `kill -9`'d holder leaking nothing.
