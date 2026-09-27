"""deepinfra_runner — the shared DeepInfra runner (RateGate + cache + sweep).

    from deepinfra_runner import run
    res = run(items, make_request, model=..., cache_path=..., max_spend=2.0)
"""
from .gate import DEFAULT_OBS_PATH, THROTTLE_CODES, Observer, RateGate, default_obs_path
from .pool import Pool, default_pool, pool_dir
from .prices import PRICES, cost_of, price_of, set_price
from .runner import (DEFAULT_WORKERS, MODEL_WORKERS, Cache, PreflightAbort, RunResult,
                     cache_key, preflight_probe, resolve_workers, run)
from .transport import RequestsTransport, base_url

__all__ = ["run", "RateGate", "Observer", "RunResult", "Cache", "cache_key", "resolve_workers",
           "preflight_probe", "PreflightAbort", "MODEL_WORKERS", "DEFAULT_WORKERS",
           "PRICES", "price_of", "cost_of", "set_price", "RequestsTransport", "base_url",
           "DEFAULT_OBS_PATH", "default_obs_path", "THROTTLE_CODES",
           "Pool", "default_pool", "pool_dir"]
__version__ = "0.1.0"
