"""DeepInfra list prices, $ per 1M tokens (prompt, completion), with an override hook.

The provider returns `usage.estimated_cost` on most models; this table is the fallback
and the basis of the spend cap when it does not. Prefix matching, longest first, so a
family entry covers its variants."""
from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.09, 0.55),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": (0.23, 2.30),
    "deepseek-ai/DeepSeek-V4-Flash": (0.09, 0.18),
    "deepseek-ai/DeepSeek-V4-Pro": (1.30, 2.60),
    "zai-org/GLM-5.3": (1.20, 4.00),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": (0.075, 0.20),
    "moonshotai/Kimi-K2.6": (0.75, 3.50),
    "moonshotai/Kimi-K2.5": (0.45, 2.25),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "BAAI/bge-m3": (0.010, 0.0),
}

# Unknown model: assume the cheap-instruct tier rather than 0, so the spend cap still bites.
FALLBACK = (0.09, 0.18)


def set_price(model: str, prompt_per_m: float, completion_per_m: float) -> None:
    """Override / add a model's price (the hook for a project with negotiated rates)."""
    PRICES[model] = (prompt_per_m, completion_per_m)


def price_of(model: str, prices: dict | None = None) -> tuple[float, float]:
    table = {**PRICES, **(prices or {})}
    if model in table:
        return table[model]
    for k in sorted(table, key=len, reverse=True):   # longest prefix wins
        if model.startswith(k) or k in model:
            return table[k]
    return FALLBACK


def cost_of(model: str, usage: dict | None, prices: dict | None = None) -> float:
    """$ for one call. Uses the provider's estimated_cost when it gives one."""
    usage = usage or {}
    est = usage.get("estimated_cost")
    if est is not None:
        return float(est)
    pin, pout = price_of(model, prices)
    pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return (pt * pin + ct * pout) / 1e6
