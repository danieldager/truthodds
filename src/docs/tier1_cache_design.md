# Tier-1 verdict cache — design  →  MOVED

> **ARCHIVED 2026-07-28.** Tier 1/2 were specified but never wired into the live loop
> (`verify_tweet_claims.py` imports none of them), so `cache.py`, `embedding.py`,
> `models.py` and `fact_api.py` moved to `src/_archive/tier1_tier2_unbuilt/` to keep
> `pipeline/` to what actually runs. This design is unchanged and still the plan of
> record for when the tier gets built.


**This spec was merged into [`cache_design.md`](cache_design.md) on 2026-06-26.**

The cache is now documented in one place as **"the cache"** (the "Tier-1" label is dropped). The
full, current design — text-claim matching, the equivalence gate + slot-veto, the multimodal
post-space + pHash index, the data model, TTL policy, storage, parameters, and build status —
lives in `cache_design.md`.

*This stub is kept only because `pipeline/config.py` and `pipeline/cache.py` still reference this
filename in comments; redirect those to `cache_design.md` the next time that code is touched.*
