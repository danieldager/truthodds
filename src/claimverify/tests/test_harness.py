"""run_claims with a fake verify function: shard resume, budget abort, manifest."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from claimverify import harness
from claimverify.config import LOOP_VERSION, ClaimVerifyConfig, OrchestrationConfig


def fake_pools(**over):
    return SimpleNamespace(cfg=OrchestrationConfig(progress_every=3600, **over),
                           serper=SimpleNamespace(credits=0, metrics=SimpleNamespace(cache_hits=0)),
                           exa=SimpleNamespace(calls=0), status_line=lambda: "fake pools")


def script_verify(monkeypatch, seen: list, cost: float = 0.5, fail=()):
    async def verify(claim, pools, cfg, trace):
        seen.append(claim["claim_id"])
        if claim["claim_id"] in fail:
            raise RuntimeError("boom")
        return {"status": "unsupported", "cost_usd": cost, "llm_calls": 1,
                "llm_cache_hits": 0, "scrapes": 0}
    monkeypatch.setattr(harness, "verify_claim", verify)


def claims(*ids):
    return [{"claim_id": i, "text": f"claim {i}", "date": "2020-01-01"} for i in ids]


@pytest.mark.asyncio
async def test_resume_skips_done_claims(tmp_path, monkeypatch):
    seen: list = []
    script_verify(monkeypatch, seen)
    out = await harness.run_claims(claims("a", "b"), fake_pools(), ClaimVerifyConfig(), tmp_path)
    assert out["ran"] == 2 and out["errors"] == 0
    seen.clear()
    out = await harness.run_claims(claims("a", "b", "c"), fake_pools(), ClaimVerifyConfig(), tmp_path)
    assert out["ran"] == 1 and seen == ["c"]
    assert set(harness.load_records(tmp_path)) == {"a", "b", "c"}
    assert (tmp_path / "trace" / "c.json").exists()


@pytest.mark.asyncio
async def test_failed_claim_is_rerun_on_resume(tmp_path, monkeypatch):
    seen: list = []
    script_verify(monkeypatch, seen, fail={"b"})
    out = await harness.run_claims(claims("a", "b"), fake_pools(), ClaimVerifyConfig(), tmp_path)
    assert out["errors"] == 1
    assert harness.load_records(tmp_path, ok_only=False)["b"]["ok"] is False
    seen.clear()
    script_verify(monkeypatch, seen)
    await harness.run_claims(claims("a", "b"), fake_pools(), ClaimVerifyConfig(), tmp_path)
    assert seen == ["b"]
    assert harness.load_records(tmp_path)["b"]["ok"] is True


@pytest.mark.asyncio
async def test_budget_abort_stops_launching(tmp_path, monkeypatch):
    seen: list = []
    script_verify(monkeypatch, seen, cost=0.5)
    ids = [f"c{i}" for i in range(10)]
    out = await harness.run_claims(claims(*ids), fake_pools(k_claims=1), ClaimVerifyConfig(),
                                   tmp_path, budget_usd=1.0, min_for_projection=2)
    # after 2 completions the projection is 0.5/2*10 = $2.50 > $1 cap: nothing else launches
    assert out["budget_abort"] is True
    assert out["ran"] == 2 and seen == ids[:2]
    assert len(harness.load_records(tmp_path)) == 2


def test_manifest_written_with_config(tmp_path):
    cfg = ClaimVerifyConfig(pages_per_round=10)
    m = harness.write_manifest(tmp_path / "run", cfg=cfg, ocfg=OrchestrationConfig(),
                               dataset_path=None, n=3, arm="all10", seed=707, smoke=None)
    on_disk = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert on_disk == m
    assert on_disk["config"] == cfg.to_dict() and on_disk["config"]["pages_per_round"] == 10
    assert on_disk["arm"] == "all10" and on_disk["n_claims"] == 3
    assert on_disk["loop_version"] == LOOP_VERSION
    assert set(on_disk["prompts_sha256"]) >= {"QUERY_SYSTEM", "READ_SYSTEM", "RESOLVE_SYSTEM"}
    assert on_disk["cache_dir"].endswith("cache")           # the tmp cache from conftest
    assert json.loads((tmp_path / "run" / "config.json").read_text())["loop"] == cfg.to_dict()
