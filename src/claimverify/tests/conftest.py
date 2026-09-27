"""Every test runs against a throwaway disk cache: nothing reads or writes src/pipeline/.cache."""
from __future__ import annotations

import pytest

from claimverify import disk_cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path):
    old = disk_cache.get_dir()
    disk_cache.set_dir(tmp_path / "cache")
    yield
    disk_cache.set_dir(old)
