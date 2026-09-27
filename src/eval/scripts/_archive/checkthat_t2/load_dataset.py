"""Load the CheckThat! 2025 Task 2 English dev split.

Clones the official gitlab repo into .data/ on first run, then reads the CSV.
"""
from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://gitlab.com/checkthat_lab/clef2025-checkthat-lab.git"
HERE = Path(__file__).parent
CACHE = HERE / ".data" / "clef2025-checkthat-lab"
DEV_ENG_CSV = CACHE / "task2" / "data" / "dev" / "dev-eng.csv"


@dataclass
class Example:
    post: str
    gold: str


def _ensure_dataset() -> None:
    if DEV_ENG_CSV.exists():
        return
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(CACHE)],
        check=True,
    )


def load_eng_dev(limit: int | None = None) -> list[Example]:
    _ensure_dataset()
    out: list[Example] = []
    with open(DEV_ENG_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(Example(post=row["post"], gold=row["normalized claim"]))
            if limit is not None and len(out) >= limit:
                break
    return out


if __name__ == "__main__":
    ex = load_eng_dev(limit=3)
    for e in ex:
        print(f"POST ({len(e.post)} chars): {e.post[:120]}...")
        print(f"GOLD: {e.gold}")
        print()
    print(f"Total in full dev set: {len(load_eng_dev())}")
