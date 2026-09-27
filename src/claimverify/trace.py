"""Per-claim trace: EVERY intermediate the loop produced, so any verdict can be replayed
and audited offline without re-running.

  trace/<claim_id>.json   CallRecords (each LLM call: messages, raw output, usage,
                          latency, attempts, cache hit), search records (payload with
                          the exact q and tbs, raw JSON), scrape records, the numbered
                          block each READ saw, the dossier text each RESOLVE saw,
                          ledger transitions, config snapshot.
  pages/<sha256(url)>.txt content-addressed scraped text, deduplicated across claims
                          and arms; pages/index.jsonl = {sha, url, chars, source,
                          first_seen_claim}.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


class PageStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.jsonl"
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if self._index.exists():
            for line in self._index.open():
                try:
                    self._seen.add(json.loads(line)["sha"])
                except Exception:
                    pass

    @staticmethod
    def sha(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def put(self, url: str, text: str, source: str, claim_id: str) -> str:
        sha = self.sha(url)
        with self._lock:
            if sha in self._seen:
                return sha
            (self.root / f"{sha}.txt").write_text(text)
            with self._index.open("a") as f:
                f.write(json.dumps({"sha": sha, "url": url, "chars": len(text),
                                    "source": source, "first_seen_claim": claim_id}) + "\n")
            self._seen.add(sha)
        return sha


class Trace:
    """Collects one claim's intermediates. Passed into every pool call as `trace=`."""

    def __init__(self, claim_id: str, pages: PageStore | None = None):
        self.claim_id = claim_id
        self.pages = pages
        self.llm_calls: list[dict] = []
        self.searches: list[dict] = []
        self.scrapes: list[dict] = []
        self.reads: list[dict] = []       # {doc, url, block, stats}
        self.resolves: list[dict] = []    # {round, dossier_text, user_message}
        self.ledger: list[dict] = []      # {round, before, after, guards}
        self.config: dict = {}
        self.notes: list[dict] = []

    def page(self, url: str, text: str, source: str) -> str | None:
        if self.pages is None or not text:
            return None
        return self.pages.put(url, text, source, self.claim_id)

    def to_dict(self) -> dict:
        return {"claim_id": self.claim_id, "config": self.config,
                "llm_calls": self.llm_calls, "searches": self.searches,
                "scrapes": self.scrapes, "reads": self.reads, "resolves": self.resolves,
                "ledger": self.ledger, "notes": self.notes}

    def dump(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{self.claim_id}.json"
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False))
        return p
