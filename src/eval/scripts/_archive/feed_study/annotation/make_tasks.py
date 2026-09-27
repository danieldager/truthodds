"""Generate self-contained per-annotator labeling files from a tweet source.

Each output is ONE `labels_<id>.html` you can email to an annotator: they open it
in any browser (no install), flip the FLAG / R-V-H switches, and click "Download
results" to get a `labels_<id>.json` they email back. `collect.py` merges those.

Labour split (default): a shared OVERLAP subset goes to EVERY annotator (this is
where collect.py measures inter-annotator agreement), and the remaining posts are
partitioned DISJOINTLY among annotators. Use --everyone-labels-all to instead give
every annotator every post (full multi-labelling). Conversation threads are kept
together so the same thread never splits across annotators.

Per annotator we fix the two-pass design: a randomized pass order (FLAG-first vs
categories-first) and two INDEPENDENT shuffles (gate_order, cat_order) so the
holistic gate is collected without exposure to V/R/H.

Example (smoke, one annotator, 20 home-feed posts):
  uv run make_tasks.py --source ../../../../zeerover --op HomeTimeline \
      --annotators A --limit 20 --out out

Real run (3 annotators over a sample, 150 shared overlap):
  uv run make_tasks.py --source ../../../../zeerover --op HomeTimeline \
      --annotators A B C --overlap 150 --out out
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TEMPLATE = Path(__file__).parent / "template.html"

# Flat zeerover ndjson fields we surface to the annotator + keep for analysis.
def load_posts(source: Path, op: str | None) -> list[dict]:
    files = sorted(source.glob("x_capture_*.ndjson")) if source.is_dir() else [source]
    if not files:
        raise SystemExit(f"no x_capture_*.ndjson under {source}")
    seen, posts = set(), []
    for fp in files:
        # Split on "\n" only — NOT str.splitlines(), which also breaks on Unicode
        # line separators (  etc.) that appear inside tweet text and would
        # shatter a single JSON record across "lines".
        for line in fp.read_text().split("\n"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("id")
            if not pid or pid in seen:
                continue
            if op and r.get("operation") != op:
                continue
            text = (r.get("full_text") or "").strip()
            if not text:
                continue
            seen.add(pid)
            posts.append({
                "post_id": pid,
                "text": text,
                "screen_name": r.get("screen_name"),
                "lang": r.get("lang"),
                "created_at": r.get("created_at"),
                "operation": r.get("operation"),
                "group_id": r.get("conversation_id") or pid,   # keep threads together
                "prefill": None,                                # populated later by the LLM teacher
            })
    return posts


def partition(posts: list[dict], annotators: list[str], overlap: int,
              everyone_all: bool, rng: random.Random) -> dict[str, list[dict]]:
    """Return {annotator_id: [posts...]}. Threads (group_id) stay intact."""
    if everyone_all:
        return {a: list(posts) for a in annotators}

    # Bucket by thread so a conversation is one indivisible unit.
    groups: dict[str, list[dict]] = {}
    for p in posts:
        groups.setdefault(p["group_id"], []).append(p)
    units = list(groups.values())
    rng.shuffle(units)

    # Peel off whole threads until we reach the overlap target.
    shared, rest = [], []
    for u in units:
        if len(shared) < overlap:
            shared.extend(u)
        else:
            rest.append(u)
    # Round-robin the remaining threads across annotators.
    chunks: dict[str, list[dict]] = {a: [] for a in annotators}
    for i, u in enumerate(rest):
        chunks[annotators[i % len(annotators)]].extend(u)
    return {a: shared + chunks[a] for a in annotators}


def build_task(annotator: str, posts: list[dict], rng: random.Random) -> dict:
    n = len(posts)
    pass_order = ["gate", "cats"]
    if rng.random() < 0.5:                     # randomize which pass comes first
        pass_order = ["cats", "gate"]
    gate_order = list(range(n)); rng.shuffle(gate_order)
    cat_order = list(range(n));  rng.shuffle(cat_order)
    return {
        "annotator": annotator,
        "pass_order": pass_order,
        "gate_order": gate_order,
        "cat_order": cat_order,
        "posts": posts,
    }


def write_html(task: dict, out_dir: Path) -> Path:
    template = TEMPLATE.read_text()
    # Embed as JSON in a <script type="application/json">; escape "<" so a tweet
    # containing "</script>" can't break out of the tag.
    payload = json.dumps(task, ensure_ascii=False).replace("<", "\\u003c")
    html = template.replace("__TASK_DATA__", payload)
    out = out_dir / f"labels_{task['annotator']}.html"
    out.write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="zeerover dir (or a single .ndjson)")
    ap.add_argument("--op", default=None, help="filter by operation, e.g. HomeTimeline")
    ap.add_argument("--annotators", nargs="+", required=True, help="annotator ids, e.g. A B C")
    ap.add_argument("--overlap", type=int, default=150, help="posts shared by all annotators")
    ap.add_argument("--everyone-labels-all", action="store_true",
                    help="give every annotator every post (full multi-labelling)")
    ap.add_argument("--limit", type=int, default=None, help="cap total posts (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    posts = load_posts(args.source, args.op)
    rng.shuffle(posts)
    if args.limit:
        posts = posts[: args.limit]
    if not posts:
        raise SystemExit("no posts after filtering")

    overlap = 0 if args.everyone_labels_all else min(args.overlap, len(posts))
    split = partition(posts, args.annotators, overlap, args.everyone_labels_all, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    for a in args.annotators:
        task = build_task(a, split[a], rng)
        out = write_html(task, args.out)
        print(f"{out}  ({len(split[a])} posts, pass_order={task['pass_order']})")
    print(f"\ntotal pool: {len(posts)} posts"
          + ("" if args.everyone_labels_all else f", overlap={overlap} shared by all"))


if __name__ == "__main__":
    main()
