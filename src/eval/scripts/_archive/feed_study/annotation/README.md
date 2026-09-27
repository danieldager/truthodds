# annotation/ — fast tweet-labeling tool

Builds the **selection-gold set** (FLAG + V/R/H per tweet) for the post-discrimination
classifier. See `src/clog/TASKS.md` #1 and `../docs/classifier_dataset_methods.md`.

Designed so a **non-technical** annotator needs only a browser and email: you send
them one self-contained `.html`, they label, they email back one `.json`.

## Round-trip

```
make_tasks.py  →  labels_A.html        (you email this)
                       ↓ annotator opens in browser, labels, "Download results"
                  labels_A.json         (they email back)
                       ↓ drop all returned files in one folder
collect.py     →  labels.jsonl + agreement report
```

## 1. Generate per-annotator files

```bash
# smoke: one annotator, 20 home-feed posts
uv run make_tasks.py --source /path/to/zeerover --op HomeTimeline \
    --annotators A --limit 20 --out out

# real: 3 annotators, 150-post shared overlap, rest partitioned disjointly
uv run make_tasks.py --source /path/to/zeerover --op HomeTimeline \
    --annotators A B C --overlap 150 --out out
```

- `--source` a zeerover dir (globs `x_capture_*.ndjson`) or a single `.ndjson`.
- `--everyone-labels-all` gives every annotator every post (full multi-labelling)
  instead of partition+overlap.
- Conversation threads (`conversation_id`) are kept intact and never split across
  annotators. `--seed` makes the split reproducible.

Email each `out/labels_<id>.html`. The file embeds its tweets; no network needed.

## 2. The annotator's experience

Open the file, click **Start**. Two short passes (order fixed per annotator):

- **FLAG pass** — one light per post. `↑` = ON (our tool would catch & check it),
  `↓` = OFF (ignore). One keystroke, auto-advances.
- **CATEGORIES pass** — three lights, `←` Relevant · `↓` Verifiable · `→` Harmful.
  Tap a key to flip its light (tap again = **unsure**). **Enter** commits all three
  and advances — an all-"no" post is a single Enter.

Any time: **`g`** opens the in-app **Guidelines** page (FLAG/V/R/H definitions +
examples, mirrors `../docs/labeling_guideline.md`); **`,`** opens **Keys** to rebind
the category arrows (middle defaults to `↓`; persisted per browser); **`i`** marks a
post **image-dependent** (the captures are text-only — see "Images" below).

The two passes are deliberately separate (and re-shuffled) so the holistic FLAG
judgment isn't contaminated by V/R/H. Progress auto-saves to the browser; they can
close and reopen. **Undo** = Backspace. At the end, **Download results** →
`labels_<id>.json`.

## Images

The zeerover captures carry **text only** — no media — and the production gate
(mmBERT) is also text-only, so annotators label **what the text says**. When the text
alone is insufficient, they mark the post **image-dependent** (`i`); `collect.py`
reports the rate. To actually show images later, zeerover would need to capture media
URLs (a separate change) so thumbnails could be embedded.

## 3. Collect + agreement

```bash
uv run collect.py --in returned/ --out labels.jsonl
```

Writes `labels.jsonl` (one row per post × annotator) and prints, on the overlap
subset: pairwise **Cohen's κ** + **Fleiss' κ** per dimension, the disagreements to
adjudicate, and median seconds/post per pass.

## Files

| file | role |
|---|---|
| `template.html` | the tool UI/logic; `__TASK_DATA__` is replaced with embedded tweets |
| `make_tasks.py` | tweets → per-annotator `.html` (split, two-pass orders, overlap) |
| `collect.py` | returned `.json` → `labels.jsonl` + κ / disagreement / speed report |

## Not done yet

- **LLM prefill.** The tool reads an optional per-post `prefill` (ENTER accepts the
  suggested label) but ships empty. Populate it from the Phase-0 LLM-teacher prompts
  (TASKS #1) to turn labeling into accept/fix.
- **Image thumbnails** — would require zeerover to capture media URLs first.
- **Train/dev/test group-split** of the collected labels (downstream of this tool).
