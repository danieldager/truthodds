"""Apply an exported edit set from the dev-500 review doc back into the source.

The review doc (build_full_review.py) lets Daniel edit every piece of narrative text in
place: snapshot intro, per-example intros, step explanations, assumptions, open questions,
the gauntlet description, and the prompts. Clicking "export" produces a JSON blob of edits.
This script reads that blob and writes the changes back into build_full_review.py, so the
next build carries them. Round trip: edit in the browser -> export -> run this -> rebuild.

Edit ids the doc emits:
  snap::intro                          the snapshot intro paragraph
  snap::<pid>::intro                   one example's intro line
  snap::<pid>::<step>::x               a step explanation
  snap::<pid>::<step>::asm<i>          an assumption note   (text or __DEL__)
  snap::<pid>::<step>::oq<i>           an open question     (text or __DEL__)
  gauntlet::text                       the gauntlet description
  prompt::<NAME>                       a live pipeline prompt
  add::<pid>::<i>                      a note Daniel added

Doc narrative is applied automatically. PROMPT edits are NOT applied by default: those
strings are live pipeline behaviour, so they are printed as a diff and only written with
--prompts, and never without showing what changes. Added notes are reported rather than
inserted, since they have no home in the SNAP structure until we decide where they belong.

  cd src && uv run python eval/scripts/claim_sourcing/apply_review_edits.py edits.json
      [--prompts] [--dry-run]
"""
import argparse
import ast
import difflib
import json
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
BUILDER = Path(__file__).parent / "build_full_review.py"
# where each prompt name lives, so a prompt edit can be written back to the real code
PROMPT_FILES = {
    "Pass 1 — extract (system prompt)": (SRC / "eval/scripts/claim_sourcing/extract_tweet_claims.py", "SYSTEM"),
    "Pass 2 — audit / normalize (system prompt)": (SRC / "eval/scripts/claim_sourcing/normalize_tweet_claims.py", "SYSTEM"),
}


def _py_str(value: str) -> str:
    """Render a Python string literal that survives a round trip."""
    if "\n" in value:
        body = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        return f'"""{body}"""'
    return json.dumps(value, ensure_ascii=False)


def _find_snap(src: str):
    """Byte range of the SNAP list literal plus its parsed value."""
    start = src.index("SNAP = [")
    open_at = src.index("[", start)
    depth, i = 0, open_at
    while i < len(src):
        if src[i] == "[":
            depth += 1
        elif src[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    lit = src[open_at:i + 1]
    return open_at, i + 1, ast.literal_eval(lit)


def apply_edits(edits: list[dict], write_prompts: bool, dry: bool) -> None:
    src = BUILDER.read_text()
    s0, s1, snap = _find_snap(src)
    by_pid = {e["pid"]: e for e in snap}

    applied, skipped, prompt_edits, added = [], [], [], []
    intro_new = gaunt_new = None

    for ed in edits:
        eid, text = ed.get("id", ""), ed.get("text")
        removed = ed.get("removed") or text == "__DEL__"
        if ed.get("added") or eid.startswith("add::"):
            added.append(ed)
            continue
        if eid == "snap::intro":
            intro_new = text
            applied.append(eid)
            continue
        if eid == "gauntlet::text":
            gaunt_new = text
            applied.append(eid)
            continue
        if eid.startswith("prompt::"):
            prompt_edits.append((eid[len("prompt::"):], text))
            continue
        m = re.match(r"snap::(\d+)::(.+)$", eid)
        if not m:
            skipped.append((eid, "unrecognised id"))
            continue
        pid, rest = m.groups()
        ex = by_pid.get(pid)
        if ex is None:
            skipped.append((eid, f"no example {pid}"))
            continue
        if rest == "intro":
            if removed:
                ex.pop("intro", None)      # drop the line, keep the example
            else:
                ex["intro"] = text
            applied.append(eid)
            continue
        m2 = re.match(r"([^:]+)::(x|asm(\d+)|oq(\d+))$", rest)
        if not m2:
            skipped.append((eid, "unrecognised step id"))
            continue
        step_id, field = m2.group(1), m2.group(2)
        step = next((st for st in ex["steps"] if st.get("id") == step_id), None)
        if step is None:
            skipped.append((eid, f"no step {step_id}"))
            continue
        if field == "x":
            # removing a step's text drops the prose but KEEPS the step: the data it
            # renders (the post, the results table, the ledger) is the point
            step["x"] = "" if removed else text
        else:
            key, idx = ("asm", int(m2.group(3))) if field.startswith("asm") else ("oq", int(m2.group(4)))
            lst = step.get(key) or []
            if idx >= len(lst):
                skipped.append((eid, f"{key} index {idx} out of range"))
                continue
            if removed:
                lst.pop(idx)
            else:
                lst[idx] = text
            step[key] = lst
            if not lst:
                step.pop(key, None)
        applied.append(eid)

    # rebuild the SNAP literal
    body = ",\n ".join(_dump(e) for e in snap)
    src = src[:s0] + "[\n " + body + ",\n]" + src[s1:]
    if intro_new is not None:
        src = re.sub(r"SNAP_INTRO = \(.*?\)\n", f"SNAP_INTRO = ({_py_str(intro_new)})\n", src, count=1, flags=re.S)
    if gaunt_new is not None:
        src = re.sub(r'GAUNTLET = """.*?"""', f"GAUNTLET = {_py_str(gaunt_new)}", src, count=1, flags=re.S)

    ast.parse(src)   # never write a file that will not import
    if dry:
        print("dry run: no files written")
    else:
        BUILDER.write_text(src)

    print(f"applied {len(applied)} narrative edit(s) to {BUILDER.name}")
    for eid, why in skipped:
        print(f"  SKIPPED {eid}: {why}")
    if added:
        print(f"\n{len(added)} added note(s) — no home in the structure yet, decide placement:")
        for a in added:
            print(f"  {a['id']}: {a.get('text', '')[:120]}")
    if prompt_edits:
        print(f"\n{len(prompt_edits)} PROMPT edit(s) — these change live pipeline behaviour:")
        for name, text in prompt_edits:
            tgt = PROMPT_FILES.get(name)
            print(f"\n  {name} -> {tgt[0].name + ':' + tgt[1] if tgt else 'UNMAPPED, apply by hand'}")
            if tgt and tgt[0].exists():
                cur = _read_const(tgt[0].read_text(), tgt[1])
                if cur is not None:
                    diff = list(difflib.unified_diff(cur.splitlines(), (text or "").splitlines(),
                                                     lineterm="", n=1))[2:]
                    print("\n".join("    " + d for d in diff[:40]) or "    (no change)")
                    if write_prompts and not dry and cur != text:
                        f = tgt[0]
                        f.write_text(_replace_const(f.read_text(), tgt[1], text))
                        print(f"    WRITTEN to {f.name}")
        if not write_prompts:
            print("\n  (prompts not written — rerun with --prompts to apply them)")


def _dump(obj, ind=1):
    """Deterministic pretty-printer for the SNAP literal (keeps it readable in git)."""
    pad = " " * ind
    if isinstance(obj, dict):
        items = ", ".join(f"{_py_str(k)}: {_dump(v, ind + 1)}" for k, v in obj.items())
        return "{" + items + "}"
    if isinstance(obj, list):
        return "[" + ", ".join(_dump(v, ind + 1) for v in obj) + "]"
    if obj is None:
        return "None"
    if isinstance(obj, str):
        return _py_str(obj)
    return repr(obj)


def _read_const(src: str, name: str):
    m = re.search(rf'^{name} = ("""|\'\'\')(.*?)\1', src, re.S | re.M)
    return m.group(2) if m else None


def _replace_const(src: str, name: str, value: str) -> str:
    return re.sub(rf'^{name} = ("""|\'\'\')(.*?)\1', lambda _: f'{name} = """{value}"""',
                  src, count=1, flags=re.S | re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edits", help="JSON exported from the review doc (file path, or - for stdin)")
    ap.add_argument("--prompts", action="store_true", help="also write prompt edits into the pipeline")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raw = input() if args.edits == "-" else Path(args.edits).read_text()
    blob = json.loads(raw)
    edits = blob.get("edits", blob if isinstance(blob, list) else [])
    comments = blob.get("comments") or []
    apply_edits(edits, args.prompts, args.dry_run)
    if comments:
        print(f"\n{len(comments)} comment(s) on posts (not applied, for reading):")
        for c in comments[:20]:
            print(f"  [{c.get('tab')}] @{c.get('handle')}: {c.get('comment', '')[:140]}")


if __name__ == "__main__":
    main()
