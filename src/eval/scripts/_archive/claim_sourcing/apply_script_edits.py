"""Apply an exported edit set from the pipeline-script doc back into build_script_doc.py.

Round trip: edit in the browser -> export -> run this -> rebuild. Edit ids:
  script::intro / method::intro     the two tab intros
  script::s<n>                      a procedure step body
  ex::<pid>::<i>                     a worked-example trace line
  method::m<n>                      a methods-draft paragraph

  cd src && uv run python eval/scripts/claim_sourcing/apply_script_edits.py edits.json [--dry-run]
"""
import argparse
import ast
import json
import re
from pathlib import Path

BUILDER = Path(__file__).parent / "build_script_doc.py"


def _lit(name, src):
    """Byte range + parsed value of a module-level literal `NAME = ...`."""
    m = re.search(rf"^{name} = ", src, re.M)
    start = m.end()
    depth, i, opener = 0, start, src[m.end()]
    close = {"[": "]", "(": ")", "{": "}"}[opener]
    while i < len(src):
        if src[i] == opener:
            depth += 1
        elif src[i] == close:
            depth -= 1
            if depth == 0:
                break
        i += 1
    return start, i + 1, ast.literal_eval(src[start:i + 1])


def _pystr(v):
    if "\n" in v:
        return '"""' + v.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '"""'
    return json.dumps(v, ensure_ascii=False)


def _dump_list(rows, tuples):
    """Re-emit SCRIPT/EXAMPLES/METHODS as a readable literal. tuples: len of each tuple, or 0 for METHODS."""
    out = ["["]
    for r in rows:
        if isinstance(r, tuple):
            out.append("    (" + ", ".join(_dump_val(x) for x in r) + "),")
        else:
            out.append("    " + _dump_val(r) + ",")
    out.append("]")
    return "\n".join(out)


def _dump_val(x):
    if isinstance(x, str):
        return _pystr(x)
    if isinstance(x, (list, tuple)):
        br = "[]" if isinstance(x, list) else "()"
        inner = ", ".join(_dump_val(y) for y in x)
        if isinstance(x, tuple) and len(x) == 1:
            inner += ","
        return br[0] + inner + br[1]
    return repr(x)


def _lit_str(name, src):
    """Byte range (of the whole assignment) + value of a `NAME = ...` string literal.
    Handles both the paren implicit-concat form and a triple-quoted string (whichever a
    prior apply last wrote)."""
    m = re.search(rf"^{name} = ", src, re.M)
    start = m.end()
    if src[start] == "(":
        depth, i = 0, start
        while i < len(src):
            if src[i] == "(": depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0: break
            i += 1
        end = i + 1
    elif src[start:start + 3] == '"""':
        end = src.index('"""', start + 3) + 3
    else:                                   # single-line quoted string
        end = src.index("\n", start)
    return m.start(), end, ast.literal_eval(src[start:end])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edits")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    blob = json.loads(Path(args.edits).read_text() if args.edits != "-" else __import__("sys").stdin.read())
    edits = blob.get("edits", blob if isinstance(blob, list) else [])

    src = BUILDER.read_text()
    s0, s1, SCRIPT = _lit("SCRIPT", src)
    e0, e1, EXAMPLES = _lit("EXAMPLES", src)
    mb0, mb1, _mbody = _lit_str("METHODS_BODY", src)
    METHODS_BODY = [_mbody]
    intro_script = intro_methods = None
    applied, skipped = [], []

    SCRIPT = [list(r) for r in SCRIPT]
    EXAMPLES = [list(r) for r in EXAMPLES]

    for ed in edits:
        eid, text, removed = ed.get("id", ""), ed.get("text"), ed.get("removed")
        if eid == "script::intro":
            intro_script = text; applied.append(eid); continue
        if eid == "method::intro":
            intro_methods = text; applied.append(eid); continue
        m = re.match(r"script::(s\d+)$", eid)
        if m:
            row = next((r for r in SCRIPT if r[0] == m.group(1)), None)
            if row: row[2] = text; applied.append(eid)
            else: skipped.append((eid, "no step"))
            continue
        if eid == "method::body":
            METHODS_BODY[0] = text; applied.append(eid); continue
        m = re.match(r"ex::(\d+)::(\d+)$", eid)
        if m:
            pid, idx = m.group(1), int(m.group(2))
            row = next((r for r in EXAMPLES if r[0] == pid), None)
            if row and idx < len(row[4]):
                if removed: row[4].pop(idx)
                else: row[4][idx] = text
                applied.append(eid)
            else: skipped.append((eid, "no example line"))
            continue
        skipped.append((eid, "unrecognised id"))

    SCRIPT = [tuple(r) for r in SCRIPT]
    EXAMPLES = [tuple(r[:4]) + (r[4],) for r in EXAMPLES]

    new = src
    # splice literals back, highest offset first so earlier offsets stay valid
    splices = [(s0, s1, _dump_list(SCRIPT, 0)), (e0, e1, _dump_list(EXAMPLES, 0)),
               (mb0, mb1, "METHODS_BODY = " + _pystr(METHODS_BODY[0]))]
    for a, b, txt in sorted(splices, key=lambda t: -t[0]):
        new = new[:a] + txt + new[b:]
    if intro_script is not None:
        new = re.sub(r"INTRO_SCRIPT = \(.*?\)\n", f"INTRO_SCRIPT = ({_pystr(intro_script)})\n", new, 1, re.S)
    if intro_methods is not None:
        new = re.sub(r"INTRO_METHODS = \(.*?\)\n", f"INTRO_METHODS = ({_pystr(intro_methods)})\n", new, 1, re.S)

    ast.parse(new)
    if args.dry_run:
        print("dry run, no write")
    else:
        BUILDER.write_text(new)
    print(f"applied {len(applied)} edit(s) to {BUILDER.name}")
    for eid, why in skipped:
        print(f"  SKIPPED {eid}: {why}")


if __name__ == "__main__":
    main()
