#!/usr/bin/env python
"""Build the pipeline code walkthrough as ONE self-contained HTML page.

Reads manifest.json (ordered sections + code excerpts given as
repo-relative path + start/end line + an `expect` token that MUST appear inside
the range), pulls the EXACT lines from the working tree at build time, renders
them with real line numbers and Pygments syntax highlighting (inline CSS, so the
page needs no network), stamps the commit SHA and per-block GitHub permalinks,
and writes code_walkthrough.html beside this script.

FAILS LOUDLY when a line range is out of bounds or when an excerpt's `expect`
string is not found inside its range (so a stale line number is caught on every
rebuild). Run:

    uv run --with pygments python src/docs/code_walkthrough/build_walkthrough.py

No arguments, no network, no paid calls. Read-only except for the output HTML.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import JsonLexer, PythonLexer, TextLexer, get_lexer_for_filename
from pygments.util import ClassNotFound

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # code_walkthrough -> docs -> src -> repo root
MANIFEST = HERE / "manifest.json"
OUT = HERE / "code_walkthrough.html"
REMOTE_BLOB = "https://github.com/OWNER/REPO/blob"

SECRET_PATTERNS = [re.compile(p) for p in (r"sk-[A-Za-z0-9]", r"api_key\s*=\s*['\"]",
                                           r"Bearer\s+[A-Za-z0-9]")]


class BuildError(RuntimeError):
    pass


def sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()


def short_sha(s: str) -> str:
    return s[:12]


def lexer_for(path: str):
    if path.endswith(".py"):
        return PythonLexer()
    if path.endswith(".json"):
        return JsonLexer()
    try:
        return get_lexer_for_filename(path)
    except ClassNotFound:
        return TextLexer()


def read_range(abs_path: Path, start: int, end: int, expect: str, ref: str) -> str:
    if not abs_path.exists():
        raise BuildError(f"[{ref}] file does not exist: {abs_path}")
    lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    if not (1 <= start <= end <= n):
        raise BuildError(f"[{ref}] range {start}-{end} out of bounds (file has {n} lines)")
    chunk = "\n".join(lines[start - 1:end])
    if expect and expect not in chunk:
        raise BuildError(
            f"[{ref}] expect string not found in {start}-{end}: {expect!r}\n"
            f"  (line numbers are probably stale; re-locate the excerpt)")
    return chunk


def render_code(code: str, path: str, start: int, cw_id: str) -> str:
    fmt = HtmlFormatter(cssclass=f"cw {cw_id}", linenos="table", linenostart=start,
                        nowrap=False, wrapcode=True)
    return highlight(code, lexer_for(path), fmt)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def permalink(path: str, start: int, end: int, full_sha: str) -> str:
    return f"{REMOTE_BLOB}/{full_sha}/{path}#L{start}-L{end}"


def render_excerpt(ex: dict, full_sha: str, ref: str) -> str:
    path = ex["path"]
    start, end = int(ex["start"]), int(ex["end"])
    external = bool(ex.get("external"))
    abs_path = Path(path) if external else (REPO / path)
    code = read_range(abs_path, start, end, ex.get("expect", ""), ref)
    code_html = render_code(code, path, start, "cwblk")
    caption = ex.get("caption", "")
    collapsed = bool(ex.get("collapsed"))
    display_path = path if not external else f"{path}  (shared tools repo — external)"
    if external:
        link_html = '<span class="ext">not in this repo</span>'
    else:
        link = permalink(path, start, end, full_sha)
        link_html = f'<a class="perma" href="{esc(link)}" target="_blank" rel="noopener">GitHub&#8599;</a>'
    open_attr = "" if collapsed else " open"
    cap_html = f'<p class="cap">{esc(caption)}</p>' if caption else ""
    return (
        f'<details class="excerpt"{open_attr}>'
        f'<summary><code class="loc">{esc(display_path)}:{start}-{end}</code>{link_html}'
        f'<span class="chev" aria-hidden="true"></span></summary>'
        f'{cap_html}'
        f'<div class="codewrap">{code_html}</div>'
        f'</details>'
    )


def render_prose(text: str) -> str:
    if not text:
        return ""
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        if lines and all(ln.lstrip().startswith("- ") for ln in lines):
            lis = "".join(f"<li>{esc(ln.lstrip()[2:]).strip()}</li>" for ln in lines)
            out.append(f"<ul class=\"prose-list\">{lis}</ul>")
        else:
            out.append(f"<p>{esc(b).strip()}</p>")
    return "\n".join(out)


def render_glossary(items: list) -> str:
    rows = "".join(
        f'<div class="gloss-row"><dt>{esc(i["term"])}</dt><dd>{esc(i["def"])}</dd></div>'
        for i in items)
    return f'<dl class="glossary">{rows}</dl>'


def render_commands(cmds: list) -> str:
    body = "\n".join(esc(c) for c in cmds)
    return f'<pre class="cmd"><code>{body}</code></pre>'


def render_dirmap(rows: list) -> str:
    trs = "".join(
        f'<tr><td><code>{esc(r.get("path"))}</code></td><td>{esc(r.get("what"))}</td>'
        f'<td>{esc(r.get("writer"))}</td><td>{esc(r.get("reader"))}</td></tr>'
        for r in rows)
    return ('<div class="tablewrap"><table class="dirmap">'
            '<thead><tr><th>path</th><th>what</th><th>written by</th><th>read by</th></tr></thead>'
            f'<tbody>{trs}</tbody></table></div>')


def render_schemas(schemas: list) -> str:
    out = []
    for s in schemas:
        cols = "".join(
            f'<tr><td><code>{esc(c.get("col"))}</code></td><td class="dt">{esc(c.get("dtype"))}</td>'
            f'<td>{esc(c.get("meaning"))}</td></tr>'
            for c in s.get("columns", []))
        head = (f'<p class="schema-head"><code>{esc(s.get("path") or s.get("name"))}</code>'
                f'<span class="shape">{esc(s.get("shape"))}</span></p>')
        table = ('<div class="tablewrap"><table class="schema">'
                 '<thead><tr><th>column</th><th>dtype</th><th>meaning</th></tr></thead>'
                 f'<tbody>{cols}</tbody></table></div>') if cols else ""
        out.append(f'<div class="schema-block">{head}{table}</div>')
    return "\n".join(out)


def render_also(items: list) -> str:
    lis = "".join(f"<li>{esc(i)}</li>" for i in items)
    return ('<details class="also"><summary>Also relevant, not embedded</summary>'
            f'<ul>{lis}</ul></details>')


def render_subsection(sub: dict, full_sha: str, ref: str) -> str:
    heading = esc(sub.get("heading", ""))
    hid = slugify(sub.get("heading", ""))
    parts = [f'<h3 id="{hid}">{heading}</h3>']
    if sub.get("prose"):
        parts.append(render_prose(sub["prose"]))
    for i, ex in enumerate(sub.get("excerpts", [])):
        parts.append(render_excerpt(ex, full_sha, f"{ref}/{hid}#{i}"))
    if sub.get("stored_at"):
        parts.append(f'<p class="stored">Result stored at <code>{esc(sub["stored_at"])}</code></p>')
    if sub.get("also_relevant"):
        parts.append(render_also(sub["also_relevant"]))
    return '<div class="subsection">' + "\n".join(parts) + "</div>"


def render_section(sec: dict, full_sha: str) -> str:
    sid = sec["id"]
    title = esc(sec["title"])
    parts = [f'<section id="{esc(sid)}"><h2>{title}</h2>']
    if sec.get("prose"):
        parts.append(render_prose(sec["prose"]))
    if sec.get("diagram"):
        parts.append(f'<div class="diagram">{sec["diagram"]}</div>')   # trusted raw HTML from manifest
    if sec.get("glossary"):
        parts.append(render_glossary(sec["glossary"]))
    if sec.get("commands"):
        parts.append(render_commands(sec["commands"]))
    if sec.get("dirmap"):
        parts.append(render_dirmap(sec["dirmap"]))
    if sec.get("schemas"):
        parts.append(render_schemas(sec["schemas"]))
    for i, ex in enumerate(sec.get("excerpts", [])):
        parts.append(render_excerpt(ex, full_sha, f"{sid}#{i}"))
    for sub in sec.get("subsections", []):
        parts.append(render_subsection(sub, full_sha, sid))
    if sec.get("also_relevant"):
        parts.append(render_also(sec["also_relevant"]))
    parts.append("</section>")
    return "\n".join(parts)


def render_toc(sections: list) -> str:
    items = []
    for sec in sections:
        subs = "".join(
            f'<a class="toc-sub" href="#{slugify(sub.get("heading",""))}">{esc(sub.get("heading",""))}</a>'
            for sub in sec.get("subsections", []))
        items.append(
            f'<a class="toc-item" href="#{esc(sec["id"])}">{esc(sec["title"])}</a>'
            + (f'<div class="toc-subs">{subs}</div>' if subs else ""))
    return "\n".join(items)


def pygments_css() -> str:
    """Light palette on bare :root; dark palette re-scoped so it wins under
    prefers-color-scheme:dark (system) and [data-theme=dark] (explicit)."""
    light = HtmlFormatter(style="default").get_style_defs(".cw")
    dark_raw = HtmlFormatter(style="github-dark").get_style_defs(".cw")

    def rescope(defs: str, prefix: str) -> str:
        out = []
        for ln in defs.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            out.append(f"{prefix} {ln}")
        return "\n".join(out)

    media = ("@media (prefers-color-scheme: dark){\n"
             + rescope(dark_raw, ':root:not([data-theme="light"])') + "\n}")
    forced = rescope(dark_raw, ':root[data-theme="dark"]')
    return f"{light}\n{media}\n{forced}"


def build() -> None:
    if not MANIFEST.exists():
        raise BuildError(f"manifest not found: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    full_sha = sha()
    sections = manifest["sections"]

    body = "\n".join(render_section(s, full_sha) for s in sections)
    toc = render_toc(sections)
    hl_css = pygments_css()

    page = TEMPLATE.format(
        title=esc(manifest.get("title", "Pipeline Code Walkthrough")),
        subtitle=esc(manifest.get("subtitle", "")),
        short_sha=short_sha(full_sha),
        full_sha=full_sha,
        repo_blob=f"{REMOTE_BLOB}/{full_sha}",
        toc=toc,
        body=body,
        hl_css=hl_css,
    )

    # Secret scan on the RENDERED page before writing (defence in depth).
    for pat in SECRET_PATTERNS:
        m = pat.search(page)
        if m:
            raise BuildError(f"SECRET-LIKE STRING in output near: {page[max(0,m.start()-40):m.start()+40]!r}")

    OUT.write_text(page, encoding="utf-8")
    n_ex = sum(len(s.get("excerpts", [])) + sum(len(sub.get("excerpts", []))
               for sub in s.get("subsections", [])) for s in sections)
    print(f"OK: {len(sections)} sections, {n_ex} excerpts -> {OUT}  ({len(page)//1024} KB, sha {short_sha(full_sha)})")


TEMPLATE = """<title>{title}</title>
<style>
:root {{
  --bg: #f7f7f5; --surface: #ffffff; --surface-2: #f0f0ee; --border: #e0e0dc;
  --ink: #1c1c1a; --ink-soft: #55554f; --ink-faint: #8a8a82;
  --accent: #3257c4; --accent-soft: #e8ecf9;
  --code-bg: #fbfbfa; --code-border: #e6e6e1;
  --mono: "SFMono-Regular", ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, system-ui, "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
  --sidebar: 280px; --content: 1080px;
}}
:root:not([data-theme="light"]) {{}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #16161a; --surface: #1d1d22; --surface-2: #24242b; --border: #34343c;
    --ink: #e9e9ec; --ink-soft: #b3b3bb; --ink-faint: #7d7d87;
    --accent: #8ea6ff; --accent-soft: #23273a;
    --code-bg: #1a1a1f; --code-border: #2c2c34;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #16161a; --surface: #1d1d22; --surface-2: #24242b; --border: #34343c;
  --ink: #e9e9ec; --ink-soft: #b3b3bb; --ink-faint: #7d7d87;
  --accent: #8ea6ff; --accent-soft: #23273a;
  --code-bg: #1a1a1f; --code-border: #2c2c34;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
  font-size: 15px; line-height: 1.6; -webkit-font-smoothing: antialiased; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ font-family: var(--mono); font-size: 0.86em; }}

.layout {{ display: flex; align-items: flex-start; gap: 0; }}
.sidebar {{ position: sticky; top: env(safe-area-inset-top, 0px); align-self: flex-start;
  width: var(--sidebar); height: 100vh; overflow-y: auto; padding: 22px 14px 40px 22px;
  border-right: 1px solid var(--border); background: var(--surface); flex: 0 0 auto; }}
.sidebar h1 {{ font-size: 15px; margin: 0 0 4px; letter-spacing: -0.01em; }}
.sidebar .sha {{ font-family: var(--mono); font-size: 11px; color: var(--ink-faint);
  display: block; margin-bottom: 18px; word-break: break-all; }}
.toc-item {{ display: block; color: var(--ink-soft); padding: 5px 8px; border-radius: 6px;
  font-size: 13px; line-height: 1.35; }}
.toc-item:hover {{ background: var(--surface-2); text-decoration: none; }}
.toc-subs {{ margin: 0 0 4px 12px; border-left: 1px solid var(--border); }}
.toc-sub {{ display: block; color: var(--ink-faint); padding: 3px 8px; font-size: 12px; }}
.toc-sub:hover {{ color: var(--accent); text-decoration: none; }}

.main {{ flex: 1 1 auto; min-width: 0; }}
.wrap {{ max-width: var(--content); margin: 0 auto; padding-block: 40px 96px;
  padding-left: 32px; padding-right: 32px; }}
.pagehead {{ margin-bottom: 40px; }}
.pagehead h1 {{ font-size: 30px; line-height: 1.15; margin: 0 0 10px; letter-spacing: -0.02em;
  text-wrap: balance; }}
.pagehead p {{ color: var(--ink-soft); margin: 0; max-width: 68ch; }}

.themebtn {{ position: fixed; top: 14px; right: 16px; z-index: 20; border: 1px solid var(--border);
  background: var(--surface); color: var(--ink-soft); border-radius: 999px; padding: 6px 12px;
  font-size: 12px; font-family: var(--sans); cursor: pointer; }}

section {{ margin-bottom: 52px; scroll-margin-top: 20px; }}
section h2 {{ font-size: 22px; letter-spacing: -0.015em; margin: 0 0 12px; padding-bottom: 8px;
  border-bottom: 2px solid var(--border); text-wrap: balance; }}
section p {{ max-width: 72ch; }}
.subsection {{ margin: 26px 0; }}
.subsection h3 {{ font-size: 16px; margin: 0 0 8px; color: var(--ink); scroll-margin-top: 20px; }}
.stored {{ font-size: 13px; color: var(--ink-soft); }}
.prose-list {{ max-width: 76ch; padding-left: 20px; }}
.prose-list li {{ margin: 6px 0; color: var(--ink-soft); }}

.glossary {{ display: grid; grid-template-columns: 1fr; gap: 0; margin: 16px 0;
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface); }}
.gloss-row {{ display: grid; grid-template-columns: 160px 1fr; gap: 14px; padding: 10px 14px;
  border-bottom: 1px solid var(--border); }}
.gloss-row:last-child {{ border-bottom: none; }}
.glossary dt {{ font-family: var(--mono); font-size: 12.5px; color: var(--accent); margin: 0; }}
.glossary dd {{ margin: 0; font-size: 13.5px; color: var(--ink-soft); }}

.diagram {{ margin: 20px 0; padding: 20px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; overflow-x: auto; }}

.cmd {{ background: var(--code-bg); border: 1px solid var(--code-border); border-radius: 10px;
  padding: 12px 14px; overflow-x: auto; font-family: var(--mono); font-size: 12.5px;
  color: var(--ink); line-height: 1.5; }}

.excerpt {{ margin: 14px 0; border: 1px solid var(--border); border-radius: 10px;
  background: var(--surface); overflow: hidden; }}
.excerpt > summary {{ display: flex; align-items: center; gap: 12px; cursor: pointer;
  padding: 9px 14px; background: var(--surface-2); list-style: none; font-size: 12.5px; }}
.excerpt > summary::-webkit-details-marker {{ display: none; }}
.excerpt .loc {{ color: var(--ink); font-weight: 600; word-break: break-all; }}
.excerpt .perma {{ font-size: 11.5px; white-space: nowrap; }}
.excerpt .ext {{ font-size: 11px; color: var(--ink-faint); white-space: nowrap; }}
.excerpt .chev {{ margin-left: auto; width: 8px; height: 8px; border-right: 2px solid var(--ink-faint);
  border-bottom: 2px solid var(--ink-faint); transform: rotate(45deg); transition: transform .15s; flex: 0 0 auto; }}
.excerpt[open] .chev {{ transform: rotate(-135deg); }}
.excerpt .cap {{ margin: 10px 14px 2px; font-size: 13px; color: var(--ink-soft); max-width: 76ch; }}
.codewrap {{ overflow-x: auto; padding: 4px 4px 8px; }}

.also {{ margin: 12px 0; font-size: 13px; }}
.also > summary {{ cursor: pointer; color: var(--ink-soft); padding: 6px 0; }}
.also ul {{ margin: 4px 0 4px 4px; padding-left: 18px; color: var(--ink-soft); }}
.also li {{ margin: 3px 0; font-family: var(--mono); font-size: 12px; }}

.tablewrap {{ overflow-x: auto; margin: 14px 0; border: 1px solid var(--border); border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: var(--surface); }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
th {{ background: var(--surface-2); font-weight: 600; font-size: 12px; letter-spacing: 0.02em;
  text-transform: uppercase; color: var(--ink-soft); }}
tr:last-child td {{ border-bottom: none; }}
td.dt {{ font-family: var(--mono); font-size: 12px; color: var(--ink-faint); }}
.schema-head {{ display: flex; gap: 12px; align-items: baseline; margin: 20px 0 6px; }}
.schema-head .shape {{ font-size: 12px; color: var(--ink-faint); font-family: var(--mono); }}

/* Pygments: line-number gutter + wrapping */
.cw {{ background: var(--code-bg); border: 1px solid var(--code-border); border-radius: 8px; }}
.cw pre {{ margin: 0; padding: 10px 0; font-family: var(--mono); font-size: 12.5px; line-height: 1.5;
  background: transparent; }}
.cw table.highlighttable, .cw table.cwtable {{ border: none; background: transparent; width: auto; margin: 0; }}
.cw table.highlighttable td, .cw table.cwtable td {{ border: none; padding: 0; }}
.cw td.linenos {{ -webkit-user-select: none; user-select: none; text-align: right;
  padding: 0 12px 0 12px; color: var(--ink-faint); background: var(--surface-2);
  border-right: 1px solid var(--code-border); }}
.cw td.linenos pre {{ color: var(--ink-faint); }}
.cw td.code {{ padding-left: 14px !important; width: 100%; }}
.cw .code pre {{ white-space: pre; }}

@media (max-width: 820px) {{
  .sidebar {{ display: none; }}
  .wrap {{ padding-left: 16px; padding-right: 16px; }}
}}
/* --- Pygments syntax colors (light on :root, dark re-scoped) --- */
{hl_css}
</style>

<button class="themebtn" id="themebtn" type="button">theme</button>
<div class="layout">
  <nav class="sidebar">
    <h1>{title}</h1>
    <span class="sha">frozen @ {short_sha}</span>
    {toc}
  </nav>
  <div class="main">
    <div class="wrap">
      <header class="pagehead">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <p style="font-size:12.5px;color:var(--ink-faint);margin-top:8px;">
          Every code block below is pulled verbatim from the working tree at commit
          <a href="{repo_blob}" target="_blank" rel="noopener"><code>{short_sha}</code></a>
          by the build script; line numbers are real. Blocks link to the pinned GitHub permalink.
        </p>
      </header>
      {body}
    </div>
  </div>
</div>
<script>
(function () {{
  var btn = document.getElementById("themebtn");
  var root = document.documentElement;
  function cur() {{
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }}
  try {{ var saved = localStorage.getItem("cw-theme"); if (saved) root.setAttribute("data-theme", saved); }} catch (e) {{}}
  btn.addEventListener("click", function () {{
    var next = cur() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {{ localStorage.setItem("cw-theme", next); }} catch (e) {{}}
  }});
}})();
</script>
"""


if __name__ == "__main__":
    try:
        build()
    except BuildError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        sys.exit(1)
