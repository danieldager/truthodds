"""Write pipeline.svg (light) and pipeline-dark.svg.   python docs/figures/src/make_pipeline.py"""
# Sizes are in viewBox units; the README shows the SVG at about 880px (73%), so titles
# render near 16px and sub-labels near 13px.
from pathlib import Path
OUT = Path(__file__).parent.parent
STAGES = [("Extract", ["claims + queries"]),
          ("Search", ["top 10 pages,", "own source excluded"]),
          ("Read", ["one of seven flags", "per page"]),
          ("Weigh", ["fitted log-likelihood", "weights"]),
          ("Decide", ["log-odds vs boundary"])]
THEMES = {  # tuned for GitHub's #ffffff and #0d1117 page grounds
    "pipeline.svg":      dict(fill="#eef1f5", stroke="#57606a", title="#1f2328", sub="#424a53", num="#57606a", arrow="#57606a"),
    "pipeline-dark.svg": dict(fill="#21262d", stroke="#8b949e", title="#f0f6fc", sub="#c9d1d9", num="#9198a1", arrow="#8b949e")}
W, H, BW, BH, GAP, X0, Y0 = 1200, 160, 200, 64, 45, 10, 10
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
LABEL = ("Pipeline: Extract claims and queries; Search the top 10 pages, own source excluded; "
         "Read each page and give it one of seven flags; Weigh the flags with fitted log-likelihood "
         "weights; Decide by comparing the summed log-odds with a boundary.")
for name, c in THEMES.items():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{LABEL}">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{c["arrow"]}"/></marker></defs>']
    for i, (t, subs) in enumerate(STAGES):
        x = X0 + i * (BW + GAP)
        o.append(f'<rect x="{x}" y="{Y0}" width="{BW}" height="{BH}" rx="10" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.5"/>')
        o.append(f'<text x="{x + BW / 2}" y="{Y0 + BH / 2 + 7}" text-anchor="middle" font-family="{FONT}" font-size="22" '
                 f'font-weight="600" fill="{c["title"]}"><tspan font-size="18" font-weight="500" fill="{c["num"]}">{i + 1}</tspan> {t}</text>')
        for j, s in enumerate(subs):
            o.append(f'<text x="{x + BW / 2}" y="{Y0 + BH + 32 + j * 23}" text-anchor="middle" font-family="{FONT}" '
                     f'font-size="18" fill="{c["sub"]}">{s}</text>')
        if i < len(STAGES) - 1:
            o.append(f'<line x1="{x + BW + 5}" y1="{Y0 + BH / 2}" x2="{x + BW + GAP - 5}" y2="{Y0 + BH / 2}" '
                     f'stroke="{c["arrow"]}" stroke-width="2" marker-end="url(#ah)"/>')
    o.append("</svg>")
    (OUT / name).write_text("\n".join(o) + "\n")
