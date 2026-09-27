"""Flag and resolve fc-gold claims whose text cannot stand alone (unresolved deixis).

A claim like "My company was building the Keystone Pipeline" is ill-posed for READ and
for retrieval until the speaker is resolved (2026-08-03, Keystone calibration case).
Resolution is tiered so the leak-free path does the bulk of the work:

  ok             no first-person deixis — claim stands alone, resolved = raw
  quoted-speech  first person confined to quoted spans with an attribution verb in
                 the narration ("Jane Fonda said 'I'm moving…'") — self-contained
  self-attributed  deixis present but the claimant is already named in the claim text
  resolved-t1    deixis + a usable `claimant` in GFC metadata → prefix, code-only,
                 zero leak: {claimant} said: "{raw}"
  needs-article  deixis + claimant missing/generic ("Viral image") → tier-2 queue for
                 the firewalled review_url extraction (NOT run here — LLM step, gated)

Detection is deliberately narrow: first-person pronouns only. Demonstrative deixis on
post media ("this video shows…") is the media-authenticity axis, carved out elsewhere.

  uv run python -m eval.scripts.build_eval.resolve_claims

Writes eval/data/claim_resolution.parquet (one row per v3 row, keyed by review_url +
claim_text) and the audit page eval/data/claim_resolution_audit.html (Daniel reads:
tier-1 diffs first, then the tier-2 queue).
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SRC = Path("eval/data/fc_gold_v3.parquet")
OUT = Path("eval/data/claim_resolution.parquet")
PAGE = Path("eval/data/claim_resolution_audit.html")

# First-person only. [Uu]s does not match "US" (the country); bare "I" is
# case-sensitive by construction.
# Lowercase-only for us (capitalized = brands: Toys "R" Us); the lookbehind keeps
# ".us" domains out. "mine" dropped entirely — the noun (coal mine, FR mine)
# swamps the possessive in claim text. "I" guarded against Roman numerals
# (phase I study) and "My" against titles/brands (My Secret Terrius, My Pillow).
FIRST_PERSON = re.compile(
    r"(?<![Pp]hase )(?<![Tt]ype )(?<![Cc]lass )(?<![Ww]ar )\bI\b"
    r"|\b[Mm]y\b(?! [A-Z])|\b([Ww]e|[Oo]ur|[Oo]urs)\b|(?<![.\w])us\b")

# Claimants that name a phenomenon, not a speaker — prefixing these resolves nothing.
GENERIC = re.compile(
    r"viral|social media|facebook|instagram|whatsapp|tiktok|twitter|telegram"
    r"|multiple|multiples|various|posts?\b|websites?|blogs?|bloggers?|rumou?r"
    r"|internet|online|sources|users|netizens|forwarded|chain", re.I)


QUOTED = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|\'[^\']{6,}\'')
ATTRIB = re.compile(
    r"\b(said|says|saying|tweeted|posted|wrote|writes|claimed|claims|stated"
    r"|told|announced|asked|declared|joked|quipped|improvised|sang|remarked"
    r"|admitted|responded|replied|argued|boasted|promised|warned|read|reads"
    r"|reading)\b", re.I)


# Image/video-reference claims. Two flavours (Daniel 2026-08-03):
#   media-locus — the proposition IS about what the media shows ("a photo shows…",
#   "as seen in a video…"): text-only verification cannot judge it → excluded,
#   UNLESS the context resolves the referents into a world-proposition we can
#   check without the pixels.
#   media-incidental — media is just the venue ("said in a video interview
#   that…"): the assertion is textual → keep.
MEDIA_LOCUS = re.compile(
    r"\b(photo(?:graph)?s?|images?|videos?|screenshots?|footage|clip)\b"
    r".{0,40}\b(shows?|showing|authentic\w*|captures?|depicts?|of\b)"
    r"|\bas seen in\b|\bpictured\b|\bis seen\b|\bcan be seen\b", re.I)


def referent_resolved(claim: str, context: str) -> bool:
    """Does the extracted context NAME the demonstrative's referent?

    "This mosque was destroyed…" + context naming "Tinmel Mosque" → resolved,
    usable. Context that only restates the setting ("a mosque in Morocco…")
    resolves nothing → the claim is skipped (Daniel 2026-08-03: unless the
    article yields the name, these claims don't enter an eval).
    Heuristic: the context contributes at least one capitalized token that is
    neither in the claim nor a sentence starter.
    """
    if not context:
        return False
    claim_toks = {t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ']+", claim)}
    fresh = []
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-ZÀ-Þ][a-zà-ÿ'-]{2,})", context):
        if m.group(1).lower() not in claim_toks:
            fresh.append(m.group(1))
    return bool(fresh)


def claimant_in_text(claimant: str, claim: str) -> bool:
    toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ]{3,}", claimant.lower())]
    low = claim.lower()
    return bool(toks) and all(t in low for t in toks)


# Demonstrative deixis: "This mosque was destroyed…" — the referent lives in the
# post (usually its media), not the claim text. Temporal demonstratives ("this
# year") are excluded: the claim date resolves those. These claims are ill-posed
# until tier-2 article extraction names the referent (claim_context does exactly
# that), so they queue rather than pass.
DEMONSTRATIVE = re.compile(
    r"^(?:[\"“']\s*)?(?:This|These|That|Those)\s+"
    r"(?!year|month|week|time|day|morning|evening|weekend|summer|winter|spring|fall"
    r"|autumn|at\b|who\b|whom\b|which\b|with\b|in\b|of\b|on\b|is\b|was\b|are\b|were\b)"
    r"[a-z0-9]", re.M)

# "These 13 books…" — a demonstrative over an ENUMERATED set. A short context can
# name one mosque; it cannot enumerate 13 books, so no contextualization makes the
# exact claim assertable. Excluded from evals outright (Daniel 2026-08-03).
DEMONSTRATIVE_LIST = re.compile(
    r"^(?:[\"“']\s*)?(?:These|Those)\s+\d+\s+\w+|\b[Tt]hese\s+\d+\s+\w+", re.M)


def classify(claim: str, claimant) -> tuple[str, str]:
    if DEMONSTRATIVE_LIST.search(claim or ""):
        return "unresolvable-list", claim
    if DEMONSTRATIVE.search(claim or ""):
        return "needs-article", claim
    if not FIRST_PERSON.search(claim or ""):
        return "ok", claim
    # First person confined to quoted spans is self-contained when the narration
    # either attributes the quote ("Jane Fonda said 'I'm moving…'") or describes it
    # as displayed content ("A photo shows a man in an 'I don't care…' shirt").
    # A claim that is one bare quote (empty narration) still needs a speaker.
    narration = QUOTED.sub(" ", claim)
    if not FIRST_PERSON.search(narration) and (
            ATTRIB.search(narration) or len(narration.strip()) >= 20):
        return "quoted-speech", claim
    c = (claimant or "").strip() if isinstance(claimant, str) else ""
    if c and claimant_in_text(c, claim):
        return "self-attributed", claim
    if not c or GENERIC.search(c):
        return "needs-article", claim
    return "resolved-t1", f'{c} said: "{claim}"'


def build_page(df: pd.DataFrame) -> str:
    def esc(s):
        return html.escape(str(s or ""))

    t1 = df[df.resolution_status == "resolved-t1"]
    t2 = df[df.resolution_status == "needs-article"]
    sa = df[df.resolution_status == "self-attributed"]
    counts = df.resolution_status.value_counts().to_dict()

    rows_t1 = "\n".join(
        f'<div class="case"><div class="meta">{esc(r.publisher_site)} · veracity '
        f'{r.veracity} · <a href="{esc(r.review_url)}">review</a></div>'
        f'<div class="raw">− {esc(r.claim_text)}</div>'
        f'<div class="res">+ {esc(r.claim_resolved)}</div></div>'
        for r in t1.itertuples())
    rows_t2 = "\n".join(
        f'<div class="case t2"><div class="meta">{esc(r.publisher_site)} · claimant: '
        f'<b>{esc(r.claimant) or "∅"}</b> · veracity {r.veracity} · '
        f'<a href="{esc(r.review_url)}">review</a></div>'
        f'<div class="raw">{esc(r.claim_text)}</div></div>'
        for r in t2.itertuples())
    rows_sa = "\n".join(
        f'<div class="case sa"><div class="meta">{esc(r.publisher_site)} · claimant '
        f'already in text: <b>{esc(r.claimant)}</b></div>'
        f'<div class="raw">{esc(r.claim_text)}</div></div>'
        for r in sa.head(60).itertuples())

    return f"""<!doctype html><meta charset="utf-8">
<title>Claim resolution audit</title>
<style>
body{{font:14px/1.45 -apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#111}}
h2{{border-bottom:2px solid #111;padding-bottom:.2rem;margin-top:2.2rem}}
.case{{border-left:3px solid #888;padding:.4rem .8rem;margin:.7rem 0}}
.case.t2{{border-left-color:#b00}} .case.sa{{border-left-color:#bbb;color:#555}}
.meta{{font-size:12px;color:#666;margin-bottom:.25rem}}
.raw{{color:#802}} .res{{color:#062}}
code{{background:#eee;padding:0 .3em}}
</style>
<h1>Claim resolution audit — fc_gold_v3</h1>
<p>Detection: first-person deixis only. Counts: <code>{counts}</code> of {len(df)} rows.</p>
<h2>Tier 2 queue — needs article extraction ({len(t2)}) — read these first</h2>
<p>Claimant missing or generic; the firewalled review_url extraction (gated, unbuilt)
would resolve these. Anything unresolvable is excluded from CAL/VAL.</p>
{rows_t2}
<h2>Tier 1 — resolved from claimant metadata ({len(t1)})</h2>
<p>Code-only prefix, zero leak. Red = raw, green = resolved.</p>
{rows_t1}
<h2>Self-attributed sample ({len(sa)} total, first 60)</h2>
<p>Deixis present but the claimant is already named inside the claim text — left as-is.</p>
{rows_sa}
"""


def main() -> None:
    df = pd.read_parquet(SRC, columns=[
        "review_url", "claim_text", "claimant", "publisher_site", "veracity"])
    res = [classify(r.claim_text, r.claimant) for r in df.itertuples()]
    df["resolution_status"] = [s for s, _ in res]
    df["claim_resolved"] = [t for _, t in res]
    df.to_parquet(OUT)
    PAGE.write_text(build_page(df))
    print(df.resolution_status.value_counts().to_string())
    print(f"\nwrote {OUT} and {PAGE}")


if __name__ == "__main__":
    main()
