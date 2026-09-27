"""Article Claim Extractor (ACE).

Purpose-built for the misinformation-study source harvest (GitHub #13/#14) — NOT the
production pipeline extractor. Given one news article, extract the CENTRAL, headline-level,
CHECK-WORTHY factual claims the article asserts IN ITS OWN EDITORIAL VOICE — consequential,
empirically verifiable statements a reader would want checked. It does NOT select for how true,
false, surprising, or suspicious a claim looks (that conditioning biased the pool toward contested
framing and compressed the veracity range — GitHub #14 review); veracity is judged independently in
Stage 3. The extracted claims become items in a survey experiment, so we optimize for
standalone, self-contained claims. Returns an EMPTY list when the article has none (a first-class
outcome: those articles are flagged/logged). ACE does not judge truth — verification is Stage 3.

Model: DeepSeek-V4-Flash (config.EXTRACTION_MODEL, via DeepInfra).
"""
import json
import re

from openai import OpenAI

import config

MODEL = config.EXTRACTION_MODEL
_client = OpenAI(base_url=config.EXTRACTION_BASE_URL, api_key=config.EXTRACTION_API_KEY, max_retries=4)

SYSTEM = """You are an Article Claim Extractor (ACE) for a misinformation research study. The claims you \
extract become items in a survey experiment and then get an INDEPENDENT veracity check in \
Stage 3. Extract the article's CENTRAL, headline-level factual claims that are CHECK-WORTHY — consequential \
and empirically verifiable. Do NOT select for whether a claim looks true, false, surprising, or suspicious: \
extract check-worthy claims of every kind, mundane and sensational alike, and let Stage 3 judge truth.

Apply these tests to every candidate claim:

STEP 1 — THE OUTLET'S OWN ASSERTION. Extract only what the ARTICLE asserts as fact in its own editorial \
voice. If a statement is attributed ("[source] says / claims / alleges that X"), keep it ONLY when the \
article endorses X as established fact in its own voice (routine sourcing — "the Pentagon said it withdrew \
the troops" — counts as the outlet reporting a fact). If the article merely relays X, or reports it \
skeptically or as a contested allegation, do NOT flatten it into a bare assertion "X". Either drop it, or — \
when the speech-act itself is the central, checkable news — keep it WITH the attribution intact ("[source] \
said X"), so Stage 3 checks "did [source] say X", not "is X true". Extract the VERIFIABLE CORE ACTION, not \
an intent-laden characterization: from "deceived donors" extract the concrete act ("told donors the money \
funded America250 while it funded partisan programming").

STEP 2 — CHECK-WORTHY & CONSEQUENTIAL. Keep a claim only if it is EMPIRICALLY VERIFIABLE (a specific act, \
number, event, or causal link that evidence could confirm or refute) AND consequential (a reasonable person \
would want to know whether it is true). This is INDEPENDENT of how true or false it looks — an accurate, \
official, or mundane statement is exactly as extractable as a surprising or contested one. Drop only \
opinions, value judgments, recommendations, predictions/forecasts, subjective rankings, and trivial or \
purely procedural detail.

STEP 3 — CENTRAL. Keep only the article's main points — not minor, buried, or peripheral details.

STEP 4 — SELF-CONTAINED, WITHOUT INVENTING. Decontextualize so the claim verifies on its own — resolve \
pronouns; include who/what/when. But do NOT strengthen, sharpen, or assert beyond what the article states, \
and do NOT simply copy the headline when the body frames the claim more carefully. Stay faithful to what the \
article actually says.

Extract the few strongest qualifying claims — typically 0 to 3. You do NOT decide whether a claim is true or \
false. If the article states no central, checkable, consequential claim in its own voice, return an empty \
list (COMMON — opinion pieces, snark, routine reporting, own-poll write-ups).

Return ONLY JSON in exactly this shape:
{"claims": [{"claim": "<self-contained, checkable claim>", "basis": "<short phrase: what makes it \
central and empirically checkable>", "quote": "<short verbatim span from the article>"}]}
If none: {"claims": []}"""


def _parse(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"claims": []}
    try:
        data = json.loads(m.group(0))
        if isinstance(data.get("claims"), list):
            return data
    except Exception:
        pass
    return {"claims": []}


def extract(article_text: str, headline: str | None = None, max_chars: int = 8000,
            model: str = MODEL, temperature: float = 0.0) -> dict:
    """Return {"claims": [...], "n": int}. Empty claims list = no misinformation-risky claims."""
    body = (article_text or "").strip()[:max_chars]
    user = (f"HEADLINE: {headline}\n\n" if headline else "") + f"ARTICLE:\n{body}"
    resp = _client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=1400,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    )
    data = _parse(resp.choices[0].message.content)
    claims = data.get("claims") or []
    return {"claims": claims, "n": len(claims)}
