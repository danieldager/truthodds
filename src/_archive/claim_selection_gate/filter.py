"""The production FILTER (Stage 1) — the system UNDER TEST, decoupled from the audit/gold.

The gold (`filter_dev.parquet`) is an INDEPENDENT ground truth: built by the AUDIT (235B +
`eval.audit.PASS_A_SYSTEM` + human review). The production filter is a SEPARATE, small/cheap model running
its OWN prompt (`FILTER_SYSTEM`, tuned for deployment) and is evaluated against that fixed gold
(`eval.scripts.filter_eval`). Keeping the two prompts separate is what makes the eval valid — tuning the
filter never moves the gold, and the gold can expose the filter's prompt-level biases.
"""
from __future__ import annotations

from eval.audit import _chat, _imgs   # shared transport/image helpers only — NOT the audit prompt

FILTER_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"

FILTER_SYSTEM = (
    "You are the FILTER stage of a misinformation-warning tool. From a social-media POST ALONE (its text and "
    "any image — the claim is often IN the image), decide whether to FLAG it for fact-checking. Never decide "
    "whether the claim is true.\n\n"
    "FLAG (true) iff the post makes a SPECIFIC, checkable factual claim whose VERACITY has POLITICAL-"
    "POLARIZATION implications, in a global sense (any country; any political division — left/right, pro/anti-"
    "government, immigration, religion or ethnicity as politics): if believed, it would sow division — make a "
    "political side or group look unreasonably bad, or a favoured side unreasonably good, or shift sentiment "
    "for or against a group or those in power. This is about IMPLICATIONS, not the surface topic.\n\n"
    "Do NOT flag (false):\n"
    "- no specific checkable claim (an opinion, rhetoric, a question, a bare fragment);\n"
    "- a claim whose veracity has NO political-polarization implication (a mundane fact, even about a "
    "politician; sports; a celebrity's private life; a consumer product; a neutral government-process fact);\n"
    "- an OBVIOUS joke or parody no reasonable reader would take as a real claim — BUT do flag DECEPTIVE "
    "satire that reads as a genuine claim.\n\n"
    "Output JSON only: {\"flag\": true|false, \"political_implication\": true|false, \"has_claim\": "
    "true|false, \"obvious_joke\": true|false, \"reason\": \"<short>\"}"
)


def pass_filter(row: dict, model: str) -> dict:
    """The filter's blind keep/reject judgment on a post (text + image). -> {flag, ...} or {_err}."""
    content = [{"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}}
               for b in _imgs(row.get("image_paths"))]
    content.append({"type": "text", "text": f"{FILTER_SYSTEM}\n\nPOST TEXT: {row.get('raw_context') or '(none)'}"})
    return _chat(model, content)
