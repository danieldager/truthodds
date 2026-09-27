"""Prompt templates for CheckThat! 2025 Task 2 (Claim Normalization) eval.

Five LLM systems + a no-op baseline. Each builder returns OpenAI-style messages.
"""
from __future__ import annotations

import json
import re
from typing import Callable

Messages = list[dict[str, str]]


# --- 1. Baseline: post-as-is ----------------------------------------------------

def baseline(post: str) -> str:
    """Identity transform — returns the post unchanged. No LLM call."""
    return post.strip()


# --- 2. Our prompt (single-claim adaptation of pipeline/claim_extraction.py) -----

_OURS_SYSTEM = """You are a fact-checking assistant. Given a social media post, identify the single central factual claim that captures the main verifiable assertion of the post.

Rules:
- Output ONE clear, standalone factual claim — not multiple
- Tone and framing are irrelevant: accusatory, sarcastic, emotional, or rhetorical phrasing does NOT make a claim uncheckable
- Ignore pure opinions, predictions, questions, and personal sentiment
- Do NOT pre-judge whether the claim is true or false — that is determined later
- The claim must be independently checkable — no pronouns or unresolved references
- Use concrete named entities from the post; do not introduce information not in the post

Output ONLY the normalized claim as a single sentence, with no preamble, label, or explanation."""


def ours(post: str) -> Messages:
    return [
        {"role": "system", "content": _OURS_SYSTEM},
        {"role": "user", "content": f"Post:\n{post}"},
    ]


# --- 3. dfkinit2b zero-shot (Fig. 7) — #1 EN team's published zero-shot prompt --

_DFKINIT2B_SYSTEM = (
    "You are an expert in misinformation detection and fact-checking. "
    "Your task is to identify the central claim in the given post while "
    "preserving its original language."
)

_DFKINIT2B_USER = """You are an expert in misinformation detection and fact-checking. Your task is to identify the central claim in the given post while preserving its original language.
The central claim should meet the following criteria:
- **Verifiable**: It must be a factual assertion that can be checked against evidence.
- **Concise**: It should be a single, clear sentence that captures the main claim of the post.
- **Socially impactful**: It should be a statement that could influence public opinion, health, or policy.
- **Free from rhetorical elements**: Do not include opinions, rhetorical questions, or unnecessary context.
- **Preserve Original Language**: The output should be in the same language as the input post.
Output only the central claim without additional explanation or formatting.

Post: {post}
Normalized claim:"""


def dfkinit2b(post: str) -> Messages:
    return [
        {"role": "system", "content": _DFKINIT2B_SYSTEM},
        {"role": "user", "content": _DFKINIT2B_USER.format(post=post)},
    ]


# --- 4. DS@GT — #2 EN team's LLM prompt (with CoT) ------------------------------

_DSGT_SYSTEM = (
    "You are an assistant that, given a post, identifies the central "
    "check-worthy claim contained within it. Summarize it in one sentence."
)


def dsgt(post: str) -> Messages:
    return [
        {"role": "system", "content": _DSGT_SYSTEM},
        {
            "role": "user",
            "content": f"Identify the central claim in the given post: {post}\nLet's think step by step.",
        },
    ]


# --- 5. AKCIT-FN zero-shot prompt ----------------------------------------------

_AKCIT_PROMPT = (
    "You have received an informal and disorganized social media post. "
    "Summarize this post into a clear and concise statement, without adding "
    "any new information.\n\nPost: {post}"
)


def akcit(post: str) -> Messages:
    return [{"role": "user", "content": _AKCIT_PROMPT.format(post=post)}]


# --- 6. TIFIN 5W1H (verbatim from paper Listings 1+2) --------------------------

_TIFIN_SYSTEM = """You are an AI assistant that analyzes social media posts to extract factual claims. For each post, you will analyze it using the WH questions framework and extract the main factual claim. Make sure to reflect same language the post is mentioned in. If the post is in Hindi, respond in Hindi. Your output must be valid JSON with the following structure:
{
  "what": "Subject or topic of the post",
  "who": "Key individuals, organizations, or groups mentioned",
  "where": "Location information (if mentioned)",
  "when": "Time information (if mentioned)",
  "how": "Process information (if described)",
  "why": "Reason or motivation information (if explained)",
  "claim": "The single main factual crisp claim made in the post within 10-15 words"
}
If information for a particular field is not available, use an empty string. Also if information is not clearly written, don't assume anything from your end. Always stick to the post, don't add anything from your end. Keep things concise."""

_TIFIN_USER = """Carefully analyze the following social media post and answer each question thoughtfully to identify the main factual claim:

Post: {post}

Please answer each of these questions, based only on what is stated in the post:
1. What is the subject/topic of the post?
2. Who is the post talking about (key individuals, organizations, or groups)?
3. Where is this situation taking place (if mentioned)?
4. When did this situation take place (if mentioned)?
5. How did the situation take place (if described)?
6. Why did the situation take place (if explained)?

After answering these questions, extract the main factual claim being made in the post in a single, clear, concise sentence.
Provide your response in the specified JSON format:
{{
  "what": "...",
  "who": "...",
  "where": "...",
  "when": "...",
  "how": "...",
  "why": "...",
  "claim": "..."
}}"""


def tifin(post: str) -> Messages:
    return [
        {"role": "system", "content": _TIFIN_SYSTEM},
        {"role": "user", "content": _TIFIN_USER.format(post=post)},
    ]


# --- Output post-processing ----------------------------------------------------

def extract_claim(system_name: str, raw: str) -> str:
    """Pull the normalized claim from an LLM's raw response.

    TIFIN returns JSON with a 'claim' key. Others return the claim as plain text
    (sometimes with a 'Normalized claim:' label or stray formatting). Be forgiving.
    """
    text = raw.strip()
    # Strip <think>...</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Drop an unclosed <think> tail if max_tokens cut it off mid-reasoning
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if system_name == "tifin":
        try:
            data = json.loads(text)
            return str(data.get("claim", "")).strip()
        except json.JSONDecodeError:
            m = re.search(r'"claim"\s*:\s*"([^"]+)"', text)
            return m.group(1).strip() if m else text

    # Strip common labels other systems leak
    text = re.sub(r"^(Normalized [Cc]laim:?\s*|Claim:?\s*)", "", text)
    # Take only the first line / sentence to avoid CoT reasoning bleed
    text = text.split("\n")[0].strip()
    return text


# --- Registry -------------------------------------------------------------------

SYSTEMS: dict[str, Callable[[str], Messages] | None] = {
    "baseline": None,  # special: handled in run_eval
    "ours": ours,
    "dfkinit2b": dfkinit2b,
    "dsgt": dsgt,
    "akcit": akcit,
    "tifin": tifin,
}
