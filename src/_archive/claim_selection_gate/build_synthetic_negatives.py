"""Build + validate synthetic negative posts for the Stage-1 gate eval.

Produces eval/data/synthetic_negatives.parquet — posts the gate should REJECT
(expected_label="reject"). ~1000 posts drawn from three sources:

  A. Real X/Twitter posts captured from Daniel's feed (zeerover/*.ndjson) that are
     clearly non-check-worthy: fan reactions, personal opinions, entertainment
     commentary, sports reactions. These are filtered by lightweight heuristics and
     assigned categories.

  B. Hand-crafted synthetic posts across all 6 categories (including decorative-image
     and video-dependent, which are rare in the real feed). Pillow-generated images
     are attached for decorative-image posts.

  C. LLM-generated posts (DeepSeek V4-Flash) to fill each category to target counts.

Categories:
  opinion_personal       — personal updates, jokes, questions, greetings, reactions
  subjective_aesthetic   — aesthetic/taste judgments (films, music, food, sport)
  ad_promo               — ads, influencer promotions, giveaways, spam
  decorative_image       — chit-chat text + a scenic/food/selfie image (no claim in image)
  video_dependent        — vague text + video poster frame (claim lives in video → DROP)
  hard_negative          — assertive-sounding but fundamentally evaluative, not checkable

Fold-in loader note:
  To fold these negatives into the eval alongside real positives:

    import polars as pl
    negatives = pl.read_parquet("eval/data/synthetic_negatives.parquet")
    positives = (
        pl.read_parquet("eval/data/snopes_harvest.parquet")
        .filter(pl.col("has_image") == True)
        .with_columns(pl.lit("pass").alias("expected_label"))
        .rename({"claim_text": "text"})   # align text column for the gate
    )
    eval_df = pl.concat([positives, negatives], how="diagonal_relaxed")

Usage:
    uv run python -m eval.scripts.build_synthetic_negatives --smoke 4
    uv run python -m eval.scripts.build_synthetic_negatives
    uv run python -m eval.scripts.build_synthetic_negatives --no-gate --no-generate
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
from pathlib import Path

import polars as pl
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, EXTRACTION_MODEL

GATE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
GEN_MODEL = EXTRACTION_MODEL  # DeepSeek V4-Flash for LLM generation
NDJSON_DIR = Path("~/dev/disinform/zeerover")
OUT = Path("eval/data/synthetic_negatives.parquet")
IMG_DIR = Path("eval/data/images/synthetic")

# Category target counts
CATEGORY_TARGETS = {
    "opinion_personal": 250,
    "subjective_aesthetic": 150,
    "ad_promo": 150,
    "decorative_image": 120,
    "video_dependent": 100,
    "hard_negative": 130,
}
TOTAL_TARGET = sum(CATEGORY_TARGETS.values())  # 900


# =============================================================================
# Stage-1 gate prompt (per core_pipeline_spec.md)
# =============================================================================

GATE_SYSTEM = """\
You are a check-worthiness filter for a social-media fact-checking pipeline. \
Your job is to decide whether a post contains a claim worth fact-checking — NOT to judge whether the claim is true.

Read the full post carefully: resolve references ("this", "they", "the parent"), \
read any text visible in images (OCR), and consider what the user is asserting \
by posting (including any quoted tweet they are amplifying or endorsing).

A claim is check-worthy if ALL of the following hold:
1. public_interest: it concerns politics, economics, public health, science, \
current events, global affairs, or similar matters of societal consequence — \
NOT purely personal, entertainment, or lifestyle content.
2. misinfo_if_false: it would be CONSEQUENTIAL misinformation IF false — \
NOT an opinion, aesthetic judgment, advertisement, personal update, joke, or \
rhetorical question.

IMPORTANT:
- misinfo_if_false judges CONSEQUENCE, not truth. An obviously-true verifiable \
claim about public affairs is still check-worthy.
- A confident-sounding but fundamentally evaluative statement ("politicians lie", \
"the system is broken", "things were better before") is NOT a checkable factual \
claim — mark misinfo_if_false=false.
- A greeting, joke, question, ad, or purely personal update is NOT check-worthy.

Also identify claim_locus — where does the checkable claim live?
- "text": claim is stated in the post text
- "image": claim is in the image only (screenshot, OCR text, chart)
- "both": claim is in both text and image
- "parent": the quoted tweet carries the claim (user is amplifying/endorsing it)
- "video": the claim lives in a video (only a poster frame is available)
- "none": no checkable claim present

Output strictly valid JSON and nothing else:
{
  "reasoning": "what the post (main + quoted, text + image) asserts; resolve references, read image text, consider the quote-tweet dynamic BEFORE deciding",
  "public_interest": true or false,
  "misinfo_if_false": true or false,
  "claim_locus": "text|image|both|parent|video|none",
  "check_worthy": true or false
}
check_worthy MUST equal (public_interest AND misinfo_if_false).\
"""


# =============================================================================
# Shared JSON helpers (mirrors verify_prompts.py pattern)
# =============================================================================

def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_obj(text: str) -> dict:
    cleaned = _strip(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object found in: {text[:200]!r}")
        return json.loads(m.group(0))


def _chat(model: str, messages: list[dict], max_tokens: int = 600,
          json_mode: bool = True) -> str:
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        f"{EXTRACTION_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


# =============================================================================
# Gate
# =============================================================================

def run_gate(row: dict) -> dict:
    """Call the Stage-1 gate on one post dict. Returns parsed gate output."""
    content: list[dict] = []

    # Images first (labeled), per spec multimodal layout
    for i, p in enumerate((row.get("image_paths") or []), 1):
        label = "Image 1: main post" if i == 1 else f"Image {i}: main post (additional)"
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{b64}"}})

    text_block = f"Main post text: {row['text']}"
    if row.get("is_quote") and row.get("quoted_text"):
        text_block += f"\n\nQuoted tweet text: {row['quoted_text']}"
    if row.get("has_video"):
        text_block += (
            "\n\n[Note: this post contains a video. "
            "Only the poster frame (above) is available — the video itself cannot be processed.]"
        )

    content.append({"type": "text", "text": text_block})
    content.append({"type": "text", "text": "Evaluate this post for check-worthiness."})

    raw = _chat(GATE_MODEL,
                [{"role": "system", "content": GATE_SYSTEM},
                 {"role": "user", "content": content}],
                max_tokens=600)
    return _load_obj(raw)


# =============================================================================
# Decorative image generation (Pillow)
# =============================================================================

def _make_images(img_dir: Path) -> dict[str, str]:
    img_dir.mkdir(parents=True, exist_ok=True)

    def gradient(path: Path, top=(255, 180, 80), bot=(220, 70, 20), size=(800, 500)):
        img = Image.new("RGB", size)
        d = ImageDraw.Draw(img)
        for y in range(size[1]):
            t = y / size[1]
            c = tuple(int(top[k] + t * (bot[k] - top[k])) for k in range(3))
            d.line([(0, y), (size[0], y)], fill=c)
        img.save(path, format="WebP", quality=90)

    def gradient2(path: Path):  # blue-purple sky
        gradient(path, top=(100, 160, 240), bot=(60, 40, 120))

    def food_plate(path: Path):
        img = Image.new("RGB", (600, 600), color=(245, 235, 220))
        d = ImageDraw.Draw(img)
        d.ellipse([80, 80, 520, 520], fill=(230, 220, 200), outline=(180, 160, 140), width=10)
        d.ellipse([190, 190, 410, 410], fill=(170, 110, 55))
        d.ellipse([230, 250, 275, 295], fill=(70, 150, 50))
        d.ellipse([325, 250, 370, 295], fill=(70, 150, 50))
        img.save(path, format="WebP", quality=90)

    def coffee_cup(path: Path):
        img = Image.new("RGB", (500, 500), color=(240, 230, 215))
        d = ImageDraw.Draw(img)
        d.ellipse([120, 100, 380, 360], fill=(80, 50, 30), outline=(60, 35, 15), width=8)
        d.ellipse([155, 135, 345, 325], fill=(50, 30, 15))
        d.arc([340, 180, 420, 280], start=-30, end=120, fill=(80, 50, 30), width=12)
        img.save(path, format="WebP", quality=90)

    def selfie_abstract(path: Path):
        img = Image.new("RGB", (600, 750), color=(200, 215, 240))
        d = ImageDraw.Draw(img)
        d.ellipse([175, 90, 425, 360], fill=(220, 185, 145))
        d.ellipse([40, 380, 560, 850], fill=(90, 110, 180))
        img.save(path, format="WebP", quality=90)

    def landscape(path: Path):
        img = Image.new("RGB", (900, 550), color=(135, 195, 235))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 340, 900, 550], fill=(34, 120, 34))
        d.polygon([(180, 340), (370, 110), (560, 340)], fill=(90, 90, 95))
        d.polygon([(450, 340), (620, 160), (790, 340)], fill=(115, 115, 120))
        img.save(path, format="WebP", quality=90)

    def beach(path: Path):
        img = Image.new("RGB", (900, 550), color=(100, 180, 230))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 310, 900, 550], fill=(240, 220, 170))
        d.ellipse([600, 60, 760, 200], fill=(255, 240, 80))
        d.rectangle([0, 260, 900, 350], fill=(0, 130, 200))
        img.save(path, format="WebP", quality=90)

    def pet_photo(path: Path):
        img = Image.new("RGB", (600, 600), color=(220, 210, 200))
        d = ImageDraw.Draw(img)
        d.ellipse([150, 120, 450, 420], fill=(200, 160, 100))
        d.ellipse([155, 125, 245, 200], fill=(180, 140, 80))  # left ear
        d.ellipse([355, 125, 445, 200], fill=(180, 140, 80))  # right ear
        d.ellipse([220, 230, 260, 270], fill=(80, 60, 40))  # left eye
        d.ellipse([340, 230, 380, 270], fill=(80, 60, 40))  # right eye
        d.ellipse([270, 280, 330, 320], fill=(240, 160, 140))  # nose
        img.save(path, format="WebP", quality=90)

    def video_frame(path: Path):
        img = Image.new("RGB", (800, 450), color=(25, 25, 25))
        d = ImageDraw.Draw(img)
        d.polygon([(310, 165), (310, 285), (510, 225)], fill=(190, 190, 190))
        img.save(path, format="WebP", quality=90)

    specs = {
        "sunset": (gradient, {}),
        "sky": (gradient2, {}),
        "food": (food_plate, {}),
        "coffee": (coffee_cup, {}),
        "selfie": (selfie_abstract, {}),
        "landscape": (landscape, {}),
        "beach": (beach, {}),
        "pet": (pet_photo, {}),
        "video_frame": (video_frame, {}),
    }
    paths: dict[str, str] = {}
    for name, (fn, kwargs) in specs.items():
        path = img_dir / f"{name}.webp"
        fn(path, **kwargs)
        paths[name] = str(path)
    return paths


# =============================================================================
# Source A: Real X posts from the captured feed
# =============================================================================

# Patterns indicating check-worthy factual content — EXCLUDE these
_FACTUAL_PATTERNS = re.compile(
    r"\b("
    r"million[s]?|milliard[s]?|billion[s]?|trillion[s]?"  # economic figures
    r"|%|pour cent|percent"                                # statistics
    r"|BREAKING|alerte|urgent|exclusif|exclusive"          # breaking news
    r"|étude|study|survey|sondage|rapport|report"          # research claims
    r"|arrested|arrêté|condamné|convicted|sentenced"       # legal facts
    r"|killed|mort|dead|décédé|victims|victime|murder"     # incident facts
    r"|vaccine|vaccin|covid|virus|cancer|pandemic"         # health claims
    r"|elected|élu|won the election|a remporté l'élection" # election claims
    r"|signed.*(?:order|law|bill)|executive order|décret"  # legislative facts
    r"|bitcoin|crypto|\$\d+k|\$\d+,\d{3}"                  # financial prices
    r"|revealed|confirmed|admitted|exposed|leaked"         # revelation framing
    r"|according to|selon|d'après|source[s]? say"          # sourced claims
    r")\b",
    re.IGNORECASE,
)

# Additional hard-exclude patterns (URL-heavy headline-style posts)
_HEADLINE_PATTERNS = re.compile(
    r"^[🚨🔴⚡📢🗣️🔔⚠️]\s*[A-ZÀ-ɏ]",  # emoji + title-case headline
    re.UNICODE,
)

# FR-specific casual reaction markers that are safe
_FR_SAFE_REACTIONS = re.compile(
    r"\b(putain|franchement|bordel|wallah|wesh|ptdr|mdrr+|lmaooo|t\'as vu|"
    r"j'en reviens|c'est quoi|c'est quand|c'est qui|je vais hurler|"
    r"je comprends pas|j'adore|j'aime pas|too vrai|trop vrai|"
    r"quelqu'un a|vous pouvez|j'vous jure)\b",
    re.IGNORECASE,
)

# EN-specific casual reaction markers that are safe
_EN_SAFE_REACTIONS = re.compile(
    r"\b(lmao|lol|omg|tbh|ngl|istg|bruh|honestly|literally|fr fr|"
    r"no cap|deadass|lowkey|highkey|slay|bestie|periodt|iykyk|"
    r"not to be dramatic|me pretending|pov:|tell me why)\b",
    re.IGNORECASE,
)

# Sports fan content without factual claims
_FAN_HASHTAGS = re.compile(
    r"#(niska|psg|fraciv|mbappe|ligue1|ldc|nba|knicks|mapr|mariesaupremierregard|"
    r"teamfrance|equipedefrance|bleue|bleus|rugby|om|foot|ucl|cl|stadedefrance)",
    re.IGNORECASE,
)


def _classify_real_tweet(text: str, lang: str) -> str | None:
    """Assign a category to a real tweet, or None to exclude it.

    Very conservative: only keep posts that are UNAMBIGUOUSLY non-check-worthy.
    Core rule: a tweet with a t.co URL is excluded UNLESS it has a clear fan hashtag
    (because the URL could link to news/claims the gate will try to infer).
    """
    t = text.strip()

    # Basic quality gates
    if not t or len(t) < 15:
        return None
    if lang not in ("en", "fr"):
        return None
    # Exclude bare or near-bare URL tweets
    if re.match(r"^https?://\S+\s*$", t):
        return None

    has_url = bool(re.search(r"https?://t\.co/\S+", t))

    # Strip trailing URL(s) to get the main text
    main = re.sub(r"\s*https?://\S+", " [URL]", t).strip()
    main_clean = re.sub(r"\[URL\]", "", main).strip()
    if not main_clean:
        return None

    # Hard-exclude check-worthy content in main text
    if _FACTUAL_PATTERNS.search(main_clean):
        return None
    if len(main_clean) > 60 and _HEADLINE_PATTERNS.match(main_clean):
        return None

    # Hard-exclude: specific event/claim patterns
    _claim_event_patterns = re.compile(
        r"\b(trophy parade|press conference|executive order|bill in congress|"
        r"senate vote|nato summit|court ruling|supreme court|prime minister|"
        r"signed a|signed an|lived in|live in [A-Z]|palestine|israel|"
        r"gaza|ukraine|taliban|hamas|cia|fbi|nsa|dem staffers|staffers|"
        r"résultats officiels|victoire officielle|"
        r"is going nowhere|going nowhere|transfer|signing|signed for|"  # transfer rumour claims
        r"bakchich|corruption|bribery|"                                 # corruption claims
        r"accès.*(internet|fibre|4g|5g)|5g.*(clients|abonnés)|facture plein|"  # telecom service claims
        r"toujours pas de (4g|5g)|sans accès|panne)\b",
        re.IGNORECASE,
    )
    if _claim_event_patterns.search(main_clean):
        return None

    # Hard-exclude: tweets quoting a named person's specific claim
    # (e.g. "CEO: 'we have ...'", "Macron said ...", "[Quote]: ...")
    if re.search(r'(?:CEO|CTO|CFO|president|ministre|senator|MP|PM|")\s*[:""]', main_clean, re.I):
        return None

    # If tweet has a URL, only keep it if there's a clear fan hashtag marking it as
    # fan content (ticket link, fan reaction link, etc.)
    if has_url and not _FAN_HASHTAGS.search(t):
        return None

    # Now whitelist: must match at least one clear "safe" pattern

    # 1. Reply threads (first token is @handle) — NO URL required check since replies
    #    referencing URLs to external content are risky
    if re.match(r"^@\w+\b", main_clean) and not has_url:
        if not _FACTUAL_PATTERNS.search(main_clean):
            return "opinion_personal"

    # 2. FR safe-reaction posts
    if lang == "fr" and _FR_SAFE_REACTIONS.search(main_clean):
        return "opinion_personal"

    # 3. EN safe-reaction posts
    if lang == "en" and _EN_SAFE_REACTIONS.search(main_clean):
        return "opinion_personal"

    # 4. Fan hashtag posts with emotional content (sports/entertainment)
    if _FAN_HASHTAGS.search(t):
        emotional = re.search(
            r"[😭😂🔥❤️💔🤯😤😍🥲💀🙌😮💪🤌🫠🎉🪦😩😋🫶]|"
            r"\b(j'en peux plus|je vais craquer|trop fort|trop nul|dingue|"
            r"incroyable|magnifique|adorable|scandaleux|pitoyable|honteux|"
            r"insane|crazy|unreal|omg|lmao|wtf|love|hate|best|worst)\b",
            t, re.IGNORECASE,
        )
        if emotional:
            return "opinion_personal"
        return None

    # 5. Short personal posts: no URL, no factual markers, clearly personal/opinion tone
    if not has_url and len(main_clean) < 180:
        personal = re.search(
            r"\b(je |j'|i |my |mon |ma |mes |je suis|i'm|i am|"
            r"on a|we had|j'ai|j'avais|j'aime|je veux|i want|i need|"
            r"honestly|franchement|tbh|ngl|personnellement)\b",
            main_clean, re.IGNORECASE,
        )
        if personal and not _FACTUAL_PATTERNS.search(main_clean):
            return "opinion_personal"

    return None  # exclude (uncertain)


def load_real_tweets(ndjson_dir: Path, max_per_category: int = 300) -> list[dict]:
    """Load and pre-filter real tweets from ndjson files."""
    records: list[dict] = []
    for f in sorted(ndjson_dir.glob("*.ndjson")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    cat_counts: dict[str, int] = {}
    posts: list[dict] = []
    for r in records:
        text = r.get("full_text", "").strip()
        lang = r.get("lang", "")
        cat = _classify_real_tweet(text, lang)
        if cat is None:
            continue
        n = cat_counts.get(cat, 0)
        if n >= max_per_category:
            continue
        cat_counts[cat] = n + 1
        posts.append({
            "text": text,
            "image_paths": [],
            "has_video": False,
            "is_quote": False,
            "quoted_text": None,
            "expected_label": "reject",
            "category": cat,
            "lang": lang,
            "source": "real_feed",
        })

    print(f"  Real tweets: {len(posts)} kept from {len(records)} records")
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat}: {n}")
    return posts


# =============================================================================
# Source B: Hand-crafted synthetic posts
# =============================================================================

def hand_crafted_posts(imgs: dict[str, str]) -> list[dict]:
    """Return hand-crafted synthetic posts covering all 6 categories."""
    sunset, sky = imgs["sunset"], imgs["sky"]
    food, coffee = imgs["food"], imgs["coffee"]
    selfie = imgs["selfie"]
    landscape, beach = imgs["landscape"], imgs["beach"]
    pet = imgs["pet"]
    vframe = imgs["video_frame"]

    def p(text, cat, lang, *, image_paths=None, has_video=False,
          is_quote=False, quoted_text=None):
        return {
            "text": text, "image_paths": image_paths or [],
            "has_video": has_video, "is_quote": is_quote, "quoted_text": quoted_text,
            "expected_label": "reject", "category": cat, "lang": lang,
            "source": "hand_crafted",
        }

    opinion = [
        p("Just got back from the most relaxing weekend ever 😌 Soul fully recharged. Hope everyone had a great Sunday! 🌸", "opinion_personal", "en"),
        p("Hot take: pineapple on pizza is actually delicious and anyone who disagrees has no taste 🍕🍍", "opinion_personal", "en"),
        p("Good morning everyone! ☀️ Starting the week with a positive mindset. What's your goal this week?", "opinion_personal", "en"),
        p("I feel like the older I get the less I want to explain myself to anyone. Protect your peace 🕊️", "opinion_personal", "en"),
        p("Does anyone else get irrationally angry when someone chews with their mouth open? Asking for science 😅", "opinion_personal", "en"),
        p("Pouring one out for my Monday motivation that didn't survive the 9 am meeting ☕🪦 #Relatable", "opinion_personal", "en"),
        p("Can't believe how fast this year is going. Feels like January was last week 😳 #TimeFlies", "opinion_personal", "en"),
        p("Nothing beats a good cup of coffee and a quiet morning before the world wakes up ☕🌅", "opinion_personal", "en"),
        p("Thread: things I've learned after turning 30. This year changed everything for me 🧵 (1/10)", "opinion_personal", "en"),
        p("Why do people still talk on speakerphone in public?? Society is truly lost 😤 #MondayRage", "opinion_personal", "en"),
        p("Not to be dramatic but I think I was born for a different era. Does anyone else feel this way?", "opinion_personal", "en"),
        p("My dog looked at me like I betrayed him when I came home 2 hours late. The guilt is real 🐶💔", "opinion_personal", "en"),
        p("Okay but WHY is it so hard to find a dentist who takes new patients? Genuinely asking 🦷", "opinion_personal", "en"),
        p("Bonne journée à tous ! ☀️ Profitez bien de ce mardi, il passe vite 😊 #BonjourTwitter", "opinion_personal", "fr"),
        p("Franchement, les gens qui laissent leurs chiens aboyer toute la nuit méritent une amende salée 😤", "opinion_personal", "fr"),
        p("C'est quoi votre série du moment ? Je cherche quelque chose à regarder ce soir 🍿", "opinion_personal", "fr"),
        p("Le lundi matin en France c'est une épreuve nationale. Bon courage à tous 💪 #Lundi", "opinion_personal", "fr"),
        p("Pourquoi les supermarchés changent toujours la disposition des rayons ? C'est stressant pour rien 😭", "opinion_personal", "fr"),
        p("Nouvelle règle de vie : arrêter de répondre aux textos en moins de 2 secondes pour avoir l'air moins désespéré 😂", "opinion_personal", "fr"),
        # Quote posts
        p("Lol okay 😂", "opinion_personal", "en", is_quote=True,
          quoted_text="Thread on why avocado toast is actually cheaper than making coffee at home ☕🥑"),
        p("This!! So relatable 👇", "opinion_personal", "en", is_quote=True,
          quoted_text="Sometimes you just need to take a day off and do absolutely nothing. Recharging IS productive."),
        p("C'est exactement ça 😂 Trop vrai", "opinion_personal", "fr", is_quote=True,
          quoted_text="Le lundi matin quand le réveil sonne pour la 3ème fois 😩⏰ #Humour"),
        p("Same energy tbh 😭", "opinion_personal", "en", is_quote=True,
          quoted_text="Me pretending to be productive while answering emails from bed at 11am 💻"),
    ]

    aesthetic = [
        p("The new Beyoncé album is a MASTERPIECE. Honestly her best work in 20 years. Every track is flawless 🎶🔥 #Renaissance", "subjective_aesthetic", "en"),
        p("Season 3 of The Last of Us is peak television. Nothing even comes close this year. Don't @ me 📺", "subjective_aesthetic", "en"),
        p("The new Zara collection is absolutely stunning. The beige linen set is my summer everything 🤌 #Fashion", "subjective_aesthetic", "en"),
        p("Paris is overrated tbh. Prague is a much better European city and half the price. Fight me 🗺️", "subjective_aesthetic", "en"),
        p("Football was genuinely more beautiful in the 90s. No debate. Today's game is too tactical and sterile.", "subjective_aesthetic", "en"),
        p("That film was the most overhyped thing I've seen in years. Two hours of my life I'm not getting back 😑", "subjective_aesthetic", "en"),
        p("Obsessed with this look 😍👏", "subjective_aesthetic", "en", is_quote=True,
          quoted_text="Finally dropped my new collection. What do you think? #Fashion #NewDrop"),
        p("Couldn't agree more 🙌", "subjective_aesthetic", "en", is_quote=True,
          quoted_text="The Beatles will never be topped. Their catalogue is the GOAT of popular music, end of discussion."),
        p("The architecture in that new district is soulless. Just glass boxes. Where's the humanity?? 😔", "subjective_aesthetic", "en"),
        p("Gordon Ramsay is the most overrated chef alive. Anyone who disagrees has clearly never eaten in Lyon.", "subjective_aesthetic", "en"),
        p("Le nouveau film de Lanthimos est génial, un chef-d'œuvre absolu. Le meilleur de l'année sans aucun doute 🎬 #Cinéma", "subjective_aesthetic", "fr"),
        p("Ce café à Lyon est le meilleur de toute la France, aucune discussion. Les croissants sont incroyables ☕🥐", "subjective_aesthetic", "fr"),
        p("Cet album est une catastrophe musicale. Comment peut-on appeler ça de la musique ? 😤 #Critique", "subjective_aesthetic", "fr"),
        p("Franchement le cinéma français des années 90 reste imbattable. Les films d'aujourd'hui sont trop lisses.", "subjective_aesthetic", "fr"),
        p("Aucun match ne ressemble à un El Classico. Les autres rivalités c'est du folklore comparé à ça. ⚽🔥", "subjective_aesthetic", "fr"),
    ]

    ads = [
        p("🚨 FLASH SALE 🚨 Up to 70% off everything at StyleHaus! Use code SUMMER70 at checkout. Ends tonight at midnight! 🛍️ #Sale", "ad_promo", "en"),
        p("I've been using this skincare brand for 3 weeks and my skin is GLOWING 🌟 Link in bio. Use my code GLOW20 for 20% off! #AD #Skincare", "ad_promo", "en"),
        p("📢 LIMITED SEATS for our Productivity Masterclass! Learn what top CEOs do every morning. Register now ➡️ bit.ly/xxxxx #Entrepreneur", "ad_promo", "en"),
        p("My dentist uses this exact brand on her own teeth. I was skeptical but now I'm obsessed 😍 [sponsored] Link in bio", "ad_promo", "en"),
        p("Finally found an app that makes managing my finances actually fun 📈 Check it out — link in bio, code SAVE10 for 10% off premium 💸", "ad_promo", "en"),
        p("Win a $500 Amazon gift card! RT + follow to enter 🎁 Drawing this Friday! Good luck #Giveaway #Free", "ad_promo", "en"),
        p("Friends — I just launched my online coaching program 🎉 First 10 spots are 50% off. Link in bio!", "ad_promo", "en"),
        p("This new skincare routine has been a total game changer for me. So obsessed 💊✨ Use BEAUTY15 for 15% off. Not a drill.", "ad_promo", "en"),
        p("LAST CHANCE: our Black Friday deal ends in 2 hours ⏰ 40% off everything. Go go go 🛒 [link in bio]", "ad_promo", "en"),
        p("Just started using this meal kit service and honestly it's worth every penny. Code FIRSTBOX for your first box free! 🍽️ #sponsored", "ad_promo", "en"),
        p("🎉 SOLDES D'ÉTÉ jusqu'à -60% sur notre boutique en ligne ! Code promo : ÉTÉ60 valable jusqu'à dimanche soir 🛒 #Promo #Soldes", "ad_promo", "fr"),
        p("Cette routine beauté a tout changé pour moi 💆‍♀️ Trop contente de ce partenariat ! Lien en bio, code MINCE15 pour -15% #Partenariat", "ad_promo", "fr"),
        p("🎁 Tentez de gagner un séjour pour 2 à Barcelone ! RT + abonnement requis. Tirage vendredi ! #Concours #Voyage", "ad_promo", "fr"),
        p("Je travaille avec cette marque depuis 6 mois et je n'ai jamais eu autant de compliments sur ma peau 💆‍♀️ Code LUCIE20 pour -20% [partenariat]", "ad_promo", "fr"),
    ]

    decorative = [
        p("Golden hour hits different on a Thursday 🌅 No filter needed. #Sunset #Nature",
          "decorative_image", "en", image_paths=[sunset]),
        p("Made homemade pasta for the first time!! So proud of myself 🍝 Nonna would approve",
          "decorative_image", "en", image_paths=[food]),
        p("New haircut, new me 💇‍♀️✨ What do you think? #Selfie #GlowUp",
          "decorative_image", "en", image_paths=[selfie]),
        p("Weekend hike with the squad 🏔️❤️ Nature really is the best therapy #Hiking #Mountains",
          "decorative_image", "en", image_paths=[landscape]),
        p("Sunday brunch vibes 🥞☕ Blueberry pancakes with maple syrup because why not? #FoodPhotography",
          "decorative_image", "en", image_paths=[food]),
        p("The sky outside my window right now 🌅 I could stare at this all day. #GoodMorning",
          "decorative_image", "en", image_paths=[sunset]),
        p("Beach day ☀️🌊 Sometimes you just need to hear the waves and do absolutely nothing",
          "decorative_image", "en", image_paths=[beach]),
        p("My morning ritual ☕ Coffee and silence before the chaos begins",
          "decorative_image", "en", image_paths=[coffee]),
        p("Rescued this little guy last week 🐾 Still can't believe how lucky I am 🥹",
          "decorative_image", "en", image_paths=[pet]),
        p("Late afternoon walk = free therapy 🚶‍♀️ This view never gets old",
          "decorative_image", "en", image_paths=[landscape]),
        p("Coucher de soleil depuis la terrasse ce soir 🌅😍 La vie est belle #NatureFrance",
          "decorative_image", "fr", image_paths=[sunset]),
        p("Ma tarte aux pommes du dimanche 🍎 Je suis assez fier de moi ! #Cuisine #FaitMaison",
          "decorative_image", "fr", image_paths=[food]),
        p("Randonnée dans les Alpes ce matin 🏔️🇫🇷 Une vue à couper le souffle ! #Montagne",
          "decorative_image", "fr", image_paths=[landscape]),
        p("Mon café du matin ☕ Le seul moment de paix avant que tout commence #Matinal",
          "decorative_image", "fr", image_paths=[coffee]),
        p("J'ai adopté ce petit bonheur il y a deux semaines 🐶 Je l'adore déjà trop",
          "decorative_image", "fr", image_paths=[pet]),
        p("Le ciel bleu de ce soir 🌌 Pas besoin de filtre pour trouver ça magnifique",
          "decorative_image", "fr", image_paths=[sky]),
    ]

    video = [
        p("The minister just admitted it in this interview. Watch until the end 👇",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("This leaked footage changes everything. Judge for yourself 📹",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("Watch the whole clip and judge for yourself. Make your own mind up 🎥",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("BREAKING: footage has emerged. Watch before it gets taken down 📹",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("You HAVE to watch this 👇 I can't believe what just happened",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("This documentary explains everything. Worth every second of the 2-hour runtime 🎬",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("Raw footage from the scene. Draw your own conclusions 🎥",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("His speech at the conference. What they're not telling you is at 4:20 in 👇",
          "video_dependent", "en", image_paths=[vframe], has_video=True),
        p("Regardez cette vidéo jusqu'au bout, vous verrez ce que je veux dire 👇",
          "video_dependent", "fr", image_paths=[vframe], has_video=True),
        p("Cette vidéo fait le tour du web et je comprends pourquoi. À voir impérativement 📹",
          "video_dependent", "fr", image_paths=[vframe], has_video=True),
        p("L'enregistrement complet. Formez votre propre opinion en regardant jusqu'à la fin 🎥",
          "video_dependent", "fr", image_paths=[vframe], has_video=True),
        p("Images exclusives. Vous jugerez par vous-même 📹",
          "video_dependent", "fr", image_paths=[vframe], has_video=True),
    ]

    hard = [
        p("The government has FAILED the working class. Completely and utterly failed. This isn't an opinion — it's obvious to anyone paying attention 📢 #Politics",
          "hard_negative", "en"),
        p("Big Pharma doesn't want you healthy, they want you dependent. Wake up people 👁️ #BigPharma #Truth",
          "hard_negative", "en"),
        p("Politicians are the same no matter the party. They promise everything and deliver nothing. Always been that way, always will be 🤷‍♂️",
          "hard_negative", "en"),
        p("Social media is making an entire generation mentally ill and no one in power cares enough to do anything 😔 #MentalHealth",
          "hard_negative", "en"),
        p("This country is being run into the ground and the mainstream media refuses to admit it 🔥 Open your eyes!",
          "hard_negative", "en"),
        p("It's always the same in this world. The system is completely broken and I'm tired of pretending otherwise 💸 #Inequality",
          "hard_negative", "en"),
        p("Schools don't teach people to think for themselves. Never have, probably never will 🎓",
          "hard_negative", "en"),
        p("Everything wrong with modern politics can be summed up in one word: greed. Full stop 🤦‍♂️",
          "hard_negative", "en"),
        p("There are two sets of rules in this world and we all know it. Has nothing to do with politics 🙄",
          "hard_negative", "en"),
        p("The mainstream narrative and reality are two very different things. Always have been 👁️",
          "hard_negative", "en"),
        p("The housing situation for my generation is genuinely hopeless and the people in power clearly don't care.",
          "hard_negative", "en"),
        p("We talk about freedom but everything I observe tells me we have less of it than we used to. Just an observation.",
          "hard_negative", "en"),
        p("Le système protège toujours les mêmes et sacrifie toujours les mêmes. C'est une opinion mais je la maintiens 😤 #Politique",
          "hard_negative", "fr"),
        p("Les médias nous donnent leur version des choses. Cherchez vous-mêmes, vous serez étonnés 👁️ #Vérité",
          "hard_negative", "fr"),
        p("On nous raconte des histoires depuis des années et on les accepte parce que c'est plus simple 🙄",
          "hard_negative", "fr"),
        p("Les politiciens font tous à peu près la même chose une fois qu'ils ont le pouvoir. C'est mon impression depuis longtemps.",
          "hard_negative", "fr"),
    ]

    return opinion + aesthetic + ads + decorative + video + hard


# =============================================================================
# Source C: LLM-generated posts (DeepSeek V4-Flash)
# =============================================================================

GEN_PROMPTS = {
    "opinion_personal": {
        "en": """Generate 10 distinct realistic X/Twitter posts that are personal opinions, reactions, \
jokes, questions, or personal life updates. These should NOT contain any verifiable factual claims \
about public affairs. Authentic Twitter style: emojis, hashtags, casual language, varying length \
(30–280 chars). Mix of tones (humorous, reflective, frustrated, excited). Output JSON: \
{"posts": ["<post1>", "<post2>", ...]}.  Do not number them. All in English.""",
        "fr": """Génère 10 publications distinctes et réalistes style X/Twitter qui sont des opinions personnelles, \
réactions, blagues, questions ou mises à jour de vie personnelle. Ne doivent PAS contenir de faits \
vérifiables sur des affaires publiques. Style authentique Twitter en français : emojis, hashtags, \
langage familier. Longueur variée (30–280 chars). Output JSON : {"posts": ["<post1>", ...]}.  Ne pas numéroter.""",
    },
    "subjective_aesthetic": {
        "en": """Generate 10 distinct realistic X/Twitter posts expressing subjective opinions about culture, \
entertainment, art, food, travel, or sport. These are aesthetic/taste judgments — not verifiable \
factual claims. Authentic Twitter style: emojis, hashtags, casual language, strong opinions. \
Output JSON: {"posts": ["<post1>", "<post2>", ...]}. All in English.""",
        "fr": """Génère 10 publications style X/Twitter exprimant des jugements subjectifs sur la culture, \
l'entertainment, la gastronomie, le voyage ou le sport. Ce sont des jugements de goût, pas des faits \
vérifiables. Style authentique Twitter en français : emojis, hashtags, opinions tranchées. \
Output JSON : {"posts": ["<post1>", ...]}. En français.""",
    },
    "ad_promo": {
        "en": """Generate 10 distinct realistic X/Twitter posts that are ads, promotions, influencer posts, \
spam, or giveaways. These should look like social media marketing content. Include discount codes, \
calls to action, sponsored labels, giveaway mechanics. Output JSON: {"posts": ["<post1>", ...]}. All in English.""",
        "fr": """Génère 10 publications style X/Twitter qui sont des publicités, promotions, posts d'influenceurs \
ou concours. Doivent ressembler à du contenu marketing. Inclure codes promo, appels à l'action, \
mentions [partenariat] ou #sponsored. Output JSON : {"posts": ["<post1>", ...]}. En français.""",
    },
    "hard_negative": {
        "en": """Generate 10 distinct realistic X/Twitter posts that SOUND assertive about political/social topics \
but are fundamentally evaluative opinions, not verifiable factual claims. Examples: normative \
judgments ("the system is broken"), vague generalizations ("politicians always lie"), \
conspiratorial tone without specific verifiable claims. Should pass as opinionated commentary \
but not as factual claims a fact-checker could verify. Authentic Twitter style. \
Output JSON: {"posts": ["<post1>", ...]}. All in English.""",
        "fr": """Génère 10 publications style X/Twitter qui SEMBLENT assertives sur des sujets politiques/sociaux \
mais sont fondamentalement des jugements de valeur, pas des faits vérifiables. Exemples : \
jugements normatifs ("le système est cassé"), généralisations vagues ("les politiciens mentent toujours"), \
ton conspirationniste sans affirmations factuelles précises. Style Twitter authentique en français. \
Output JSON : {"posts": ["<post1>", ...]}. En français.""",
    },
}


def generate_posts_llm(category: str, lang: str, n_batches: int = 2) -> list[dict]:
    """Generate posts for a category+lang using DeepSeek V4-Flash (10 posts per call)."""
    if category not in GEN_PROMPTS or lang not in GEN_PROMPTS.get(category, {}):
        return []

    prompt = GEN_PROMPTS[category][lang]
    posts: list[dict] = []

    for batch in range(n_batches):
        try:
            raw = _chat(GEN_MODEL,
                        [{"role": "system", "content": "You generate realistic social media posts. Output only the JSON requested."},
                         {"role": "user", "content": prompt}],
                        max_tokens=1500, json_mode=True)
            data = _load_obj(raw)
            batch_posts = data.get("posts", [])
            if not isinstance(batch_posts, list):
                continue
            for text in batch_posts:
                if isinstance(text, str) and len(text.strip()) > 10:
                    posts.append({
                        "text": text.strip(),
                        "image_paths": [],
                        "has_video": False,
                        "is_quote": False,
                        "quoted_text": None,
                        "expected_label": "reject",
                        "category": category,
                        "lang": lang,
                        "source": "llm_generated",
                    })
        except Exception as e:
            print(f"    LLM generation error ({category}/{lang}): {e}", flush=True)

    return posts


# =============================================================================
# Real-positive spot-check
# =============================================================================

def spot_check_positives(n: int = 10) -> dict:
    src = Path("eval/data/snopes_harvest.parquet")
    df = pl.read_parquet(src)
    rows = (
        df.filter(
            (pl.col("has_image") == True)
            & pl.col("image_paths").list.len().gt(0)
            & pl.col("claim_text").is_not_null()
        )
        .head(n)
        .to_dicts()
    )

    results = []
    for i, row in enumerate(rows, 1):
        preview = (row.get("raw_context") or row["claim_text"])[:60]
        print(f"  [{i:2d}/{n}] {preview!r}...", flush=True)
        try:
            gate = run_gate({
                "text": row.get("raw_context") or row["claim_text"],
                "image_paths": row["image_paths"][:1],
                "has_video": False,
                "is_quote": False,
                "quoted_text": None,
            })
            cw = gate.get("check_worthy", False)
            locus = gate.get("claim_locus", "")
            passed = cw  # for positives: passed = check_worthy=True (and locus!=video)
            if locus == "video":
                passed = False  # video posts are dropped (legitimate positive gets dropped)
            results.append({
                "claim": row["claim_text"][:80],
                "check_worthy": cw,
                "claim_locus": locus,
                "passed": passed,
            })
        except Exception as e:
            results.append({"claim": row["claim_text"][:80], "error": str(e), "passed": None})

    valid = [r for r in results if r.get("passed") is not None]
    n_passed = sum(1 for r in valid if r.get("passed"))
    print(f"\nReal-positive spot-check: {n_passed}/{len(valid)} passed (want ~100%)")
    return {"results": results, "pass_rate": n_passed / len(valid) if valid else 0}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", type=int, default=0,
                    help="gate-validate only the first N negatives (smoke test)")
    ap.add_argument("--no-gate", action="store_true",
                    help="build the parquet but skip gate validation")
    ap.add_argument("--no-generate", action="store_true",
                    help="skip LLM generation (use real + hand-crafted only)")
    ap.add_argument("--no-spotcheck", action="store_true",
                    help="skip the real-positive spot-check")
    ap.add_argument("--validate-sample", type=int, default=120,
                    help="number of negatives to gate-validate in full mode (default 120)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Decorative images
    # ------------------------------------------------------------------
    print("Generating decorative images...", flush=True)
    imgs = _make_images(IMG_DIR)
    print(f"  Saved {len(imgs)} images to {IMG_DIR}", flush=True)

    # ------------------------------------------------------------------
    # 2. Real feed tweets (Source A)
    # ------------------------------------------------------------------
    print(f"\nLoading real feed tweets from {NDJSON_DIR}...", flush=True)
    if NDJSON_DIR.exists():
        real = load_real_tweets(NDJSON_DIR, max_per_category=300)
    else:
        print(f"  WARNING: {NDJSON_DIR} not found; skipping real tweets")
        real = []

    # ------------------------------------------------------------------
    # 3. Hand-crafted synthetic (Source B)
    # ------------------------------------------------------------------
    print("\nBuilding hand-crafted posts...", flush=True)
    crafted = hand_crafted_posts(imgs)
    print(f"  Hand-crafted: {len(crafted)} posts")

    all_posts = real + crafted

    # ------------------------------------------------------------------
    # 4. LLM generation (Source C) — fill each category to target
    # ------------------------------------------------------------------
    if not args.no_generate:
        from collections import Counter
        cat_counts = Counter(p["category"] for p in all_posts)
        print("\nLLM generation to fill category targets...", flush=True)
        for cat, target in CATEGORY_TARGETS.items():
            if cat in ("decorative_image", "video_dependent"):
                # These need images — skip LLM generation (hand-crafted + real cover them)
                continue
            current = cat_counts.get(cat, 0)
            needed = max(0, target - current)
            if needed <= 0:
                print(f"  {cat}: already at {current}/{target} — skipping")
                continue
            print(f"  {cat}: {current}/{target} — generating ~{needed}...", flush=True)
            # Distribute evenly between EN and FR; cap at 3 batches × 10 posts = 30 per lang
            for lang in ("en", "fr"):
                lang_needed = needed // 2
                if lang_needed <= 0:
                    continue
                n_batches = min(3, max(1, lang_needed // 10))
                generated = generate_posts_llm(cat, lang, n_batches=n_batches)
                print(f"    {lang}: generated {len(generated)}", flush=True)
                all_posts.extend(generated)
    else:
        print("\n--no-generate: skipping LLM generation")

    # ------------------------------------------------------------------
    # 5. Build and save parquet
    # ------------------------------------------------------------------
    print(f"\nCorpus: {len(all_posts)} posts total", flush=True)
    from collections import Counter
    cat_counts = Counter(p["category"] for p in all_posts)
    lang_counts = Counter(p["lang"] for p in all_posts)
    src_counts = Counter(p.get("source", "unknown") for p in all_posts)
    for label, counter in [("category", cat_counts), ("lang", lang_counts), ("source", src_counts)]:
        print(f"  {label}: {dict(counter.most_common())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Normalise quoted_text: ensure it's always str or None (never missing)
    for p in all_posts:
        if p.get("quoted_text") is None:
            p["quoted_text"] = None  # keep None, will become Utf8 null
    df = pl.DataFrame(
        [{k: v for k, v in p.items()} for p in all_posts],
        schema_overrides={
            "image_paths": pl.List(pl.Utf8),
            "quoted_text": pl.Utf8,
            "source": pl.Utf8,
        },
        infer_schema_length=None,
    )
    df.write_parquet(OUT)
    print(f"\nWrote {OUT} ({df.height} rows, {df.width} cols)", flush=True)

    # ------------------------------------------------------------------
    # 6. Gate validation
    # ------------------------------------------------------------------
    if args.no_gate:
        print("--no-gate: skipping gate validation")
    else:
        if args.smoke:
            to_validate = all_posts[:args.smoke]
            print(f"\nSmoke: validating first {args.smoke} posts...")
        else:
            # Stratified sample across categories
            by_cat: dict[str, list] = {}
            for p in all_posts:
                by_cat.setdefault(p["category"], []).append(p)
            to_validate: list[dict] = []
            per_cat = max(1, args.validate_sample // len(by_cat))
            for cat, posts in by_cat.items():
                sample = random.sample(posts, min(per_cat, len(posts)))
                to_validate.extend(sample)
            random.shuffle(to_validate)
            to_validate = to_validate[:args.validate_sample]
            print(f"\nValidating stratified sample of {len(to_validate)} posts...")

        false_positives: list[dict] = []
        gate_rows: list[dict] = []
        errors = 0

        for i, row in enumerate(to_validate, 1):
            prefix = f"[{i:3d}/{len(to_validate)}] {row['category']:22s} {row['lang']:2s}"
            try:
                gate = run_gate(row)
                cw = gate.get("check_worthy", False)
                locus = gate.get("claim_locus", "")
                rejected = (not cw) or (locus == "video")
                status = "OK (rejected)" if rejected else "FP (wrongly passed)"
                print(f"  {prefix}  check_worthy={str(cw):5s}  locus={locus:<8}  {status}", flush=True)
                gate_rows.append({**row, "gate_check_worthy": cw, "gate_locus": locus,
                                   "gate_rejected": rejected,
                                   "gate_reasoning": gate.get("reasoning", "")[:300]})
                if not rejected:
                    false_positives.append({
                        "category": row["category"], "lang": row["lang"],
                        "text": row["text"][:120],
                        "check_worthy": cw, "locus": locus,
                        "reasoning": gate.get("reasoning", "")[:300],
                    })
            except Exception as e:
                print(f"  {prefix}  ERROR: {e}", flush=True)
                errors += 1

        n_done = len(gate_rows)
        n_fp = len(false_positives)
        fp_rate = n_fp / n_done if n_done else 0

        print(f"\n{'=' * 70}")
        print(f"Gate validation: {n_done} evaluated | {n_fp} false positives | "
              f"{errors} errors | FP rate: {fp_rate:.1%}")

        if false_positives:
            print("\nFalse positives (posts the gate wrongly passed):")
            for fp in false_positives:
                print(f"  [{fp['category']:22s} {fp['lang']:2s}] "
                      f"check_worthy={fp['check_worthy']} locus={fp['locus']}")
                print(f"    text:      {fp['text']}")
                print(f"    reasoning: {fp['reasoning'][:150]}")
        else:
            print("  None — all evaluated negatives correctly rejected.")

        # Annotate parquet with gate results for the validated subset
        if gate_rows:
            gate_df = pl.DataFrame([
                {"text": r["text"], "category": r["category"],
                 "gate_check_worthy": r["gate_check_worthy"],
                 "gate_locus": r["gate_locus"],
                 "gate_rejected": r["gate_rejected"],
                 "gate_reasoning": r["gate_reasoning"]}
                for r in gate_rows
            ])
            annotated = df.join(gate_df, on=["text", "category"], how="left")
            annotated.write_parquet(OUT)
            print(f"\nAnnotated parquet written ({annotated.filter(pl.col('gate_check_worthy').is_not_null()).height} rows with gate results)")

    # ------------------------------------------------------------------
    # 7. Real-positive spot-check
    # ------------------------------------------------------------------
    if not args.no_spotcheck and not args.no_gate:
        print(f"\n{'=' * 70}")
        print("Spot-checking 10 real positives from snopes_harvest...")
        spot = spot_check_positives(10)
        print(f"\nPass rate: {spot['pass_rate']:.0%}")
        for r in spot["results"]:
            sym = "OK" if r.get("passed") else ("FP" if r.get("passed") is False else "ERR")
            print(f"  [{sym}] {r.get('claim', '?')[:70]}")
            print(f"        check_worthy={r.get('check_worthy')}  locus={r.get('claim_locus')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
