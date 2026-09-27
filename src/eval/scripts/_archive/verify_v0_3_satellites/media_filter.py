"""Strict media-dependency filter for the Stage-3 text-only split (EN + FR + ES aware).

A claim is EXCLUDED if its verification targets a specific media/visual/recorded artifact
(photo/video/screenshot/map/chart/audio/"authentic post"/etc.) — i.e. it requires
authenticating or reading the artifact, which a text-search verifier structurally cannot do.
has_image/has_video are NOT exclusion signals (an attached image on a textual claim is
serialized as context); only the CLAIM TEXT (or the altered_media rating subtype) drives
exclusion. Built by manual eyeball of the dev keep pool — multilingual + artifact-authenticity
gaps were found by reading, not assumed.
"""
import hashlib
import re

import polars as pl

def claim_id(u): return hashlib.md5((u or "").encode()).hexdigest()[:16]

# visual/recorded artifact nouns — EN + FR + ES
_VIS = (r"(photos?|photographs?|pictures?|images?|imagery|imagen|im[áa]genes|fotos?|"
        r"videos?|vid[ée]os?|footage|clips?|s[ée]quences?|screenshots?|screen ?grabs?|captura|capturas|"
        r"selfies?|memes?|webcams?|livestreams?|broadcasts?|montages?|gifs?|visuals?|clich[ée]s?|"
        r"maps?|cartes?|charts?|graphics?|diagrams?|banners?|paintings?|posters?|billboards?|"
        r"cartoons?|drawings?|sketches?|mugshots?|mug shots?|ads?|adverts?|advertisements?|"
        r"graffiti|depictions?)")
# authenticity / manipulation words — EN + FR + ES
_AUTH = (r"(authentic|authentique|aut[ée]ntic\w*|genuine|\breal\b|fake|truqu\w*|trucage|doctored|"
         r"altered|staged|manipul\w*|\bedit|unedited|(?:ai|artificial intelligence)[- ]?gener|deepfake|"
         r"miscaption|recycl|photoshop|splice|cgi|digitally|out of context|posed)")
# "shows/depicts" verbs — EN + FR + ES
_SHOW = r"(show|depict|document|capture|circulat|appear|prove|reveal|portray|montr|muestr|aparec)"
# recorded/posted artifacts whose AUTHENTICITY is the claim
_ARTI = r"(posts?|letters?|audio|recordings?|tapes?|voicemails?|memos?|screenshots?|emails?|articles?|songs?)"
_LEAD = r"(a |an |the |this |these |real |genuine |authentic |old |la |le |les |des |une |un |ces |cette |esta |esa |los |las |estas |unas? )"

_PATS = [re.compile(p, re.I) for p in (
    rf"\b{_AUTH}\w*\b.{{0,40}}\b{_VIS}\b",
    rf"\b{_VIS}\b.{{0,40}}\b{_AUTH}\w*",
    rf"\b{_VIS}\b.{{0,25}}\b{_SHOW}\w*\b",
    rf"\b{_SHOW}\w*\b.{{0,15}}\b{_VIS}\b",
    rf"^\W*{_LEAD}*{_VIS}\b",
    rf"\b(posted|shared|tweeted|circulat\w*)\b.{{0,25}}\b{_VIS}\b",
    rf"\b(true|real|fake|authentic|genuine|actual|leaked|truqu\w*)\b.{{0,18}}\b(video|vid[ée]o|photo|image|footage|clip|recording|audio|screenshot|tape)\b",
    rf"\b(authentic|real|actual|genuine|fake|leaked)\b.{{0,25}}\b{_ARTI}\b",
    rf"\b{_ARTI}\b.{{0,18}}\b(is|are|was|were)\b.{{0,12}}\b(authentic|real|fake|unedited|genuine|doctored|altered)",
    r"\bon video\b",
    rf"\b(post|posts|posted|posting|shares?|shared|sharing|publish\w*|circulat\w*|tweet\w*)\b.{{0,14}}\b(photos?|videos?|vid[ée]os?|pictures?|images?|depictions?|songs?|clips?|footage|memes?|gifs?)\b",
    r"\b(in|on|from)\b\s+(a |an |the )?(videos?|vid[ée]os?|clips?|footage|livestreams?|recordings?)\b",
    r"\bdans (une?|la|le|les|des) (vid[ée]os?|photos?|images?|s[ée]quences?|clips?|enregistrements?)\b",
    r"\b(videos?|vid[ée]os?|clips?|footage|photos?|pictures?|images?|audio|recordings?)\b\s+(of|from)\b",
    r"\b(FBI|police|security|cctv|surveillance|leaked|secret|hidden ?camera)\b\s+(video|footage|audio|recording|clip|photo|image)\b",
)]

# Residual media-dependent claims with NO artifact noun the regex can key on (a pure photo/video
# scene description). Matched by a unique claim_text substring; found by manual eyeball.
MANUAL_EXCLUDE_SUBSTR = [
    "Chang's billboard",
    "showing Kilmar Abrego Garcia",
    "Trump posted a depiction of himself",
    "Israeli soldiers in front of ruins in Gaza forming a Hanukkah menorah",
    "A Secret Service agent standing with Donald Trump after he was shot",
    "black man steals headphones off train commuter",
    "NASA used Algeria images",                            # image provenance (fake Mars)
    "TikTok videos about",                                # video content claim
    "banner photo of Joe Biden",                          # a photo/banner artifact
    "A BBC article shows Martin Lewis",                   # deepfake scam video/article
    'A letter called "The Brown Round-Up Part 1"',        # letter authenticity
    "the woman who posted a long message on Facebook",    # post identity/authenticity
    "posted a link on Truth Social that displayed a symbol",  # posted visual symbol
    "Are Authentic",                                      # "Posts … Are Authentic"
    "authentic article whose headline",                  # article authenticity
    "Audio circulating in September 2025 was from a real 911 call",  # audio authenticity
    '"Trump Gaza" video was published',                   # a specific AI video
]

def media_text(t: str | None) -> bool:
    t = t or ""
    return any(p.search(t) for p in _PATS)

def is_media_dependent(claim_text: str | None, rating_subtype: str | None) -> bool:
    if rating_subtype == "altered_media":
        return True
    if media_text(claim_text):
        return True
    return any(s in (claim_text or "") for s in MANUAL_EXCLUDE_SUBSTR)

def tag(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.struct(["claim_text", "rating_subtype"]).map_elements(
        lambda s: is_media_dependent(s["claim_text"], s["rating_subtype"]),
        return_dtype=pl.Boolean).alias("media_dependent"))
