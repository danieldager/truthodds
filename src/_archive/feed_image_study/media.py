"""Image capture for the image-native extraction dataset — resolve a post's media URLs, download
them, and downscale to a VLM token budget so the saved images stay generally useful but cheap.

Real X/social posts are image-heavy and the checkable content often lives IN the image (memes,
screenshots of fabricated headlines, photos of signs). The downstream extraction VLM is Qwen3-VL
on DeepInfra, so the saved resolution is tuned to ITS billing.

Qwen3-VL token math (verified, NOT a rule-of-thumb): the vision tower uses patch_size=16 and a 2x2
merge, so one image token covers a 32x32-px cell, and the total is area-bounded. `est_qwen_tokens`
replicates the HF Qwen3-VL image processor's smart_resize EXACTLY (matched the processor on every
sampled image) and the DeepInfra `usage.prompt_tokens` delta matched it to +2 structural tokens
(clog 260626). So ~1 MP ~= 1024 tokens; the default ~1280-token cap lands near 1.3 MP.

Lives apart from source_fetch.py (which owns post-text resolution) to keep these Pillow/IO helpers
import-light; source_fetch wires the URL extractors into its resolution path.
"""
from __future__ import annotations

import io
import math
import re

import requests
from PIL import Image

from eval.claimreview import UA

# --- Qwen3-VL token estimation (from the model's preprocessor_config: patch_size 16, merge_size 2,
#     size.shortest_edge 65536 px^2, longest_edge 16777216 px^2) ---
_CELL = 32              # patch_size(16) * merge_size(2) -> px per image token, each axis
_MIN_AREA = 65536       # 256x256 — processor up-pads tiny images to here
_MAX_AREA = 16_777_216  # 4096x4096 — processor down-fits huge images to here


def _smart_resize(w: int, h: int, cell: int = _CELL,
                  min_area: int = _MIN_AREA, max_area: int = _MAX_AREA) -> tuple[int, int]:
    """The (w, h) grid Qwen3-VL actually rasterises: each side rounded to a multiple of `cell`,
    the total area clamped into [min_area, max_area]. Mirrors qwen-vl-utils smart_resize."""
    hb = max(cell, round(h / cell) * cell)
    wb = max(cell, round(w / cell) * cell)
    if hb * wb > max_area:
        beta = math.sqrt(h * w / max_area)
        hb = math.floor(h / beta / cell) * cell
        wb = math.floor(w / beta / cell) * cell
    elif hb * wb < min_area:
        beta = math.sqrt(min_area / (h * w))
        hb = math.ceil(h * beta / cell) * cell
        wb = math.ceil(w * beta / cell) * cell
    return wb, hb


def est_qwen_tokens(w: int, h: int) -> int:
    """Exact Qwen3-VL image-token count for a w x h image (one token per merged 32x32 cell).
    Verified against the HF Qwen3-VL processor and DeepInfra usage (clog 260626)."""
    wb, hb = _smart_resize(w, h)
    return (wb // _CELL) * (hb // _CELL)


# --- media-URL extraction (pure; fed the already-fetched X JSON / page HTML by source_fetch) ---
def _orig(u: str) -> str:
    """Ask the pbs CDN for native resolution (the bare media_url_https serves a downsized variant)."""
    if "pbs.twimg.com" in u and "name=" not in u:
        return u + ("&" if "?" in u else "?") + "name=orig"
    return u


def media_urls_from_x(x_json: dict) -> list[str]:
    """Full-res media URLs from a syndication tweet payload, in post order: every photo plus each
    video / animated-gif POSTER frame (a still that usually carries the checkable content)."""
    out: list[str] = []
    for m in x_json.get("mediaDetails") or []:
        u = m.get("media_url_https") if isinstance(m, dict) else None
        if isinstance(u, str) and u.startswith("http"):
            out.append(_orig(u))
    return out


_OG_IMAGE = re.compile(
    r'<meta[^>]+(?:property|name)=["\']og:image(?::url|:secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
    re.I)


def og_image_urls(html: str) -> list[str]:
    """The page's og:image (the social-preview card) — the media fallback for non-X (page_meta)
    posts. One entry; empty if absent."""
    m = _OG_IMAGE.search(html or "")
    return [m.group(1).strip()] if m else []


# --- download + downscale + save ---
def download(url: str, timeout: int = 30) -> bytes | None:
    """Raw image bytes (browser UA). None on failure or a non-image response."""
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code == 200 and r.content and "image" in r.headers.get("content-type", "").lower():
            return r.content
    except requests.RequestException:
        pass
    return None


def downscale_for_vlm(img_bytes: bytes, *, max_tokens: int = 1280, max_long_edge: int = 1536):
    """Decode bytes and downscale (LANCZOS) to ~`max_tokens` Qwen3-VL tokens via an area cap, with a
    long-edge legibility guard so wildly wide/tall images don't crush their text. Never upscales.
    Returns (PIL.Image RGB, orig_dims (w,h), saved_dims (w,h)) or None if the bytes won't decode."""
    try:
        im = Image.open(io.BytesIO(img_bytes))
        im.load()
    except Exception:  # Pillow raises a variety of decode errors on truncated / non-image bytes
        return None
    im = im.convert("RGB")
    ow, oh = im.size
    max_area = max_tokens * _CELL * _CELL
    scale = 1.0
    if ow * oh > max_area:
        scale = min(scale, math.sqrt(max_area / (ow * oh)))
    if max(ow, oh) > max_long_edge:
        scale = min(scale, max_long_edge / max(ow, oh))
    if scale < 1.0:
        im = im.resize((max(_CELL, round(ow * scale)), max(_CELL, round(oh * scale))), Image.LANCZOS)
    return im, (ow, oh), im.size


def save_image(img_bytes: bytes, out_path, *, max_tokens: int = 1280, quality: int = 90) -> dict | None:
    """Downscale to the token budget and write high-quality WebP (lossless-ish text). Returns
    {orig_dims, saved_dims, est_tokens, path} or None if the bytes aren't a decodable image."""
    res = downscale_for_vlm(img_bytes, max_tokens=max_tokens)
    if res is None:
        return None
    im, (ow, oh), (sw, sh) = res
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="WEBP", quality=quality, method=6)
    return {"orig_dims": f"{ow}x{oh}", "saved_dims": f"{sw}x{sh}",
            "est_tokens": est_qwen_tokens(sw, sh), "path": str(out_path)}
