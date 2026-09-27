"""Fetch avatars, photos and video thumbnails for the survey pool posts (Daniel 2026-09-22).

Free X syndication endpoint (same token algorithm as hydrate_cn_sep_posts.py). For each post:
the author's avatar, every photo, the poster frame of every video, and the quoted post's text
and author when the post is a quote. Images are downscaled (avatars 96px, media 360px wide,
JPEG) so the selection page can carry them inline as data URIs under the artifact size cap.

  uv run python -m eval.scripts.claim_sourcing.fetch_pool_media --pool <parquet> --out <json>
$0, no paid calls. Cached per post in <out>.cache/.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from eval.scripts.claim_sourcing.hydrate_cn_sep_posts import UA, x_token

AVATAR_PX, MEDIA_W = 96, 360
lock = threading.Lock()


def _get(url, params=None, tries=5):
    for att in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=20)
        except requests.RequestException:
            time.sleep(min(2 ** att, 16) + random.random()); continue
        if r.status_code in (429,) or r.status_code >= 500:
            time.sleep(min(2 ** att, 16) + random.random()); continue
        return r
    return None


def _jpeg_uri(content: bytes, max_w: int, square: bool = False, q: int = 62) -> str | None:
    try:
        im = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    if square:
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fetch_post(tid: str) -> dict:
    r = _get("https://cdn.syndication.twimg.com/tweet-result", {"id": tid, "token": x_token(tid), "lang": "en"})
    if r is None or r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        return {"post_id": tid, "ok": False}
    j = r.json()
    if j.get("tombstone"):
        return {"post_id": tid, "ok": False}
    u = j.get("user") or {}
    out = {"post_id": tid, "ok": True, "avatar": None, "media": [], "quoted": None,
           "verified": bool(u.get("verified") or u.get("is_blue_verified"))}
    av = (u.get("profile_image_url_https") or "").replace("_normal", "_bigger")
    if av:
        ar = _get(av)
        if ar is not None and ar.status_code == 200:
            out["avatar"] = _jpeg_uri(ar.content, AVATAR_PX, square=True, q=70)
    for m in j.get("mediaDetails") or []:
        kind = m.get("type")
        url = m.get("media_url_https")
        if not url:
            continue
        mr = _get(url + ("?name=small" if kind == "photo" else ""))
        if mr is None or mr.status_code != 200:
            continue
        uri = _jpeg_uri(mr.content, MEDIA_W)
        if uri:
            out["media"].append({"type": "video" if kind in ("video", "animated_gif") else "photo", "src": uri})
    qt = j.get("quoted_tweet")
    if qt:
        qu = qt.get("user") or {}
        out["quoted"] = {"handle": qu.get("screen_name"), "name": qu.get("name"),
                         "text": (qt.get("note_tweet") or {}).get("text") or qt.get("text") or ""}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ids = sorted(set(pd.read_parquet(a.pool).post_id.astype(str)))
    cache = Path(str(a.out) + ".cache"); cache.mkdir(parents=True, exist_ok=True)
    todo = [t for t in ids if not (cache / f"{t}.json").exists()]
    print(f"{len(ids)} posts, {len(todo)} to fetch", flush=True)
    done = [0]; t0 = time.time()

    def work(tid):
        rec = fetch_post(tid)
        (cache / f"{tid}.json").write_text(json.dumps(rec))
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(todo):
                el = time.time() - t0
                print(f"  {done[0]}/{len(todo)}  {done[0] / el:.1f}/s  ETA {el / done[0] * (len(todo) - done[0]):.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, t) for t in todo]):
            f.result()
    recs = {t: json.loads((cache / f"{t}.json").read_text()) for t in ids}
    a.out.write_text(json.dumps(recs))
    n_av = sum(1 for r in recs.values() if r.get("avatar"))
    n_md = sum(len(r.get("media") or []) for r in recs.values())
    n_q = sum(1 for r in recs.values() if r.get("quoted"))
    print(f"ok {sum(1 for r in recs.values() if r['ok'])}  avatars {n_av}  media images {n_md}  quoted {n_q}  "
          f"bytes {a.out.stat().st_size / 1e6:.1f}MB  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
