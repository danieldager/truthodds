"""Recover the lost caption for image-only positives the auditor found claim-less (out-of-context photos
whose tweet text never resolved — deleted/media-only). We DO have the fact-checker's claim_text for all
of them; use it as a synthetic caption AFTER scrubbing fact-checker framing + verdict-leak words
("An old photo authentically shows…", "…is authentic", "A real picture…"). Then re-audit each WITH the
recovered caption (Qwen3-VL-235B) and flag any residual leakage, so the human can judge cleanly.

  uv run python -m eval.scripts.recover_image_captions  ->  eval/data/recovered_captions.parquet
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import polars as pl

from eval.scripts._pool import pooled_checkpointed
from eval.scripts.audit_dataset import _audit

PREFIX = re.compile(r"^\s*(an?|the|this|these|those|old|new)?\s*(old |new |real |genuine |authentic |altered |fake |doctored )*"
                    r"(photo|photos|photograph|photographs|picture|pictures|image|images|video|videos|footage|meme|"
                    r"screenshot|trail camera photographs?|satellite (?:photo|image)|clip)s?\b\s*"
                    r"(purportedly |authentically |genuinely |allegedly )*"
                    r"(showed that |shows that |show that |showed |shows |show |showing |depict |depicts |depicting |"
                    r"reveal |reveals |capture |captures |of )\s*(that\s+)?", re.I)
THESE = re.compile(r"^\s*(these are|this is|here are|here is|here's)\s+"
                   r"(real |genuine |authentic |fake |altered |doctored )*"
                   r"(photos?|pictures?|images?|videos?|screenshots?)\s+(of\s+|that\s+|showing\s+)?", re.I)
FR = re.compile(r"^\s*(cette|ce|cet|une?|des|le|la|les)\s+(photo|image|vid[ée]o|capture|radio|illustration)\w*\s+"
                r"(montre|montrent|d[ée]voile\w*|r[ée]v[èe]le\w*|prouve\w*)\s+(que\s+|qu['’]\s*)?", re.I)
POSTCLAIM = re.compile(r"^\s*(a |the )?(social media |viral )?(post|posts|meme|tweet|claim|users?)\s*"
                       r"(claims?|say|says|alleges?)\s*(that\s+)?", re.I)
TRAIL = re.compile(r"\s*(is|are|was|were)\s+(authentic|real|genuine|fake|altered|digitally altered|manipulated|"
                   r"edited|doctored)\s*\.?\s*$|\s*est-(elle|il)\s+\w+\s*\??\s*$", re.I)
LEAK = re.compile(r"\b(authentic|authentically|authentique|v[ée]ritable|real photos?|real pictures?|genuine|fake|"
                  r"truqu|altered|digitally altered|manipulated|doctored|miscaption|out of context|hors contexte|"
                  r"hoax|debunk)\b", re.I)


def scrub(t: str | None) -> str:
    t = (t or "").strip()
    for rx in (PREFIX, THESE, FR, POSTCLAIM):
        t = rx.sub("", t)
    t = TRAIL.sub("", t)
    t = re.sub(r"\b(authentically|genuinely|purportedly|allegedly)\s+", "", t, flags=re.I)
    t = t.strip()
    return (t[:1].upper() + t[1:]) if t else ""


def main() -> None:
    # map first-image-path -> claim_text
    m = {}
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        d = pl.read_parquet(f)
        if "image_paths" not in d.columns:
            continue
        for r in d.to_dicts():
            ip = (r.get("image_paths") or [None])[0]
            if ip and ip not in m:
                m[ip] = r.get("claim_text") or r.get("raw_claim") or ""
    # the image-only no-claim cases from the audit
    au = pl.read_parquet("eval/data/dataset_audit.parquet").filter(pl.col("recommend").is_not_null()).to_dicts()
    targets = [r for r in au if r["current_label"] == "positive" and not r.get("text") and r.get("has_claim") is False]
    rows = []
    for r in targets:
        ct = m.get(r["image_path"], "")
        rows.append({"uid": r["uid"], "image_path": r["image_path"], "original_claim_text": ct,
                     "recovered_caption": scrub(ct), "leakage": bool(LEAK.search(scrub(ct)))})
    print(f"recovering {len(rows)} image-only captions | leakage-flagged {sum(r['leakage'] for r in rows)}", flush=True)

    out = Path("eval/data/recovered_captions.parquet")
    done = set(pl.read_parquet(out)["uid"].to_list()) if out.exists() else set()
    todo = [i for i, r in enumerate(rows) if r["uid"] not in done and r["recovered_caption"]]
    res: dict = {}

    def flush():
        if res:
            new = pl.DataFrame(list(res.values()))
            comb = new if not out.exists() else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique("uid", keep="last")
            comb.write_parquet(out)

    def work(i):
        r = rows[i]
        try:
            g = _audit({"text": r["recovered_caption"], "image_paths": [r["image_path"]]})
            return i, {**r, "re_in_scope": g.get("in_scope"), "re_has_claim": g.get("has_claim"),
                       "re_topic": g.get("topic"), "re_note": (g.get("note") or "")[:200]}
        except Exception as e:
            return i, {**r, "re_note": f"ERR {e}"[:150]}

    pooled_checkpointed(todo, work, lambda i, p: res.__setitem__(rows[i]["uid"], p), flush, 8, "recover", checkpoint_every=50)
    flush()
    df = pl.read_parquet(out)
    inscope = df.filter(pl.col("re_in_scope") == True).height  # noqa: E712
    claim = df.filter(pl.col("re_has_claim") == True).height  # noqa: E712
    print(f"\nrecovered {df.height} | now in_scope {inscope} | now has_claim {claim} | leakage {df.filter(pl.col('leakage')==True).height}")  # noqa: E712


if __name__ == "__main__":
    main()
