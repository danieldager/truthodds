"""Handoff: normalize output -> verify input. The committed builder for what was hand-assembled
on 2026-07-11 (dev500_claims.parquet had no reproducible provenance — snapshot finding).

Joins the normalizer's per-post claims (payload HTML from normalize_tweet_claims.py) with the
locked posts parquet and emits THREE artifacts:

  1. the CLAIMS parquet (verify input) — one row per claim with the full post payload:
     ids (post_id, claim_id=post_id:n), claim fields (claim, type, checkworthy + any extra
     per-claim flags the extractor adds later, passed through automatically), post payload
     (post_text, handle, domain, url, created_at, lang, is_quote), media (image_urls,
     video_urls, n_images, n_videos — verify runs text-only today, but media context must
     travel for the serialization step and analysis), engagement (like/retweet/reply/quote/
     view counts, followers when present), and analysis strata (ng_score, lean, register,
     cell — NOT verifier inputs, never feed them to prompts).
  2. the POSTS MANIFEST parquet — one row per post in the locked set with n_claims,
     n_checkworthy, no_claim_reason and a status in {verify, skip_no_claims,
     skip_none_checkworthy}: the per-source DENOMINATORS live here, next to the claims,
     instead of being joined by convention across three files.
  3. a PROVENANCE json — input paths + sha1, models used (pass them in), row counts,
     git commit, timestamp. The "which config built this parquet" question dies here.

`load_verify_posts(claims_parquet)` is THE handoff contract both sides import: it groups the
claims parquet into the post dicts `verify_tweet_claims.verify_post` consumes
({id, url, handle, domain, date, text, claims: [{c, t, cw}]}).

  cd src && uv run python eval/scripts/claim_sourcing/build_verify_input.py \
      --normalized eval/data/survey_claims/dev500_normalized.html \
      --posts eval/data/survey_claims/survey_dev_500.parquet \
      -o eval/data/survey_claims/dev500_claims.parquet \
      --extract-model deepseek-ai/DeepSeek-V4-Flash --normalize-model deepseek-ai/DeepSeek-V4-Flash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3]

# THE CHECKWORTHY GATE (v5): a CODE rule, not an LLM judgment — tunable here without
# re-running any LLM pass. Daniel's 2026-07-14 redesign: FALSIFIABLE is the gold-standard
# criterion for measuring misinformation per source; topics and trivial no longer gate
# (both are post-hoc analysis dimensions — per-topic veracity, weighty-only rates; the
# 2026-07-14 Opus audits measured trivial's recall too low (~21-24% junk still passing)
# to carry a load-bearing role, and hand-selected tweets make the weight filter moot for
# the research measurement). unresolved still gates as a MEASURABILITY constraint (verify
# cannot search a referent it cannot pin down — report the rate per source instead,
# vagueness is itself an outlet signal). Own-voice opinion is no longer removed by the
# audit pass: it arrives here as a kept claim with falsifiable=False.
INCLUDE_TOPICS = None  # off; to re-enable set e.g. {"politics", "crime", "health",
#                        "education", "business", "science", "technology", "disaster"}
GATE_ON_TRIVIAL = True  # ON (Daniel 2026-07-19, reverses 2026-07-14): trivial claims are not checkworthy

# v4 chain (2026-07-20): pass 2 assigns ONE category per claim; the gate is one line —
# checkworthy = category=="checkable". Soft topics are excluded from the VERIFY RUN only
# (Daniel 2026-07-20): the claim keeps its checkworthy flag for per-source stats, but
# verify_eligible=False keeps it out of the loop.
EXCLUDE_TOPICS = {"lifestyle", "entertainment", "sports"}

def _gate(topic, claim, trivial, unresolved, falsifiable=True):
    """checkworthy = falsifiable AND NOT unresolved (AND NOT trivial / include-topic, when
    those toggles are configured). Pure flag algebra — a disorder-keyword override existed
    briefly and was removed (Daniel 2026-07-14: inelegant)."""
    if not falsifiable or unresolved:
        return False
    if GATE_ON_TRIVIAL and trivial:
        return False
    return INCLUDE_TOPICS is None or topic in INCLUDE_TOPICS

# post-payload columns copied onto every claim row, in output order (missing ones are skipped
# with a warning rather than crashing — older posts parquets lack e.g. followers)
POST_COLS = ["handle", "cell", "domain", "url", "created_at", "lang", "is_quote",
             "quotes", "quoted_handle", "quoted_text", "retweeted_by", "operation", "account",
             "captured_at", "is_reply", "reply_to", "conversation_id",
             "image_urls", "video_urls", "n_images", "n_videos",
             "like_count", "retweet_count", "reply_count", "quote_count", "view_count",
             "followers", "ng_score", "lean", "register"]

# Non-LLM type repair (locked design: anonymous/hedged sourcing -> assertion; the LLM flips
# on this boundary run-to-run — MSNOW "sources say" measured bistable). An attribution whose
# only sourcing is anonymous ("sources say", "officials confirmed") with NO named speaker is
# retyped to assertion: the content is what gets checked.
_ANON_SRC = re.compile(r"\b(sources?|officials?|insiders?|witnesses|authorities|reports?)\s+"
                       r"(say|says|said|claims?|claimed|told|confirm(?:ed)?|reveal(?:ed)?)\b", re.I)
_NAMED_SPEAKER = re.compile(r"\b[A-Z][\w.'-]+\s+(?:said|says|claim(?:s|ed)?|told|wrote|writes|"
                            r"argu(?:es|ed)|stated|announc(?:es|ed)|deni(?:es|ed)|confirm(?:s|ed)|"
                            r"warn(?:s|ed)|urg(?:es|ed)|accus(?:es|ed)|testif(?:ies|ied)|report(?:s|ed))\b")


# Amplification detection is STRUCTURAL (Daniel's C-section ruling 2026-07-14): a post that is
# a quotation + a name/@handle with NO reporting verb of the outlet's own OUTSIDE the quotes is
# the outlet speaking through the quote -> its attribution claims are retyped to ASSERTION
# (typing them "X said Y" would let false content verify as a true saying). "argues X" outside
# quotes IS a reporting frame -> stays attribution. Prompt wording see-sawed on this; post
# structure does not.
_QUOTE_SPANS = re.compile(r'[“"][^”"]{10,}[”"]')


def _is_amplification_post(post_text: str) -> bool:
    t = post_text or ""
    if not _QUOTE_SPANS.search(t):
        return False
    outside = _QUOTE_SPANS.sub(" ", t)
    return not _SPEECH_VERB.search(outside)


# Anonymous-collective attributees ("Ukrainian army vets and US volunteers said", "fans claim"):
# no specific person or org a fact-checker could ask -> never checkworthy (Daniel's Grayzone +
# RedState rulings). The claim stays in the record.
_COLL_NOUNS = (r"(vets?|veterans|volunteers|residents|fans|supporters|users|workers|"
               r"soldiers|locals|critics|activists|analysts|experts|witnesses|protesters|"
               r"insiders|people)")
_COLLECTIVE = re.compile(r"\b" + _COLL_NOUNS + r"\b[^.]{0,30}?\b(say|says|said|claims?|claimed|"
                         r"told|tell|allege[sd]?)\b", re.I)
_COLLECTIVE_REV = re.compile(r"\baccording to\b[^.]{0,30}?\b" + _COLL_NOUNS + r"\b", re.I)


def _repair_type(claim, typ, post_text=""):
    if typ != "attribution":
        return typ, False
    if _ANON_SRC.search(claim or "") and not _NAMED_SPEAKER.search(claim or ""):
        return "assertion", True
    if _is_amplification_post(post_text) and _NAMED_SPEAKER.search(claim or ""):
        return "assertion", True
    return typ, False


# Private inner states of unnamed individuals in human-interest stories ("the boy genuinely
# believed...") are unverifiable and never checkworthy (Daniel's EpochTimes ruling); the LLM
# controversial flag flips on these run-to-run — this rule does not.
_INNER_STATE = re.compile(r"\b(believed?|believes|felt|feels|feared?|fears|hoped?|hopes|"
                          r"thought|thinks|wanted?|wants|dreamed|loves?|loved)\b", re.I)
_GENERIC_PERSON = re.compile(r"\b(a|an|the)\s+(\d+[- ]year[- ]old\s+)?(boy|girl|man|woman|child|"
                             r"kid|mother|father|mom|dad|teen\w*|student|couple|family)\b", re.I)


def _private_inner_state(claim):
    return bool(_INNER_STATE.search(claim or "")) and bool(_GENERIC_PERSON.search(claim or ""))


def _anonymous_collective(claim, typ):
    if typ != "attribution":
        return False
    if _COLLECTIVE_REV.search(claim or "") and not _NAMED_SPEAKER.search(claim or ""):
        return True
    if not _COLLECTIVE.search(claim or ""):
        return False
    # a sentence-initial collective noun ("Analysts said...") capitalizes like a name — a
    # named-speaker match that is itself the collective phrase does not veto the flag
    m = _NAMED_SPEAKER.search(claim or "")
    return not m or bool(_COLLECTIVE.match(m.group(0)))


_SPEECH_VERB = re.compile(
    r"\b(said|says|say|saying|claims?|claimed|according to|told|tells|wrote|writes|stated|"
    r"urged|accused|called|announced|asked|testified|argued|warned|noted|described|reported|"
    r"revealed|declared|confirmed|denied|insisted|explained)\b", re.I)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_normalized(path: Path) -> list[dict]:
    """Posts from the normalize payload (HTML with an embedded JSON <script>), or a flat
    parquet from normalize_tweet_claims --parquet (fewer fields: no no_claim_reason)."""
    if path.suffix == ".html":
        m = re.search(r'<script type="application/json" id="data">(.*?)</script>',
                      path.read_text(), re.S)
        if not m:
            raise SystemExit(f"no JSON payload found in {path}")
        return json.loads(m.group(1))["posts"]
    df = pd.read_parquet(path)
    posts = []
    for url, g in df.groupby("url", sort=False):
        posts.append({"url": url, "handle": g.iloc[0].get("handle"),
                      "text": g.iloc[0].get("post_text"), "nr": None,
                      "claims": [{"c": r.claim, "t": r.type, "cw": bool(r.checkworthy)}
                                 for r in g.itertuples()]})
    return posts


def load_verify_posts(claims_parquet: str | Path) -> list[dict]:
    """THE handoff contract: claims parquet -> the post dicts verify_post consumes.
    One dict per post; ledger = ALL its claims with their cw flags (verify_post itself
    filters to cw==True). Strata columns are deliberately NOT included — they are
    analysis-only and must never reach a prompt."""
    df = pd.read_parquet(claims_parquet)
    if "removed_reason" in df.columns:      # v3: removed claims never reach the verifier
        df = df[df["removed_reason"].isna()]
    # v4 parquets: cw for the LOOP is verify_eligible (checkworthy minus the excluded
    # topics); the checkworthy column itself stays the per-source stats flag
    has_elig = "verify_eligible" in df.columns
    has_v4 = "category" in df.columns
    posts = []
    for pid, g in df.groupby("post_id", sort=False):
        r0 = g.iloc[0]
        claims = []
        for r in g.itertuples():
            c = {"c": r.claim, "t": r.type,
                 "cw": bool(r.verify_eligible) if has_elig else bool(r.checkworthy),
                 "claim_id": r.claim_id}
            if has_v4:
                c["cat"] = r.category
                c["g"] = int(r.group) if pd.notna(r.group) else None
            claims.append(c)
        posts.append({"id": str(pid), "post_id": str(pid), "url": r0.get("url"),
                      "handle": r0.get("handle"), "domain": r0.get("domain"),
                      "date": str(r0.get("created_at") or "")[:10] or None,
                      "text": r0.get("post_text"),
                      "claims": claims})
    return posts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normalized", required=True, help="normalize payload HTML (preferred) or parquet")
    ap.add_argument("--posts", required=True, help="locked posts parquet (the denominator set)")
    ap.add_argument("-o", "--output", required=True, help="claims parquet (the verify input)")
    ap.add_argument("--manifest-out", default="", help="posts manifest parquet (default: <output>_posts_manifest.parquet)")
    ap.add_argument("--extract-model", default="UNRECORDED", help="pass-1 model id, recorded in provenance")
    ap.add_argument("--normalize-model", default="UNRECORDED", help="pass-2 model id, recorded in provenance")
    args = ap.parse_args()

    norm_path, posts_path, out_path = Path(args.normalized), Path(args.posts), Path(args.output)
    manifest_path = Path(args.manifest_out) if args.manifest_out else \
        out_path.with_name(out_path.stem + "_posts_manifest.parquet")

    nposts = _load_normalized(norm_path)
    posts = pd.read_parquet(posts_path)
    posts["post_id"] = posts["post_id"].astype(str)
    by_url = {r["url"]: r for r in posts.to_dict("records")}
    missing_cols = [c for c in POST_COLS if c not in posts.columns]
    if missing_cols:
        print(f"WARN: posts parquet lacks columns {missing_cols} — skipped")
    cols = [c for c in POST_COLS if c in posts.columns]

    rows, matched_urls, type_repairs = [], set(), []
    for p in nposts:
        meta = by_url.get(p.get("url"))
        if meta is None:
            print(f"WARN: normalized post not in the locked set, skipped: {p.get('url')}")
            continue
        matched_urls.add(p["url"])
        for k, c in enumerate(p.get("claims") or [], 1):
            # cw: legacy payloads carry an LLM "cw" flag; v3 payloads carry a topic and the
            # gate is the CODE rule above
            typ, repaired = _repair_type(c.get("c"), c.get("t"),
                                         p.get("text") or meta.get("text") or "")
            if repaired:
                type_repairs.append(c.get("c"))
            if c.get("cat") is not None:
                # v4 chain: ONE category per claim, gate = category=="checkable"; the old
                # flag algebra and its code assists do not apply (pass 2 owns the judgment).
                cw = c.get("cat") == "checkable"
                base = {"post_id": meta["post_id"], "claim_id": f"{meta['post_id']}:{k}",
                        "claim": c.get("c"), "type": typ, "checkworthy": cw,
                        "verify_eligible": cw and (c.get("topic") or "other") not in EXCLUDE_TOPICS,
                        "category": c.get("cat"),
                        "group": int(c["g"]) if isinstance(c.get("g"), (int, float)) else None,
                        "removed_reason": None}
            else:
                # flags: LLM (both passes, pass-2 value final) OR-ed with the code assists —
                # the record shows exactly why a claim is gated out
                triv = bool(c.get("triv")) or _private_inner_state(c.get("c"))
                unres = bool(c.get("unres")) or _anonymous_collective(c.get("c"), typ)
                if repaired:
                    # locked design: anonymous/hedged SOURCING ("sources say", "an official says")
                    # is an assertion whose CONTENT gets verified — it is not "unresolved". The
                    # LLM flag over-fires here (Opus panel 2026-07-14: Gaza-official case); the
                    # retype and the flag must agree.
                    unres = False
                fals = bool(c.get("fals", True))
                cw = bool(c.get("cw")) if "cw" in c else _gate(c.get("topic"), c.get("c"), triv, unres, fals)
                base = {"post_id": meta["post_id"], "claim_id": f"{meta['post_id']}:{k}",
                        "claim": c.get("c"), "type": typ,
                        "checkworthy": cw, "falsifiable": fals, "trivial": triv, "unresolved": unres,
                        "removed_reason": None}
            # pass through any EXTRA per-claim fields (topic, future flags) without a schema change
            for extra_k, extra_v in c.items():
                if extra_k not in ("c", "t", "cw", "ctrl", "triv", "unres", "fals", "cat", "g"):
                    base[extra_k] = extra_v
            base["post_text"] = p.get("text") or meta.get("text")
            for col in cols:
                base[col] = meta.get(col)
            rows.append(base)
        for j, r in enumerate(p.get("removed") or [], 1):
            # the removal ledger travels with the claims: never checkworthy, never verified,
            # but countable per source (opinion rates for #15) and auditable
            base = {"post_id": meta["post_id"], "claim_id": f"{meta['post_id']}:r{j}",
                    "claim": r.get("c"), "type": None, "checkworthy": False,
                    "removed_reason": r.get("reason"), "topic": None,
                    "post_text": p.get("text") or meta.get("text")}
            for col in cols:
                base[col] = meta.get(col)
            rows.append(base)
    claims = pd.DataFrame(rows)

    # posts manifest: EVERY post in the locked set gets a row — the denominators
    man_rows = []
    nr_by_url = {p["url"]: p.get("nr") for p in nposts}
    kept_df = claims[claims["removed_reason"].isna()] if len(claims) else claims
    if "verify_eligible" not in kept_df.columns:
        kept_df = kept_df.assign(verify_eligible=kept_df.get("checkworthy", False))
    claims_by_pid = kept_df.groupby("post_id").agg(
        n_claims=("claim", "size"), n_checkworthy=("checkworthy", "sum"),
        n_verify_eligible=("verify_eligible", "sum")) \
        if len(kept_df) else pd.DataFrame(columns=["n_claims", "n_checkworthy", "n_verify_eligible"])
    removed_by_pid = claims[~claims["removed_reason"].isna()].groupby("post_id").size() \
        if len(claims) else pd.Series(dtype=int)
    for r in posts.to_dict("records"):
        pid = r["post_id"]
        n_c = int(claims_by_pid.loc[pid, "n_claims"]) if pid in claims_by_pid.index else 0
        n_cw = int(claims_by_pid.loc[pid, "n_checkworthy"]) if pid in claims_by_pid.index else 0
        n_ve = int(claims_by_pid.loc[pid, "n_verify_eligible"]) if pid in claims_by_pid.index else 0
        status = "verify" if n_ve else ("skip_none_checkworthy" if n_c else "skip_no_claims")
        man_rows.append({"post_id": pid, "handle": r.get("handle"), "domain": r.get("domain"),
                         "ng_score": r.get("ng_score"), "lean": r.get("lean"),
                         "register": r.get("register"), "n_claims": n_c,
                         "n_checkworthy": n_cw, "n_verify_eligible": n_ve,
                         "n_removed": int(removed_by_pid.get(pid, 0)),
                         "status": status,
                         "no_claim_reason": nr_by_url.get(r.get("url"))})
    manifest = pd.DataFrame(man_rows)

    # light validation — loud, never fatal
    bad_type = claims[~claims["type"].isin(["assertion", "attribution"])]
    if len(bad_type):
        print(f"WARN: {len(bad_type)} claims with unexpected type: {bad_type['type'].unique()}")
    att = claims[(claims["type"] == "attribution") & claims["removed_reason"].isna()]
    unbound = att[~att["claim"].str.contains(_SPEECH_VERB, na=False)] if len(att) else att
    if len(att):
        print(f"NOTE: {len(unbound)}/{len(att)} attribution claims lack an 'X said' binding "
              f"in the claim text (known extraction issue — verify judges the claim STRING)")
    if type_repairs:
        print(f"NOTE: {len(type_repairs)} anonymous-sourcing attribution(s) retyped to assertion (code rule)")
    dup = claims["claim_id"].duplicated().sum()
    if dup:
        print(f"WARN: {dup} duplicate claim_ids")

    claims.to_parquet(out_path, index=False)
    manifest.to_parquet(manifest_path, index=False)
    try:
        git_rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=SRC).stdout.strip()
    except Exception:
        git_rev = "unknown"
    prov = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_rev": git_rev,
            "extract_model": args.extract_model, "normalize_model": args.normalize_model,
            "inputs": {str(norm_path): _sha1(norm_path), str(posts_path): _sha1(posts_path)},
            "outputs": {str(out_path): len(claims), str(manifest_path): len(manifest)},
            "posts_matched": len(matched_urls), "posts_total": len(posts),
            "checkworthy": int(claims["checkworthy"].sum()) if len(claims) else 0,
            "status_counts": manifest["status"].value_counts().to_dict()}
    prov_path = out_path.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=1))
    print(f"claims:   {len(claims)} rows ({prov['checkworthy']} checkworthy) -> {out_path}")
    print(f"manifest: {len(manifest)} posts {prov['status_counts']} -> {manifest_path}")
    print(f"provenance -> {prov_path}")


if __name__ == "__main__":
    main()
