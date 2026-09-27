"""Community Notes dedup layers 1–3 → cn_gold_l3.parquet (Daniel 2026-07-23).

L1 language: keep EN/FR notes (stopword + Latin-script heuristic on the SUMMARY —
   provisional: the definitive language call reruns on POST text after hydration).
L2 scope: drop scam-ad boilerplate — notes citing a curated scam-marker URL, or
   copypasta-template members (>=5 identical normalized texts) matching scam keywords.
   Curated + printed per rule, so the drop is auditable and revisable.
L3 one note per tweet: rank (tier gold first, more citations, longer summary).

Layer 4 (claim-level clustering across tweets: rare-token + shared-citation blocking →
trigram Jaccard → union-find) is a separate pass, NOT built here.

  uv run python -m eval.scripts.build_eval.cn_dedup
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SRC = Path("eval/data/community_notes/cn_gold.parquet")
OUT = Path("eval/data/community_notes/cn_gold_l3.parquet")

_EN = {"the", "and", "this", "that", "with", "from", "have", "not", "was", "are",
       "for", "is", "of", "to", "in", "it", "on", "no", "there", "has", "been"}
_FR = {"le", "la", "les", "des", "une", "un", "est", "pas", "que", "qui", "dans",
       "pour", "sur", "avec", "cette", "ce", "sont", "aux", "d'un", "n'est"}

SCAM_URL_MARKERS = (
    "business.x.com/en/help/ads-policies", "business.twitter.com/en/help/ads-policies",
    "consumer.ftc.gov/articles/what-know-about-cryptocurrency-and-scams",
    "consumer.ftc.gov/articles/how-avoid-scam",
    "bbc.co.uk/news/technology-53759932",          # the 2020 crypto-hack explainer
    "forbes.com/sites/mattnovak/2023/07/23/spacex-crypto-scams",
    "vice.com/en/article/7kxepa",                  # junk-ads explainer
    "all-senmonka.jp/moneyizm", "www3.nhk.or.jp/news/special/net-koukoku",
)
SCAM_KW = re.compile(r"scam|giveaway|gambling|casino|betting|crypto ?(?:wallet|currency)"
                     r"|phishing|drop-?shipping|counterfeit|fake (?:shop|store|ad)"
                     # residual template families surfaced by cn_cluster mega-cluster
                     # inspection (2026-07-23): crypto-reply / impersonation / adult-promo
                     r"|permanently lost|impersonating|stealth promotion|adult-oriented", re.I)


def _norm(s: str) -> str:
    s = re.sub(r"https?://\S+", "", (s or "").lower())
    return re.sub(r"\W+", " ", s).strip()[:400]


def _lang_ok(s: str) -> bool:
    txt = re.sub(r"https?://\S+", "", (s or "").lower())
    letters = re.findall(r"[^\W\d_]", txt)
    if not letters:
        return False
    latin = sum(1 for c in letters if c.isascii() or c in "àâçéèêëîïôùûüœæ")
    if latin / len(letters) < 0.9:
        return False
    toks = set(re.findall(r"[a-zàâçéèêëîïôùûüœæ']+", txt))
    return len(toks & _EN) >= 2 or len(toks & _FR) >= 2


def main():
    df = pl.read_parquet(SRC)
    n0 = len(df)
    print(f"input: {n0} notes / {df['tweetId'].n_unique()} tweets")

    # L1 — language
    df = df.with_columns(pl.col("summary").map_elements(_lang_ok, return_dtype=pl.Boolean)
                         .alias("_lang_ok"))
    l1 = df.filter(pl.col("_lang_ok")).drop("_lang_ok")
    print(f"L1 lang EN/FR: {len(l1)} kept ({n0 - len(l1)} dropped)")

    # L2 — scope (scam boilerplate)
    def scam_url(s):
        return any(m in (s or "") for m in SCAM_URL_MARKERS)
    l1 = l1.with_columns(
        pl.col("summary").map_elements(scam_url, return_dtype=pl.Boolean).alias("_scam_url"),
        pl.col("summary").map_elements(_norm, return_dtype=pl.Utf8).alias("_norm"),
    )
    tpl_counts = Counter(l1["_norm"].to_list())
    templates = {t for t, c in tpl_counts.items() if c >= 5}
    l1 = l1.with_columns(
        (pl.col("_norm").is_in(list(templates))
         & pl.col("_norm").map_elements(lambda t: bool(SCAM_KW.search(t)),
                                        return_dtype=pl.Boolean)).alias("_scam_tpl"))
    by_url = int(l1["_scam_url"].sum())
    by_tpl = int((l1["_scam_tpl"] & ~l1["_scam_url"]).sum())
    l2 = l1.filter(~pl.col("_scam_url") & ~pl.col("_scam_tpl")).drop("_scam_url", "_scam_tpl")
    print(f"L2 scope: {len(l2)} kept (dropped {by_url} by scam-URL, +{by_tpl} by scam-template)")

    # L3 — one note per tweet
    l2 = l2.with_columns(
        (pl.col("tier") == "gold").cast(pl.Int8).alias("_g"),
        pl.col("summary").map_elements(lambda s: len(re.findall(r"https?://", s or "")),
                                       return_dtype=pl.Int64).alias("_ncite"),
        pl.col("summary").str.len_chars().alias("_len"),
    )
    l3 = (l2.sort(["_g", "_ncite", "_len"], descending=True)
            .unique(subset=["tweetId"], keep="first")
            .drop("_g", "_ncite", "_len"))
    print(f"L3 one/tweet: {len(l3)} notes = distinct tweets ({len(l2) - len(l3)} extra notes folded)")

    # residual copypasta going into layer 4
    resid = Counter(l3["_norm"].to_list())
    multi = sum(c for c in resid.values() if c > 1)
    print(f"residual exact-dup mass for layer 4: {multi} notes across "
          f"{sum(1 for c in resid.values() if c > 1)} texts (max {max(resid.values())})")
    l3 = l3.drop("_norm")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    l3.write_parquet(OUT)
    print(f"\nwrote {OUT}: {len(l3)} rows")
    print("tier:", dict(l3.group_by("tier").len().iter_rows()))
    print("by year:", dict(l3.with_columns(pl.col("note_date").dt.year().alias("y"))
          .group_by("y").len().sort("y").iter_rows()))


if __name__ == "__main__":
    main()
