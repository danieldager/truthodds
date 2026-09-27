"""Topic anchors for the Stage 1 embedding scope-classifier.

Each topic LABEL is represented by an *anchor* = the L2-normalized mean of the
normalized embeddings of a handful of short descriptor phrases. Posts are
classified by ``argmax cosine(post, anchor)``; a post is ``in_scope_embed``
when its top topic is in :data:`IN_SCOPE` and its top cosine clears :data:`TAU`.

The descriptor phrases mix English with French / Spanish / Arabic, because the
embedding model (``paraphrase-multilingual-MiniLM-L12-v2``) co-embeds
translations into nearby vectors — averaging a few translations into each
anchor sharpens it for the non-English ~60% of the corpus without a separate
per-language taxonomy.

This file is meant to be edited: add/remove phrases, retune :data:`TAU`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.embedding import embed  # noqa: E402  (shared multilingual model)

# Top cosine must clear this for in_scope_embed=True. Starts recall-favoring
# (the LLM scope stage is the real gate); calibrate on the labeled 50-set.
TAU = 0.0

# Posts shorter than this (essentially empty / media-only) are not embedded.
MIN_CHARS = 5

# --- In-scope topics: where consequential misinformation lives -------------
# Phrases synthesized from a 3-proposer + synthesizer design panel, with a
# discriminability pass that sharpens confusable pairs (public_figure kept to
# attributed speech-acts so it does not collide with politics_elections or
# entertainment_celebrity; crime stays on policing/judicial cues so it does not
# bleed into conflict_war; identity_public uses debate/narrative framing).
IN_SCOPE_ANCHORS: dict[str, list[str]] = {
    "politics_elections": [
        "election results",
        "presidential campaign",
        "parliament vote",
        "government policy reform",
        "election fraud claims",
        "résultats des élections",
        "fraude electoral",
        "الانتخابات البرلمانية",
    ],
    "health_medicine": [
        "vaccine safety",
        "public health guidance",
        "disease outbreak treatment",
        "miracle cure claim",
        "campagne de vaccination",
        "salud pública",
        "اللقاحات والأدوية",
        "أضرار اللقاح",
    ],
    "science_climate": [
        "climate change report",
        "carbon emissions data",
        "renewable energy transition",
        "scientific study findings",
        "réchauffement climatique",
        "découverte scientifique",
        "cambio climático",
        "تغير المناخ",
    ],
    "economics": [
        "inflation and prices",
        "unemployment figures",
        "interest rate decision",
        "stock market crash",
        "hausse des prix",
        "comercio y aranceles",
        "crisis económica",
        "التضخم والأسعار",
    ],
    "crime": [
        "violent crime arrest",
        "police investigation",
        "court trial verdict",
        "drug trafficking bust",
        "fusillade meurtrière",
        "delincuencia y violencia",
        "اعتقال الشرطة",
        "جريمة قتل",
    ],
    "conflict_war": [
        "military offensive",
        "ceasefire negotiations",
        "war in Gaza",
        "armed conflict casualties",
        "guerre en Ukraine",
        "frappes aériennes",
        "alto el fuego",
        "الحرب في غزة",
    ],
    "crisis_disaster": [
        "earthquake death toll",
        "wildfire evacuation",
        "deadly flooding",
        "emergency rescue operation",
        "catastrophe naturelle",
        "crise humanitaire",
        "desastre natural",
        "كارثة طبيعية",
    ],
    "public_figure": [
        "president's statement",
        "minister resignation",
        "official accused of corruption",
        "déclaration du président",
        "ministre accusé de corruption",
        "líder político declaró",
        "تصريح المسؤول",
        "تصريح الرئيس",
    ],
    "identity_public": [
        "immigration debate",
        "racial discrimination",
        "gender identity controversy",
        "religious freedom dispute",
        "great replacement narrative",
        "débat sur l'immigration",
        "identité nationale",
        "الهوية الدينية",
    ],
}

# --- Out-of-scope topics: rarely consequential misinformation --------------
OUT_SCOPE_ANCHORS: dict[str, list[str]] = {
    "sports": [
        "match final score",
        "league championship",
        "player transfer deal",
        "olympic medal race",
        "résultat du match",
        "transfert de joueur",
        "partido de fútbol",
        "مباراة كرة القدم",
    ],
    "entertainment_celebrity": [
        "new movie release",
        "celebrity breakup gossip",
        "music album drop",
        "streaming TV series",
        "potins de célébrités",
        "concert de musique",
        "estreno de película",
        "أخبار المشاهير",
    ],
    "personal_lifestyle": [
        "my morning coffee routine",
        "weekend travel trip",
        "favorite recipe dinner",
        "relationship advice",
        "ma vie quotidienne",
        "conseils de voyage",
        "vida cotidiana",
        "وصفة طعام",
    ],
    "promotional_ads": [
        "limited time discount",
        "giveaway enter now",
        "shop now link",
        "crypto investment opportunity",
        "code promo réduction",
        "offre spéciale achat",
        "oferta por tiempo limitado",
        "عرض خاص خصم",
    ],
    "social_format": [
        "good morning everyone",
        "follow me back",
        "like and retweet",
        "happy birthday wishes",
        "bonjour à tous",
        "abonnez-vous à ma chaîne",
        "buenos días a todos",
        "صباح الخير",
    ],
}

ANCHORS: dict[str, list[str]] = {**IN_SCOPE_ANCHORS, **OUT_SCOPE_ANCHORS}
IN_SCOPE: tuple[str, ...] = tuple(IN_SCOPE_ANCHORS)
OUT_SCOPE: tuple[str, ...] = tuple(OUT_SCOPE_ANCHORS)
TOPIC_LABELS: list[str] = list(ANCHORS)


def build_anchor_matrix() -> tuple[list[str], np.ndarray]:
    """Return ``(labels, matrix)`` where ``matrix[i]`` is the L2-normalized mean
    embedding of ``ANCHORS[labels[i]]``. Shape ``(n_labels, EMBEDDING_DIM)``.

    Cosine similarity against a post embedding reduces to ``matrix @ post_vec``
    (both sides are unit vectors).
    """
    labels = TOPIC_LABELS
    rows = []
    for label in labels:
        vecs = np.stack([embed(p) for p in ANCHORS[label]])  # each L2-normalized
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        rows.append(mean / norm if norm > 0 else mean)
    return labels, np.stack(rows).astype(np.float32)


if __name__ == "__main__":
    # Quick discriminability check: anchor-vs-anchor cosine matrix.
    labels, mat = build_anchor_matrix()
    sim = mat @ mat.T
    print(f"{len(labels)} anchors, dim {mat.shape[1]}")
    print("\nmost confusable label pairs (cosine):")
    pairs = [
        (labels[i], labels[j], float(sim[i, j]))
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    for a, b, c in sorted(pairs, key=lambda t: t[2], reverse=True)[:10]:
        print(f"  {c:.3f}  {a:<24} ~ {b}")
