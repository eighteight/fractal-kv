"""Does the codec map sentences with SIMILAR MEANING but different
wording to NEARBY 2D vectors?

If yes, the codec could replace or augment LLM embeddings; if no, it
captures sequence structure only and serves as a lossless memory format.

The codec is a positional code and is expected to FAIL this test -- the
test exists to separate structural similarity from semantic similarity.
"""

from __future__ import annotations

import math

from .codec import FractalCodec

PARAPHRASE_GROUPS = [
    [
        "the cat sat on the mat",
        "a feline rested on the rug",
        "the kitty was sitting on the carpet",
    ],
    [
        "the stock market fell sharply today",
        "share prices dropped steeply this afternoon",
        "equities plunged during today's trading",
    ],
    [
        "she cooked dinner for her family",
        "she prepared an evening meal for her relatives",
        "supper was made by her for the household",
    ],
]


def _first_vector(codec: FractalCodec, text: str) -> tuple[float, float]:
    c = codec.compress(text)
    return c.vectors[0] if c.vectors else (0.0, 0.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def run_semantic(codec: FractalCodec) -> dict:
    vecs = [[_first_vector(codec, s) for s in group] for group in PARAPHRASE_GROUPS]

    intra, inter = [], []
    for gi, group in enumerate(vecs):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                intra.append(_dist(group[i], group[j]))
            for gj in range(gi + 1, len(vecs)):
                for other in vecs[gj]:
                    inter.append(_dist(group[i], other))

    mean_intra = sum(intra) / len(intra)
    mean_inter = sum(inter) / len(inter)
    # semantic codecs: paraphrases much closer than unrelated sentences
    separation = mean_inter / mean_intra if mean_intra > 0 else float("inf")
    return {
        "mean_intra_group_dist": round(mean_intra, 4),
        "mean_inter_group_dist": round(mean_inter, 4),
        "separation_ratio": round(separation, 2),
        "semantic": separation > 2.0,
        "verdict": (
            "SEMANTIC: paraphrases cluster -- pursue the embedding-replacement path"
            if separation > 2.0
            else "NOT semantic: captures exact structure only -- pursue the "
            "lossless KV/history-compression path"
        ),
    }
