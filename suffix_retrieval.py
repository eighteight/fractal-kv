#!/usr/bin/env python3
"""Suffix-similarity retrieval inside the compressed representation.

The walk point at position i is dominated by the characters just before i,
with older characters decaying by the contraction ratio r per step:

    |p_i - p_j| <= 2 * r^s   whenever positions i and j share the same
                             last s characters.

So the stored walk points ARE a substring index: encode a query from the
center, nearest-neighbor search over the points, and every position whose
suffix matches the query lands within an r^len(query) ball.  Hits can be
verified and previewed by decoding backward FROM THE MATCHED POINT --
the original text is never consulted.

This script measures:
  1. the decay law itself (distance vs. common-suffix length),
  2. retrieval precision/recall vs. query length, at float64/32/16 index
     precision (16 / 8 / 4 bytes per character of index),
  3. an end-to-end demo: query -> NN hit -> context decoded from the point.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fractal_poc.chaos_walk import ChaosWalkEncoder
from fractal_poc.walk_codec import WalkDecoder
from scale_eval import load_corpus

N_CHARS = 100_000
QUERY_LENS = (3, 4, 6, 8, 12)
QUERIES_PER_LEN = 40
DTYPES = (np.float64, np.float32, np.float16)


def common_suffix_len(text: str, i: int, j: int, cap: int = 30) -> int:
    s = 0
    while s < cap and i - s > 0 and j - s > 0 and text[i - s - 1] == text[j - s - 1]:
        s += 1
    return s


def main() -> None:
    rng = np.random.default_rng(7)
    text = load_corpus()[:N_CHARS].lower()
    enc = ChaosWalkEncoder.decodable("".join(sorted(set(text))))
    dec = WalkDecoder(enc)
    r = enc.ratio
    n = len(text)
    print(f"corpus: {n} chars, alphabet {len(enc.alphabet)}, ratio r={r:.4f}")

    t0 = time.time()
    walk = enc.encode(text)  # walk[i] = point after text[:i]
    points = walk[1:]  # row t -> suffix ending at position t+1
    print(f"walk encoded in {time.time()-t0:.1f}s "
          f"(index sizes: {points.nbytes/2**20:.1f} MB f64, "
          f"{points.nbytes/2/2**20:.1f} MB f32, "
          f"{points.nbytes/4/2**20:.1f} MB f16)")

    # ---- 1. decay law -----------------------------------------------------
    pairs = rng.integers(1, n, size=(60_000, 2))
    suffix_lens, dists = [], []
    for i, j in pairs:
        if i == j:
            continue
        s = common_suffix_len(text, int(i), int(j))
        d = math.hypot(*(walk[i] - walk[j]))
        if d > 0:
            suffix_lens.append(s)
            dists.append(d)
    suffix_lens = np.array(suffix_lens)
    dists = np.array(dists)
    med_by_s = [
        float(np.median(dists[suffix_lens == s]))
        for s in range(suffix_lens.max() + 1)
        if (suffix_lens == s).sum() >= 5
    ]
    print("\ndecay law (median pair distance by common suffix length):")
    for s, d in enumerate(med_by_s):
        print(f"  s={s:2d}: {d:.2e}   (theory 2*r^s = {2*r**s:.2e})")

    # ---- 2. retrieval quality --------------------------------------------
    results = {dt: [] for dt in DTYPES}
    for qlen in QUERY_LENS:
        # sample queries that actually occur
        starts = rng.integers(0, n - qlen, size=QUERIES_PER_LEN)
        queries = [text[s : s + qlen] for s in starts]
        truths = []
        for q in queries:
            occ, at = set(), text.find(q)
            while at != -1:
                occ.add(at + qlen)  # suffix END position
                at = text.find(q, at + 1)
            truths.append(occ)

        for dt in DTYPES:
            P = points.astype(dt).astype(np.float64)
            eps = np.finfo(dt).eps
            tau = 2.0 * r**qlen + 8.0 * eps
            precisions, recalls = [], []
            t0 = time.time()
            for q, occ in zip(queries, truths):
                qp = enc.encode(q)[-1]
                d2 = ((P - qp) ** 2).sum(1)
                pred = set((np.nonzero(d2 <= tau * tau)[0] + 1).tolist())
                tp = len(pred & occ)
                precisions.append(tp / len(pred) if pred else 0.0)
                recalls.append(tp / len(occ))
            dt_ms = (time.time() - t0) / len(queries) * 1e3
            results[dt].append((np.mean(precisions), np.mean(recalls), dt_ms))

    print("\nretrieval (mean over 40 queries/length):")
    print(f"{'qlen':>5} | " + " | ".join(
        f"{np.dtype(dt).name:>22}" for dt in DTYPES))
    print(f"{'':>5} | " + " | ".join(
        f"{'prec':>7}{'recall':>8}{'ms/q':>7}" for _ in DTYPES))
    for qi, qlen in enumerate(QUERY_LENS):
        row = f"{qlen:>5} | "
        row += " | ".join(
            f"{results[dt][qi][0]:>7.2f}{results[dt][qi][1]:>8.2f}"
            f"{results[dt][qi][2]:>7.1f}"
            for dt in DTYPES
        )
        print(row)

    # ---- 3. end-to-end demo: search + context, no original text ----------
    query = "the time traveller"[: min(18, n)]
    if query in text:
        qp = enc.encode(query)[-1]
        d2 = ((points - qp) ** 2).sum(1)
        hit = int(d2.argmin()) + 1
        # float64 inversion is reliable for ~44/log2(1/r) chars per hop, so
        # decode the 40-char context in short hops through intermediate
        # index points (all of which the index already stores)
        hop = max(1, int(40 / math.log2(1.0 / r)))
        pieces = []
        end = hit
        while end > max(0, hit - 40):
            start = max(0, end - hop, hit - 40)
            piece = dec.decode_backward(
                tuple(walk[end]), end - start, tuple(walk[start])
            )
            pieces.append(piece if piece is not None else "?" * (end - start))
            end = start
        ctx = "".join(reversed(pieces))
        print(f"\ndemo query: {query!r}")
        print(f"  nearest point at position {hit}, "
              f"distance {math.sqrt(d2[hit-1]):.2e}")
        print(f"  context decoded from the matched point: ...{ctx!r}")
        print(f"  ground truth from original text:        "
              f"...{text[hit-40:hit]!r}")

    # ---- plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(
        suffix_lens + rng.uniform(-0.2, 0.2, len(suffix_lens)),
        np.log10(dists), s=2, alpha=0.15, color="#1f4e79",
    )
    ss = np.arange(0, len(med_by_s) if med_by_s else 1)
    axes[0].plot(ss, np.log10(2 * r**ss.astype(float)), "r--",
                 label=r"theory: $2 r^s$")
    axes[0].set_xlabel("common suffix length s (chars)")
    axes[0].set_ylabel("log10 distance between walk points")
    axes[0].set_title("Decay law: geometry encodes suffix similarity")
    axes[0].legend()
    for dt, style in zip(DTYPES, ("o-", "s-", "^-")):
        axes[1].plot(QUERY_LENS, [results[dt][i][1] for i in range(len(QUERY_LENS))],
                     style, label=f"recall {np.dtype(dt).name} "
                     f"({np.dtype(dt).itemsize*2} B/char)")
        axes[1].plot(QUERY_LENS, [results[dt][i][0] for i in range(len(QUERY_LENS))],
                     style, alpha=0.35)
    axes[1].set_xlabel("query length (chars)")
    axes[1].set_ylabel("recall (solid) / precision (faded)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Retrieval vs query length and index precision")
    axes[1].legend()
    fig.suptitle("Substring search executed directly on the compressed walk points")
    fig.savefig("suffix_retrieval.png", dpi=110, bbox_inches="tight")
    print("\nsaved suffix_retrieval.png")


if __name__ == "__main__":
    main()
