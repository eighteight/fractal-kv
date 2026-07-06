#!/usr/bin/env python3
"""GridWalkCodec: chaos walk with GRID vertices instead of ring vertices.

The ring codec places its N vertices on a circle, so the letter cells form
a necklace and the polygon interior is wasted (~32% area coverage at the
kissing ratio).  Here the vertices fill the unit square as a k x k grid
with contraction r = 1/k: the cells TILE the square exactly, so the
geometric packing loss vanishes.  With k a power of two the arithmetic is
dyadic, hence exact in float64, and backward decoding is a deterministic
digit extraction (no search).

It is still a contractive chaos walk (p <- v + r(p - v) toward grid
vertices), so the suffix decay law |p_i - p_j| <= diam * r^s survives,
and with it the in-place retrieval property.

This script measures grid vs ring on three axes, same corpus:
  A. archive rate      (bits/char of the span-anchored codec, lossless)
  B. decay law         (median pair distance vs common suffix length)
  C. retrieval quality (precision/recall of substring search on the index)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fractal_poc.chaos_walk import ChaosWalkEncoder, sierpinski_ratio  # noqa: E402
from fractal_poc.walk_codec import WalkCodec, precision_span  # noqa: E402
from weighted_chaos import load_corpus  # noqa: E402


# ----------------------------------------------------------------------
# Grid walk
# ----------------------------------------------------------------------
class GridWalkEncoder:
    """Vertices on a k x k grid, contraction 1/k (k a power of two)."""

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        n = len(alphabet)
        k = 2
        while k * k < n:
            k *= 2
        self.k = k
        self.index = {c: i for i, c in enumerate(alphabet)}
        # digit pair per symbol: (gx, gy) in {0..k-1}^2
        self.gx = {c: self.index[c] % k for c in alphabet}
        self.gy = {c: self.index[c] // k for c in alphabet}
        self.ratio = 1.0 / k

    def walk(self, text: str) -> np.ndarray:
        """Full-history walk (retrieval index): row t = point after t chars."""
        pts = np.empty((len(text) + 1, 2))
        x = y = 0.5
        pts[0] = (x, y)
        k = self.k
        for t, c in enumerate(text, 1):
            x = (x + self.gx[c]) / k
            y = (y + self.gy[c]) / k
            pts[t] = (x, y)
        return pts


class GridWalkCodec:
    """Span-anchored archive: each span restarts at (.5,.5); one point/span.

    Span m = floor(51 / log2 k) keeps every coordinate an exact dyadic with
    <= 52 mantissa bits, so decoding (repeated multiply-by-k and floor) is
    EXACT: losslessness is arithmetic, not statistical.
    """

    name = "grid-walk (k x k vertices, tiling cells)"

    def __init__(self, alphabet: str):
        self.enc = GridWalkEncoder(alphabet)
        self.span = int(51 // math.log2(self.enc.k))

    def compress(self, text: str):
        e, m, k = self.enc, self.span, self.enc.k
        pts = []
        for s in range(0, len(text), m):
            x = y = 0.5
            for c in text[s:s + m]:
                x = (x + e.gx[c]) / k
                y = (y + e.gy[c]) / k
            pts.append((x, y))
        return pts, len(text)

    def decompress(self, compressed) -> str:
        pts, n = compressed
        e, m, k = self.enc, self.span, self.enc.k
        out = []
        for i, (x, y) in enumerate(pts):
            cnt = min(m, n - i * m)
            span_syms = []
            for _ in range(cnt):
                x *= k; gx = int(x); x -= gx
                y *= k; gy = int(y); y -= gy
                span_syms.append(e.alphabet[gy * k + gx])
            assert (x, y) == (0.5, 0.5), "anchor mismatch: arithmetic not exact"
            span_syms.reverse()  # backward extraction yields last char first
            out.append("".join(span_syms))
        return "".join(out)

    def memory_bytes(self, compressed) -> int:
        return len(compressed[0]) * 16


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def axis_A_rate(text: str, ring: WalkCodec, grid: GridWalkCodec) -> None:
    print("=" * 72)
    print("A. ARCHIVE RATE (span-anchored codec, lossless)")
    print("=" * 72)
    n = len(text)
    N = len(grid.enc.alphabet)
    info = math.log2(N)

    t0 = time.time()
    gc = grid.compress(text)
    ok_g = grid.decompress(gc) == text
    tg = time.time() - t0
    g_bpc = grid.memory_bytes(gc) * 8 / n

    t0 = time.time()
    rc = ring.compress(text)
    r_bpc = rc.memory_bytes() * 8 / n
    sl = text[:20000]
    ok_r = ring.decompress(ring.compress(sl)) == sl  # slice: ring decode is slow
    tr = time.time() - t0

    r = sierpinski_ratio(N)
    print(f"corpus {n} chars, alphabet {N}, info = {info:.2f} bits/char")
    print(f"{'':24}{'ring (circle)':>16}{'grid (k=%d)' % grid.enc.k:>16}")
    print(f"{'contraction r':24}{r:>16.4f}{grid.enc.ratio:>16.4f}")
    print(f"{'span (chars/point)':24}{precision_span(r):>16}{grid.span:>16}")
    print(f"{'stored bits/char':24}{r_bpc:>16.2f}{g_bpc:>16.2f}")
    print(f"{'geometric efficiency':24}"
          f"{info / (2 * math.log2(1 / r)):>15.1%}"
          f"{info / (2 * math.log2(grid.enc.k)):>15.1%}")
    print(f"{'lossless round-trip':24}{str(ok_r) + ' (20K slice)':>16}"
          f"{str(ok_g) + ' (full)':>16}")
    print(f"{'codec time (s)':24}{tr:>16.1f}{tg:>16.1f}")
    print(f"\n=> grid stores the same text in {r_bpc / g_bpc:.2f}x fewer bits "
          f"than ring")


def common_suffix_len(text: str, i: int, j: int, cap: int = 24) -> int:
    s = 0
    while s < cap and i - s > 0 and j - s > 0 and text[i - s - 1] == text[j - s - 1]:
        s += 1
    return s


def axis_B_decay(text: str, ring_pts: np.ndarray, grid_pts: np.ndarray,
                 r_ring: float, r_grid: float) -> None:
    print("\n" + "=" * 72)
    print("B. DECAY LAW (median pair distance vs common suffix length)")
    print("=" * 72)
    rng = np.random.default_rng(7)
    n = len(text)
    pairs = rng.integers(1, n, size=(60_000, 2))
    buckets_r, buckets_g = {}, {}
    for i, j in pairs:
        if i == j:
            continue
        s = common_suffix_len(text, int(i), int(j))
        buckets_r.setdefault(s, []).append(
            math.hypot(*(ring_pts[i] - ring_pts[j])))
        buckets_g.setdefault(s, []).append(
            math.hypot(*(grid_pts[i] - grid_pts[j])))
    print(f"{'s':>3} {'ring median':>12} {'theory 2r^s':>12} "
          f"{'grid median':>12} {'theory sqrt2 k^-s':>17}")
    for s in sorted(buckets_r):
        if len(buckets_r[s]) < 5:
            continue
        mr = float(np.median(buckets_r[s]))
        mg = float(np.median(buckets_g[s]))
        print(f"{s:>3} {mr:>12.2e} {2 * r_ring ** s:>12.2e} "
              f"{mg:>12.2e} {math.sqrt(2) * r_grid ** s:>17.2e}")
    print("=> both follow their r^s law; grid decays slower per symbol "
          "(larger r),\n   so one float64 point discriminates MORE suffix "
          f"symbols: 52/log2(1/r) = "
          f"{52 / math.log2(1 / r_ring):.1f} (ring) vs "
          f"{52 / math.log2(1 / r_grid):.1f} (grid)")


def axis_C_retrieval(text: str, ring_enc: ChaosWalkEncoder,
                     grid_enc: GridWalkEncoder,
                     ring_pts: np.ndarray, grid_pts: np.ndarray) -> None:
    print("\n" + "=" * 72)
    print("C. RETRIEVAL (substring search on the stored points, float64)")
    print("=" * 72)
    rng = np.random.default_rng(11)
    n = len(text)
    P_ring = ring_pts[1:]
    P_grid = grid_pts[1:]

    def query_point_ring(q: str):
        return ring_enc.encode(q)[-1]

    def query_point_grid(q: str):
        x = y = 0.5
        for c in q:
            x = (x + grid_enc.gx[c]) / grid_enc.k
            y = (y + grid_enc.gy[c]) / grid_enc.k
        return np.array([x, y])

    print(f"{'qlen':>5} | {'ring prec':>9} {'ring rec':>9} | "
          f"{'grid prec':>9} {'grid rec':>9}")
    for qlen in (3, 4, 6, 8, 12):
        precs_r, recs_r, precs_g, recs_g = [], [], [], []
        for _ in range(40):
            s0 = int(rng.integers(0, n - qlen))
            q = text[s0:s0 + qlen]
            occ, at = set(), text.find(q)
            while at != -1:
                occ.add(at + qlen)
                at = text.find(q, at + 1)
            for P, qp, tau, precs, recs in (
                (P_ring, query_point_ring(q),
                 2 * ring_enc.ratio ** qlen + 1e-12, precs_r, recs_r),
                (P_grid, query_point_grid(q),
                 math.sqrt(2) * grid_enc.ratio ** qlen + 1e-12, precs_g, recs_g),
            ):
                d2 = ((P - qp) ** 2).sum(1)
                pred = set((np.nonzero(d2 <= tau * tau)[0] + 1).tolist())
                tp = len(pred & occ)
                precs.append(tp / len(pred) if pred else 0.0)
                recs.append(tp / len(occ))
        print(f"{qlen:>5} | {np.mean(precs_r):>9.2f} {np.mean(recs_r):>9.2f} | "
              f"{np.mean(precs_g):>9.2f} {np.mean(recs_g):>9.2f}")
    print("=> recall must be 1.00 for both (a true suffix match is inside "
          "the ball\n   by the decay law); precision compares the two "
          "boundary-adjacency regimes")


def main() -> None:
    text = load_corpus(100_000).lower()
    alphabet = "".join(sorted(set(text)))
    ring_enc = ChaosWalkEncoder.decodable(alphabet)
    grid = GridWalkCodec(alphabet)
    ring = WalkCodec()

    print(f"corpus: {len(text)} chars | alphabet {len(alphabet)} | "
          f"ring r={ring_enc.ratio:.4f} | grid k={grid.enc.k} "
          f"(r={grid.enc.ratio:.4f}, padded capacity {grid.enc.k ** 2})\n")

    axis_A_rate(text, ring, grid)

    ring_pts = ring_enc.encode(text)
    grid_pts = grid.enc.walk(text)
    axis_B_decay(text, ring_pts, grid_pts, ring_enc.ratio, grid.enc.ratio)
    axis_C_retrieval(text, ring_enc, grid.enc, ring_pts, grid_pts)


if __name__ == "__main__":
    main()
