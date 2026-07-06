"""Decoder for the chaos-walk encoding, and the full codec built on it.

Encoding recap (chaos_walk.py):  p_k = V(c_k) + r * (p_{k-1} - V(c_k)),
p_0 = center.  r = 1/2 is the midpoint rule.  Expanding the recursion,
every point carries the whole preceding text with weights decaying by a
factor r per step -- that is what makes decoding possible.

Decoding: each letter maps the polygon hull into a scaled copy of itself
pulled toward that letter's vertex ("the letter's cell").  To invert,
find the vertex whose cell contains the point, emit that letter, and undo
the map:
    c_k       = vertex V for which  (p_k - (1-r)V) / r  is inside the hull
    p_{k-1}   = (p_k - (1-r)V(c_k)) / r
For this to be unambiguous the N cells must not overlap, which bounds the
contraction: r <= sierpinski_ratio(N).  The midpoint r=1/2 is only valid
for N <= 4 (triangle / square -- the classic DNA chaos game); for larger
alphabets the ratio must shrink.  That is the "crucial detail" of the
original algorithm: WalkCodec picks the decodable ratio automatically.

Precision economics: the inverse map amplifies float error by 1/r per
step, so one float64 point can be inverted for roughly 44 / log2(1/r)
characters.  WalkCodec therefore stores every m-th walk point (m chosen
from the alphabet's ratio) and regenerates each span by backward search
anchored at the previous stored point.  Cells touch at their boundaries,
so the search keeps a tiny amount of backtracking: if a noisy point sits
exactly on a cell boundary, both candidates are tried and the anchor
check at the end of the span settles it.
"""

from __future__ import annotations

import math

import numpy as np

from .chaos_walk import ChaosWalkEncoder, sierpinski_ratio
from .codec import CompressedText, FractalCodec


def precision_span(ratio: float, mantissa_budget_bits: int = 44) -> int:
    """How many inversions fit in float64 before error swamps the signal."""
    return max(1, int(mantissa_budget_bits / math.log2(1.0 / ratio)))


class WalkDecoder:
    """Backward DFS inversion of the chaos walk (near-greedy when cells
    don't overlap; the anchor check resolves boundary ties)."""

    def __init__(self, encoder: ChaosWalkEncoder):
        self.enc = encoder
        n = len(encoder.alphabet)
        self.inradius = math.cos(math.pi / n) if n >= 3 else 1.0

    def _in_hull(self, p: np.ndarray, tol: float) -> bool:
        r = math.hypot(p[0], p[1])
        if r <= self.inradius:  # fast accept inside inscribed circle
            return True
        if r > 1.0 + tol:  # fast reject outside circumcircle
            return False
        if len(self.enc.vertices) < 3:
            return True  # degenerate hull: circumcircle test is all we have
        v = self.enc.vertices
        for i in range(len(v)):
            a, b = v[i], v[(i + 1) % len(v)]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross > tol:  # vertices wind clockwise: inside is cross <= 0
                return False
        return True

    def decode_backward(
        self,
        point: tuple[float, float],
        n_chars: int,
        anchor: tuple[float, float],
        eps: float = 1e-14,
    ) -> str | None:
        """Recover the n_chars characters that led from `anchor` to `point`."""
        enc = self.enc
        vertices = enc.vertices
        alphabet = enc.alphabet
        ratio = enc.ratio
        inv = 1.0 / ratio
        anchor_arr = np.asarray(anchor)

        def dfs(p: np.ndarray, depth: int, tol: float) -> list[str] | None:
            if depth == n_chars:
                return [] if math.hypot(*(p - anchor_arr)) <= tol else None
            candidates = []
            for i in range(len(vertices)):
                prev = (p - (1.0 - ratio) * vertices[i]) * inv
                if self._in_hull(prev, tol * inv):
                    candidates.append((math.hypot(prev[0], prev[1]), i, prev))
            candidates.sort()  # most central preimage first
            for _, i, prev in candidates:
                rest = dfs(prev, depth + 1, tol * inv)
                if rest is not None:
                    rest.append(alphabet[i])
                    return rest
            return None

        chars = dfs(np.asarray(point), 0, eps)
        if chars is None:
            return None
        # the return path appends each level's char after its predecessors,
        # so the list is already in reading order
        return "".join(chars)


class WalkCodec(FractalCodec):
    """The chaos-walk algorithm as a complete codec: keep every m-th point.

    The contraction ratio adapts to the alphabet so that decoding is
    unambiguous, and the span m adapts to the ratio so that float64
    precision suffices.  vectors[i] is the walk point after span i; each
    span is recovered by backward search anchored at the previous vector.
    """

    name = "chaos-walk (stored anchors, decodable ratio)"

    def __init__(self, span: int | None = None):
        self.span_override = span

    def _encoder(self, alphabet: str) -> ChaosWalkEncoder:
        return ChaosWalkEncoder.decodable(alphabet)

    def _span_for(self, enc: ChaosWalkEncoder) -> int:
        auto = precision_span(enc.ratio)
        return min(self.span_override, auto) if self.span_override else auto

    def compress(self, text: str) -> CompressedText:
        alphabet = "".join(sorted(set(text)))
        enc = self._encoder(alphabet)
        m = self._span_for(enc)
        vectors = []
        p = np.zeros(2)
        for k, c in enumerate(text, start=1):
            p = enc.step(p, c)
            if k % m == 0:
                vectors.append((float(p[0]), float(p[1])))
        if text and len(text) % m != 0:
            vectors.append((float(p[0]), float(p[1])))
        return CompressedText(
            vectors=vectors,
            length=len(text),
            meta={"alphabet": alphabet, "span": m},
        )

    def decompress(self, compressed: CompressedText) -> str:
        if not compressed.vectors:
            return ""
        alphabet = compressed.meta["alphabet"]
        m = compressed.meta["span"]
        dec = WalkDecoder(self._encoder(alphabet))
        pieces = []
        anchor = (0.0, 0.0)
        start = 0
        for i, point in enumerate(compressed.vectors):
            end = min(start + m, compressed.length)
            piece = dec.decode_backward(point, end - start, anchor)
            if piece is None:
                raise ValueError(
                    f"decode failed for span {start}:{end} -- precision "
                    f"exhausted (span={m} too large for this alphabet)"
                )
            pieces.append(piece)
            anchor = point
            start = end
        return "".join(pieces)

    def char_at(self, compressed: CompressedText, pos: int) -> str:
        """O(span): decode only the span containing `pos`."""
        if not 0 <= pos < compressed.length:
            raise IndexError(pos)
        m = compressed.meta["span"]
        i = pos // m
        start = i * m
        end = min(start + m, compressed.length)
        anchor = compressed.vectors[i - 1] if i > 0 else (0.0, 0.0)
        dec = WalkDecoder(self._encoder(compressed.meta["alphabet"]))
        piece = dec.decode_backward(compressed.vectors[i], end - start, anchor)
        if piece is None:
            raise ValueError(f"decode failed for span {start}:{end}")
        return piece[pos - start]

    def append(self, compressed: CompressedText, more: str) -> CompressedText:
        """O(span + len(more)): re-walk only from the last full anchor."""
        alphabet = compressed.meta["alphabet"]
        if any(c not in alphabet for c in more):
            return self.compress(self.decompress(compressed) + more)
        m = compressed.meta["span"]
        enc = self._encoder(alphabet)
        dec = WalkDecoder(enc)

        n_full = compressed.length // m
        tail_len = compressed.length - n_full * m
        vectors = list(compressed.vectors)
        tail = ""
        if tail_len:
            anchor = vectors[n_full - 1] if n_full > 0 else (0.0, 0.0)
            tail = dec.decode_backward(vectors[-1], tail_len, anchor)
            if tail is None:
                raise ValueError("decode failed while appending")
            vectors.pop()

        # re-walk from the last full anchor through tail + new text
        p = np.asarray(vectors[-1] if vectors else (0.0, 0.0), dtype=float)
        new_vectors = []
        text = tail + more
        for j, c in enumerate(text, start=1):
            p = enc.step(p, c)
            if j % m == 0:
                new_vectors.append((float(p[0]), float(p[1])))
        if len(text) % m != 0:
            new_vectors.append((float(p[0]), float(p[1])))
        return CompressedText(
            vectors=vectors + new_vectors,
            length=compressed.length + len(more),
            meta=dict(compressed.meta),
        )
