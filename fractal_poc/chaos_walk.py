"""Fractal walk encoder.

Encoding:
  1. Build a regular polygon with one vertex per alphabet symbol.
  2. Start at the polygon's center.
  3. For each letter: pick its vertex, connect it to the previous point,
     and record the MIDPOINT of that segment as the next point.
  4. The recorded points are the encoding.

This is the generalized chaos game: p_{k+1} = (p_k + V(c_k)) / 2, where
V(c) is the vertex assigned to character c.  Every point therefore carries
the full history of the text so far, with each older letter contributing
at half the weight of the next (binary decay) -- which is what makes a
decoding procedure possible at all.
"""

from __future__ import annotations

import math

import numpy as np


def sierpinski_ratio(n: int) -> float:
    """Largest contraction ratio for which the N letter cells don't overlap.

    Each letter maps the polygon hull into a scaled copy pulled toward its
    vertex.  For decoding to be unambiguous these copies must not overlap;
    the touching ("kissing") ratio for a regular N-gon is
        r = 1 / (2 * (1 + sum_{k=1..floor(N/4)} cos(2*pi*k/N)))
    which gives the familiar 1/2 for N=3 and N=4 (triangle, square = DNA
    CGR) and shrinks for larger alphabets.  THIS is the detail the midpoint
    memory loses: midpoint jumps are only decodable up to 4 symbols.
    """
    if n <= 4:
        return 0.5
    s = sum(math.cos(2 * math.pi * k / n) for k in range(1, n // 4 + 1))
    return 1.0 / (2.0 * (1.0 + s))


class ChaosWalkEncoder:
    def __init__(self, alphabet: str, radius: float = 1.0, ratio: float | None = None):
        self.alphabet = alphabet
        n = max(len(alphabet), 1)
        # ratio: fraction of the remaining distance KEPT when jumping toward
        # a vertex: p' = V + ratio * (p - V).  0.5 = the midpoint rule.
        self.ratio = 0.5 if ratio is None else ratio
        # vertex 0 at the top, going clockwise
        angles = [math.pi / 2 - 2 * math.pi * k / n for k in range(n)]
        self.vertices = np.array(
            [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
        )
        self.index = {c: i for i, c in enumerate(alphabet)}

    @classmethod
    def for_text(cls, text: str, radius: float = 1.0, ratio: float | None = None) -> "ChaosWalkEncoder":
        return cls("".join(sorted(set(text))), radius, ratio)

    @classmethod
    def decodable(cls, alphabet: str, radius: float = 1.0) -> "ChaosWalkEncoder":
        """Encoder whose walk is uniquely invertible (non-overlapping cells)."""
        return cls(alphabet, radius, ratio=sierpinski_ratio(max(len(alphabet), 1)))

    def step(self, p: np.ndarray, c: str) -> np.ndarray:
        v = self.vertices[self.index[c]]
        return v + self.ratio * (p - v)

    def encode(self, text: str) -> np.ndarray:
        """Return the walk: (len(text)+1) x 2 array, row 0 = center."""
        points = np.empty((len(text) + 1, 2))
        points[0] = (0.0, 0.0)
        p = points[0]
        for k, c in enumerate(text, start=1):
            p = self.step(p, c)
            points[k] = p
        return points
