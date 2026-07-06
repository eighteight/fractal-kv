#!/usr/bin/env python3
"""Why midpoint decoding breaks for alphabets larger than 4.

Each letter i maps the whole polygon hull into a half-scale copy of itself
pulled toward vertex V_i -- that copy is the set of ALL points a walk can
occupy right after letter i (the letter's "cell").  Decoding the last
letter = asking which cell the point is in, so cells must not overlap.

A half-scale copy has 1/4 of the hull's area, and there are N of them:
N=3 leaves gaps (Sierpinski triangle), N=4 tiles exactly (DNA chaos game
square), N>4 must overlap.  Shrinking the ratio restores disjointness.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

from fractal_poc.chaos_walk import ChaosWalkEncoder, sierpinski_ratio

ALPHA = "abcdefghijklmnopqrstuvwx"


def draw(ax, n: int, ratio: float, title: str) -> None:
    enc = ChaosWalkEncoder(ALPHA[:n], ratio=ratio)
    hull = enc.vertices
    ax.add_patch(
        MplPolygon(hull, closed=True, fill=False, edgecolor="#888888", lw=1.2)
    )
    for v in hull:
        # cell = hull scaled by `ratio` toward vertex v
        cell = v + ratio * (hull - v)
        ax.add_patch(
            MplPolygon(
                cell, closed=True, facecolor="#1f77b4", alpha=0.35,
                edgecolor="#1f4e79", lw=0.6,
            )
        )
    coverage = n * ratio**2
    ax.set_title(f"{title}\ncells cover {coverage:.0%} of polygon area", fontsize=10)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    draw(axes[0], 3, 0.5, "N=3, midpoint (r=1/2)\nSierpinski triangle: disjoint")
    draw(axes[1], 4, 0.5, "N=4, midpoint (r=1/2)\nDNA chaos game: tiles exactly")
    draw(axes[2], 6, 0.5, "N=6, midpoint (r=1/2)\nOVERLAP: darker regions ambiguous")
    draw(axes[3], 24, 0.5, "N=24, midpoint (r=1/2)\nheavy overlap everywhere")
    r24 = sierpinski_ratio(24)
    draw(axes[4], 24, r24, f"N=24, kissing ratio r={r24:.3f}\ndisjoint again -> decodable")
    fig.suptitle(
        "Letter cells: where the walk can be right after each letter "
        "(overlap = the last letter cannot be identified)",
        fontsize=13,
    )
    out = "cells_overlap.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
