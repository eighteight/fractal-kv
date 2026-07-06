#!/usr/bin/env python3
"""Encode a text with the chaos-walk encoder and visualize the result.

Usage:
    python3 visualize_walk.py                 # lorem ipsum demo
    python3 visualize_walk.py "your text"     # encode arbitrary text
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fractal_poc.chaos_walk import ChaosWalkEncoder

LOREM = (
    "lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. ut enim "
    "ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
    "aliquip ex ea commodo consequat. duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. excepteur sint occaecat cupidatat non proident, sunt in "
    "culpa qui officia deserunt mollit anim id est laborum."
)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--decodable"]
    decodable = "--decodable" in sys.argv[1:]
    text = args[0] if args else LOREM
    if decodable:
        enc = ChaosWalkEncoder.decodable("".join(sorted(set(text))))
    else:
        enc = ChaosWalkEncoder.for_text(text)
    pts = enc.encode(text)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax in axes:
        # polygon and labeled vertices
        n = len(enc.alphabet)
        poly = np.vstack([enc.vertices, enc.vertices[:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="#bbbbbb", lw=1, zorder=1)
        for c, (vx, vy) in zip(enc.alphabet, enc.vertices):
            label = repr(c)[1:-1] if c == " " else c
            ax.annotate(
                "␣" if c == " " else label,
                (vx * 1.07, vy * 1.07),
                ha="center",
                va="center",
                fontsize=11,
                color="#333333",
            )
        ax.set_aspect("equal")
        ax.axis("off")

    # left: the walk itself (path order shown by color gradient)
    order = np.linspace(0, 1, len(pts))
    axes[0].plot(pts[:, 0], pts[:, 1], color="#cccccc", lw=0.4, zorder=2)
    axes[0].scatter(
        pts[:, 0], pts[:, 1], c=order, cmap="viridis", s=14, zorder=3
    )
    axes[0].scatter(*pts[0], color="red", s=60, zorder=4, label="start (center)")
    axes[0].scatter(*pts[-1], color="orange", s=60, zorder=4, label="end")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title(
        f"Chaos walk: {len(text)} chars → {len(pts)} points "
        f"(alphabet size {len(enc.alphabet)})"
    )

    # right: points only -- the fractal (Sierpinski-like) structure
    axes[1].scatter(pts[1:, 0], pts[1:, 1], s=6, color="#1f4e79", zorder=3)
    axes[1].set_title("Encoding as a point cloud (fractal structure)")

    if decodable:
        from fractal_poc.walk_codec import WalkCodec

        c = WalkCodec().compress(text)
        anchors = np.array(c.vectors)
        axes[1].scatter(
            anchors[:, 0], anchors[:, 1], s=70, facecolors="none",
            edgecolors="red", lw=1.5, zorder=5,
            label=f"stored anchors ({len(anchors)} of {len(text)} chars)",
        )
        axes[1].legend(loc="lower right", fontsize=9)
        axes[0].set_title(
            axes[0].get_title() + f"  [decodable ratio {enc.ratio:.4f}]"
        )

    fig.suptitle(f'Input: "{text[:70]}{"..." if len(text) > 70 else ""}"')
    out = "walk_decodable.png" if decodable else "walk_visualization.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"alphabet ({len(enc.alphabet)}): {enc.alphabet!r}")
    print(f"points recorded: {len(pts)}")
    print("first 5 points:")
    for i, (x, y) in enumerate(pts[:5]):
        char = f"  after {text[i-1]!r}" if i else "  (center)"
        print(f"  p{i}: ({x:+.6f}, {y:+.6f}){char}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
