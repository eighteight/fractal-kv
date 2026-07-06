#!/usr/bin/env python3
"""Rate-distortion figure: perplexity cost vs. KV-archive compression.

All points measured on GPT-2, The Time Machine, 1024-token contexts,
4-sink + 32-recent exact window; archive quantized.  Storage is the
centroid-id stream serialized by the fractal codec (bytes/token); the
compression ratio is relative to an fp16 KV cache (36,864 B/token).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP16 = 36864.0  # bytes/token, GPT-2 fp16 KV cache

# (label, bytes/token, ppl_increase_pct, family)
POINTS = [
    ("pooled VQ k=256",       342, 51.0, "pooled"),
    ("pooled VQ k=1024",      445, 38.1, "pooled"),
    ("pooled RVQ k=256x2",    684, 37.2, "pooled"),
    ("pooled RVQ k=1024x2",   889, 29.2, "pooled"),
    ("per-head VQ k=256",     342, 23.7, "perhead"),
    ("per-head RVQ k=256x2",  684, 15.0, "perhead"),
    ("hybrid (K x4, V x2)",  1025, 11.2, "hybrid"),
]

STYLE = {
    "pooled":  dict(color="#c0504d", marker="o", label="pooled codebooks"),
    "perhead": dict(color="#1f4e79", marker="s", label="per-head codebooks"),
    "hybrid":  dict(color="#2e8b57", marker="D", label="per-head hybrid"),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for label, bpt, ppl, fam in POINTS:
        st = STYLE[fam]
        ax.scatter(
            FP16 / bpt, ppl, s=130, zorder=3,
            color=st["color"], marker=st["marker"],
            label=st["label"] if fam not in seen else None,
            edgecolor="white", linewidth=0.8,
        )
        seen.add(fam)
        dy = 1.6 if fam != "pooled" else -2.4
        ax.annotate(label, (FP16 / bpt, ppl), fontsize=8.5,
                    xytext=(6, dy * 3), textcoords="offset points")

    # guide arrow: per-head Pareto dominates pooled
    ax.annotate(
        "", xy=(FP16 / 684, 15.0), xytext=(FP16 / 684, 37.2),
        arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.4),
    )
    ax.text(FP16 / 684 * 1.03, 26, "per-head\ncodebooks\n(same bits)",
            fontsize=8.5, color="#1f4e79", va="center")

    ax.set_xscale("log")
    ax.set_xlabel("KV-archive compression ratio vs. fp16 cache  "
                  "(higher = smaller)", fontsize=11)
    ax.set_ylabel("perplexity increase over exact cache (%)", fontsize=11)
    ax.set_title("Rate–distortion of the compressed KV archive "
                 "(GPT-2, 1024-token contexts)", fontsize=12)
    ax.axhline(0, color="#888888", lw=1, ls=":")
    ax.text(105, 1.0, "exact cache (0%)", fontsize=8, color="#666666")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(30, 130)
    ax.set_ylim(-3, 56)
    fig.savefig("pareto.png", dpi=120, bbox_inches="tight")
    print("saved pareto.png")


if __name__ == "__main__":
    main()
