#!/usr/bin/env python3
"""Run the fractal codec benchmark suite: sanity, compression, speed,
incremental updates, and semantic similarity.

Usage:
    python3 run_poc.py            # reference full-mantissa ChaosGameCodec
    python3 run_poc.py --walk     # the chaos-walk codec (backward decoding)
"""

from __future__ import annotations

import argparse
import json

from fractal_poc.codec import ChaosGameCodec
from fractal_poc.benchmarks import (
    run_sanity,
    run_compression,
    run_speed,
    run_incremental,
)
from fractal_poc.semantic_test import run_semantic


def section(title: str, data: dict) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--walk",
        action="store_true",
        help="use WalkCodec (the chaos-walk algorithm with backward decoding)",
    )
    parser.add_argument(
        "--span",
        type=int,
        default=32,
        help="WalkCodec: store every m-th walk point (default 32)",
    )
    args = parser.parse_args()

    if args.walk:
        from fractal_poc.walk_codec import WalkCodec

        codec = WalkCodec(span=args.span)
    else:
        codec = ChaosGameCodec()

    print(f"Codec under test: {codec.name}")

    # Two short examples
    for s in ("the cat sat",
              "the cat sat and the dog jumped while the eagle flew"):
        c = codec.compress(s)
        pretty = ", ".join(f"[{x:.6g}, {y:.6g}]" for x, y in c.vectors)
        print(f'  "{s}"\n    -> ({pretty})   [{c.num_vectors} vector(s)]')

    section("1. SANITY (lossless round-trip)", run_sanity(codec))
    section("2. COMPRESSION (vs raw / zlib / lzma / real KV caches)",
            run_compression(codec))
    section("3. SPEED (linear complexity + random access)", run_speed(codec))
    section("4. INCREMENTAL UPDATES (KV simulator, token by token)",
            run_incremental(codec))
    section("5. SEMANTIC SIMILARITY (the critical question)",
            run_semantic(codec))


if __name__ == "__main__":
    main()
