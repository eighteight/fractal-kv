"""Benchmarks covering the four properties a KV archive needs:

  1. Sanity      -- lossless round-trip on real text
  2. Compression -- bytes/token vs raw text, zlib, lzma, and real KV caches
  3. Speed       -- linear-complexity check + random-access latency
  4. Updates     -- incremental append cost per token

Real-model KV-cache footprints are computed analytically from published
architectures (bytes/token = 2 * layers * kv_heads * head_dim * fp16).
"""

from __future__ import annotations

import lzma
import random
import time
import zlib

from .codec import FractalCodec
from .kv_simulator import ToyKVSimulator

# name: (layers, kv_heads, head_dim) -- fp16, K and V
REAL_KV_CONFIGS = {
    "GPT-2 small (124M)": (12, 12, 64),
    "Llama-3.1-8B (GQA)": (32, 8, 128),
    "Llama-3.1-70B (GQA)": (80, 8, 128),
}


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int) -> int:
    return 2 * layers * kv_heads * head_dim * 2  # K+V, fp16


def sample_text(n_chars: int, seed: int = 0) -> str:
    """English-like filler text (word soup) of the requested length."""
    rng = random.Random(seed)
    words = (
        "the cat sat and dog jumped while eagle flew over lazy fox "
        "quick brown river mountain data memory cache token model "
        "fractal vector compress inference attention layer head"
    ).split()
    out = []
    size = 0
    while size < n_chars:
        w = rng.choice(words)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)[:n_chars]


def run_sanity(codec: FractalCodec) -> dict:
    cases = [
        "the cat sat",
        "the cat sat and the dog jumped while the eagle flew",
        sample_text(10_000),
        "unicode: тест 日本語 émojis 🌀",
        "",
        "a",
    ]
    results = []
    for text in cases:
        c = codec.compress(text)
        ok = codec.decompress(c) == text
        results.append(
            {"chars": len(text), "vectors": c.num_vectors, "lossless": ok}
        )
    return {"cases": results, "all_lossless": all(r["lossless"] for r in results)}


def run_compression(codec: FractalCodec, n_chars: int = 200_000) -> dict:
    text = sample_text(n_chars)
    raw = text.encode("utf-8")
    c = codec.compress(text)
    t0 = time.perf_counter()
    z = zlib.compress(raw, 9)
    t_zlib = time.perf_counter() - t0
    t0 = time.perf_counter()
    x = lzma.compress(raw)
    t_lzma = time.perf_counter() - t0

    tokens = text.split()  # crude ~whitespace tokens
    fractal_bpt = c.memory_bytes() / len(tokens)
    rows = []
    for name, cfg in REAL_KV_CONFIGS.items():
        bpt = kv_bytes_per_token(*cfg)
        rows.append(
            {
                "model": name,
                "kv_bytes_per_token": bpt,
                "fractal_bytes_per_token": round(fractal_bpt, 2),
                "ratio": round(bpt / fractal_bpt, 1),
            }
        )
    return {
        "chars": n_chars,
        "tokens": len(tokens),
        "raw_bytes": len(raw),
        "fractal_bytes": c.memory_bytes(),
        "fractal_vectors": c.num_vectors,
        "zlib_bytes": len(z),
        "lzma_bytes": len(x),
        "zlib_time_s": round(t_zlib, 4),
        "lzma_time_s": round(t_lzma, 4),
        "vs_kv_cache": rows,
    }


def run_speed(codec: FractalCodec) -> dict:
    # linear-complexity check: time compression at growing sizes
    scaling = []
    for n in (25_000, 50_000, 100_000, 200_000, 400_000):
        text = sample_text(n)
        t0 = time.perf_counter()
        c = codec.compress(text)
        dt = time.perf_counter() - t0
        scaling.append(
            {"chars": n, "seconds": round(dt, 4), "us_per_char": round(dt / n * 1e6, 3)}
        )

    # random access latency on a large compressed document
    text = sample_text(1_000_000)
    c = codec.compress(text)
    rng = random.Random(1)
    positions = [rng.randrange(len(text)) for _ in range(5_000)]
    t0 = time.perf_counter()
    for p in positions:
        codec.char_at(c, p)
    ra = (time.perf_counter() - t0) / len(positions)
    correct = all(codec.char_at(c, p) == text[p] for p in positions[:200])
    return {
        "scaling": scaling,
        "random_access_us": round(ra * 1e6, 2),
        "random_access_correct": correct,
        "doc_chars": len(text),
    }


def run_incremental(codec: FractalCodec, n_tokens: int = 2_000) -> dict:
    sim = ToyKVSimulator(codec)
    words = sample_text(n_tokens * 6).split()[:n_tokens]
    t0 = time.perf_counter()
    for w in words:
        sim.add_token(w)
    total = time.perf_counter() - t0

    spot = all(sim.token_at(i) == words[i] for i in range(0, n_tokens, 97))
    full = sim.all_tokens() == words
    return {
        "tokens": n_tokens,
        "append_us_per_token": round(total / n_tokens * 1e6, 2),
        "random_token_access_correct": spot,
        "full_history_correct": full,
        "sim_memory_bytes": sim.memory_bytes(),
        "raw_history_bytes": sim.raw_bytes(),
    }
