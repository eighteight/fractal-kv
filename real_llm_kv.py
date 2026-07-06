#!/usr/bin/env python3
"""Measure a real model's KV cache and compare its size with the fractal
representation of the same token strings.

Requires:  pip install torch transformers
Downloads GPT-2 small (~500 MB) on first run; everything runs on CPU/MPS.
"""

from __future__ import annotations

import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit(
        "torch/transformers not installed.\n"
        "Run:  python3 -m pip install torch transformers\n"
        "then re-run this script."
    )

from fractal_poc.codec import ChaosGameCodec
from fractal_poc.walk_codec import WalkCodec
from fractal_poc.benchmarks import sample_text


def main() -> None:
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    text = sample_text(4000)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        out = model(**ids, use_cache=True)

    # KV cache: Cache object (new transformers) or tuple of (key, value) pairs
    kv = out.past_key_values
    if hasattr(kv, "key_cache"):  # DynamicCache
        tensors = list(kv.key_cache) + list(kv.value_cache)
    elif hasattr(kv, "layers"):  # newer Cache layout
        tensors = [t for l in kv.layers for t in (l.keys, l.values)]
    else:  # legacy tuple of (key, value) per layer
        tensors = [t for layer in kv for t in layer]
    kv_bytes = sum(t.numel() * t.element_size() for t in tensors if t is not None)

    token_strings = tok.convert_ids_to_tokens(ids["input_ids"][0])
    history = "\x1f".join(token_strings)

    n = len(token_strings)
    print(f"model: {name}   tokens: {n}")
    print(f"real KV cache:          {kv_bytes / 1024:10.1f} KB "
          f"({kv_bytes / n:8.1f} B/token)")
    for codec in (WalkCodec(), ChaosGameCodec()):
        compressed = codec.compress(history)
        assert codec.decompress(compressed) == history, "round-trip failed"
        fb = compressed.memory_bytes()
        print(f"{codec.name:<22.22s}  {fb / 1024:10.1f} KB "
              f"({fb / n:8.1f} B/token, {compressed.num_vectors} vectors, "
              f"{kv_bytes / fb:,.0f}x smaller)")
    print(
        "\nNote: the two stores are not information-equivalent -- the KV cache"
        "\nholds contextualized attention states, the fractal store holds the"
        "\nexact token history.  The ratio shows the memory headroom available"
        "\nif history can be re-expanded on demand (see README, 'What this"
        "\nproves')."
    )


if __name__ == "__main__":
    main()
