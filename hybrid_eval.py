#!/usr/bin/env python3
"""Capstone: hybrid per-head RVQ -- 4 stages for keys, 2 for values.

Motivation (per_head_eval.py): key quantization causes ~4x more perplexity
damage than value quantization, so bits are reallocated toward keys.
Trains 4 RVQ stages per (layer, head, K/V) once; conditions slice stages.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vq_bridge import eval_ppl, train_codebooks_per_head, storage_report
from scale_eval import load_corpus, half_ppl

K = 256
SINK, RECENT = 4, 32


def main() -> None:
    torch.manual_seed(0)
    body = load_corpus()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    eval_text = body[:5300]
    train_text = body[6000:]
    n_eval = len(tok(eval_text, truncation=True, max_length=1024)["input_ids"])
    print(f"model: gpt2   eval tokens: {n_eval}", flush=True)

    t0 = time.time()
    base_ppl, _, base_nlls = eval_ppl(model, tok, text=eval_text)
    b1, b2 = half_ppl(base_nlls)
    print(f"baseline (exact KV): ppl {base_ppl:.3f} "
          f"(1st half {b1:.1f}, 2nd half {b2:.1f}) [{time.time()-t0:.0f}s]",
          flush=True)

    t0 = time.time()
    books4 = train_codebooks_per_head(
        model, tok, K, text=train_text, stages=4, progress=True
    )
    print(f"288 codebooks x 4 stages trained [{time.time()-t0:.0f}s]",
          flush=True)

    hybrid = {
        key: (sb[:4] if key[0] == "k" else sb[:2])
        for key, sb in books4.items()
    }
    keys4 = {key: sb[:4] for key, sb in books4.items()}

    conditions = [
        ("keys RVQ x4 only (values exact)", keys4, "k"),
        ("HYBRID: keys RVQ x4 + values RVQ x2", hybrid, "kv"),
    ]
    for label, books, qt in conditions:
        t0 = time.time()
        ppl, id_stream, nlls = eval_ppl(
            model, tok, books, keep_sink=SINK, keep_recent=RECENT,
            text=eval_text, quantize_types=qt,
        )
        h1, h2 = half_ppl(nlls)
        print(f"\n{label}: ppl {ppl:.3f} ({(ppl/base_ppl-1)*100:+.1f}%) -- "
              f"1st half {h1:.1f} ({(h1/b1-1)*100:+.1f}%), "
              f"2nd half {h2:.1f} ({(h2/b2-1)*100:+.1f}%) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if label.startswith("HYBRID"):
            storage_report(id_stream, n_eval, K)
            print(f"  per-head codebook memory: "
                  f"{288 * 3 * K * 64 * 4 / 2**20:.0f} MB "
                  f"(keys x4 + values x2, avg 3 stages)", flush=True)


if __name__ == "__main__":
    main()
