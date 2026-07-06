#!/usr/bin/env python3
"""Per-head codebooks + keys-vs-values diagnostic (option (b)).

Trains 288 residual-VQ codebooks -- one per (layer, head, K/V) -- on the
whole remainder of the book (disjoint from eval), then evaluates:

  A. stage-1 only, K+V quantized      (8 bits/vector, fair vs pooled k=256)
  B. full RVQ x2, K+V quantized       (16 bits/vector, fair vs pooled RVQ)
  C. values only quantized, keys exact -- is the damage in K or V?
  D. keys only quantized, values exact

Conditions C/D reuse B's codebooks, so training happens once.
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
    n_train = len(tok(train_text)["input_ids"])
    print(f"model: gpt2   eval tokens: {n_eval}   "
          f"codebook training tokens: {n_train}")
    print(f"per-head RVQ: 288 codebooks x 2 stages, k={K}", flush=True)

    t0 = time.time()
    base_ppl, _, base_nlls = eval_ppl(model, tok, text=eval_text)
    b1, b2 = half_ppl(base_nlls)
    print(f"baseline (exact KV): ppl {base_ppl:.3f} "
          f"(1st half {b1:.1f}, 2nd half {b2:.1f}) [{time.time()-t0:.0f}s]",
          flush=True)

    t0 = time.time()
    books2 = train_codebooks_per_head(
        model, tok, K, text=train_text, stages=2, progress=True
    )
    print(f"codebooks trained [{time.time()-t0:.0f}s]", flush=True)
    books1 = {key: sb[:1] for key, sb in books2.items()}  # stage-1 subset

    conditions = [
        ("A. per-head VQ (1 stage), K+V", books1, "kv"),
        ("B. per-head RVQ x2, K+V", books2, "kv"),
        ("C. per-head RVQ x2, VALUES only (keys exact)", books2, "v"),
        ("D. per-head RVQ x2, KEYS only (values exact)", books2, "k"),
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
        if label.startswith("B"):
            storage_report(id_stream, n_eval, K)
            print(f"  per-head codebook memory: "
                  f"{288 * 2 * K * 64 * 4 / 2**20:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
