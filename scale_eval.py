#!/usr/bin/env python3
"""Scaled VQ-bridge eval: ~1000-token contexts on a real book.

- Corpus: H.G. Wells, "The Time Machine" (Project Gutenberg #35, public
  domain), cached locally as timemachine.txt.
- Codebooks trained on four disjoint 1024-token chunks from the middle of
  the book; evaluation on a disjoint ~1000-token passage from the start.
- At 1000 tokens with a 4-sink + 32-recent exact window, ~96% of the
  cache the model attends to is quantized archive (vs ~84% in the short
  vq_bridge.py run).
- Reports perplexity overall and over the second half of the document,
  where predictions depend most on long-range (archived) context.
"""

from __future__ import annotations

import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vq_bridge import eval_ppl, train_codebooks, storage_report

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
CORPUS = Path("timemachine.txt")


def load_corpus() -> str:
    if not CORPUS.exists():
        print(f"downloading {GUTENBERG_URL} ...")
        req = urllib.request.Request(
            GUTENBERG_URL, headers={"User-Agent": "fractal-poc/0.1"}
        )
        CORPUS.write_bytes(urllib.request.urlopen(req).read())
    raw = CORPUS.read_text(encoding="utf-8")
    # strip Gutenberg header/footer
    start = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.find("\n", start) + 1 if start != -1 else 0
    end = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = raw[start:end if end != -1 else None]
    return " ".join(body.split())  # collapse whitespace


def half_ppl(nlls: list[float]) -> tuple[float, float]:
    mid = len(nlls) // 2
    return (
        math.exp(sum(nlls[:mid]) / mid),
        math.exp(sum(nlls[mid:]) / (len(nlls) - mid)),
    )


def main() -> None:
    torch.manual_seed(0)
    body = load_corpus()
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    # eval: ~1000 tokens from the book's opening; train: 4 chunks from
    # later chapters (disjoint by a wide margin)
    eval_text = body[:5300]
    train_text = body[40_000:40_000 + 22_000]
    n_eval = len(tok(eval_text, truncation=True, max_length=1024)["input_ids"])
    n_train = len(tok(train_text)["input_ids"])
    print(f"model: gpt2   eval tokens: {n_eval}   "
          f"codebook training tokens: {n_train}")

    sink, recent = 4, 32
    print(f"exact window: {sink} sink + {recent} recent "
          f"= {(sink + recent) / n_eval:.1%} of context; rest is archive")

    t0 = time.time()
    base_ppl, _, base_nlls = eval_ppl(model, tok, text=eval_text)
    b1, b2 = half_ppl(base_nlls)
    print(f"\nbaseline (exact KV): ppl {base_ppl:.3f} "
          f"(1st half {b1:.1f}, 2nd half {b2:.1f}) [{time.time()-t0:.0f}s]")

    for k, stages in ((256, 1), (1024, 1), (256, 2), (1024, 2)):
        t0 = time.time()
        books = train_codebooks(model, tok, k, text=train_text, stages=stages)
        t_train = time.time() - t0
        t0 = time.time()
        ppl, id_stream, nlls = eval_ppl(
            model, tok, books, keep_sink=sink, keep_recent=recent,
            text=eval_text,
        )
        h1, h2 = half_ppl(nlls)
        label = f"VQ k={k}" if stages == 1 else f"RVQ k={k} x{stages} stages"
        print(f"\n{label} (archive quantized): ppl {ppl:.3f} "
              f"({(ppl/base_ppl-1)*100:+.1f}%) -- "
              f"1st half {h1:.1f} ({(h1/b1-1)*100:+.1f}%), "
              f"2nd half {h2:.1f} ({(h2/b2-1)*100:+.1f}%) "
              f"[train {t_train:.0f}s, eval {time.time()-t0:.0f}s]")
        storage_report(id_stream, n_eval, k)


if __name__ == "__main__":
    main()
