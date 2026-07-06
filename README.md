# Fractal KV-Cache Archives

[![arXiv](https://img.shields.io/badge/arXiv-2607.07144-b31b1b.svg)](https://arxiv.org/abs/2607.07144)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21237993.svg)](https://doi.org/10.5281/zenodo.21237993)

Code for the paper *Fractal KV-Cache Archives: Lossless Symbolic Storage
with In-Place Retrieval for Long-Context LLM Inference*
([arXiv:2607.07144](https://arxiv.org/abs/2607.07144);
[Zenodo](https://doi.org/10.5281/zenodo.21237993)). This repository is code
only; running the scripts regenerates every figure and table in the paper.

A contractive iterated-map code serializes a symbol sequence into a short
sequence of 2D float vectors. The repository studies its use as an
**archive format for a quantized LLM KV cache**: the code is lossless,
linear-time, supports O(1) random access and O(1) amortized append, and
doubles as an in-place substring-search index.

## Requirements

- Core codec and its benchmarks: Python 3, `numpy`, `matplotlib`.
- KV-cache experiments (`real_llm_kv.py`, `vq_bridge.py`, `scale_eval.py`,
  `per_head_eval.py`, `hybrid_eval.py`): additionally `torch` +
  `transformers`. These download GPT-2 (~500 MB) and a public-domain
  corpus on first run. Everything runs on CPU.

```bash
pip install numpy matplotlib          # core
pip install torch transformers        # for the KV-cache experiments
```

## Quick start

```bash
python3 run_poc.py --walk
```

Runs five self-contained benchmark sections on the chaos-walk codec:

1. **Sanity** — lossless round-trip on ASCII, Unicode, and edge-case text.
2. **Compression** — bytes/token vs. raw text, zlib, lzma, and analytic
   KV-cache footprints of GPT-2 / Llama-3.1-8B / Llama-3.1-70B.
3. **Speed** — linear-time scaling and random-access latency (look up an
   arbitrary position without decompressing the whole document).
4. **Incremental updates** — the toy KV simulator appends tokens one at a
   time and verifies the history is recoverable from compressed form only.
5. **Semantic similarity** — do paraphrases map to nearby vectors? (The
   codec is a positional code and is expected to fail this; the test exists
   to distinguish structural from semantic similarity.)

`python3 run_poc.py` (no flag) runs the same suite on the full-mantissa
`ChaosGameCodec`, the information-optimal reference point.

## The chaos-walk codec

The codec is implemented in `fractal_poc/chaos_walk.py` (encoder: a
chaos-game walk on a regular N-gon, one vertex per alphabet symbol) and
`fractal_poc/walk_codec.py` (decoder + full codec).

```bash
python3 visualize_walk.py              # midpoint walk on lorem ipsum -> PNG
python3 visualize_walk.py --decodable  # decodable-ratio walk + stored anchors
python3 visualize_cells.py             # why the contraction ratio must adapt
```

How it works:

- Each walk point encodes the entire preceding text, with older symbols
  decaying by the contraction ratio `r` per step. Decoding inverts the
  walk backward: `c_k` is the vertex whose cell contains `p_k`, then
  `p_(k-1) = (p_k - (1-r)·V(c_k)) / r`.
- The midpoint rule (`r = 1/2`) is uniquely decodable only for alphabets of
  up to 4 symbols (the classic DNA chaos-game square). For `N` symbols the
  ratio must shrink to the kissing ratio
  `r = 1 / (2(1 + Σ_{k≤N/4} cos(2πk/N)))` so the per-symbol cells stay
  disjoint — for a 24-char alphabet, `r ≈ 0.116`.
- Float64 precision admits about `44 / log2(1/r)` backward steps, so the
  codec stores one point per fixed-length span (~14 chars for a 24-char
  alphabet) and decodes each span anchored at the previous stored point.
  Lossless, linear-time, with O(span) random access and append.

## Reproducing the note

Each command prints the numbers and/or writes the figure it backs.

| Command | Produces |
| --- | --- |
| `python3 run_poc.py --walk` | Table 1 (codec primitives); lossless round-trips |
| `python3 visualize_cells.py` | Figure 1 (`cells_overlap.png`), decodability |
| `python3 real_llm_kv.py` | real GPT-2 KV cache vs. codec size |
| `python3 vq_bridge.py` | VQ bridge, short-context sanity run |
| `python3 scale_eval.py` | 1024-token pooled vs. residual-VQ sweep |
| `python3 per_head_eval.py` | per-head codebooks + key/value asymmetry |
| `python3 hybrid_eval.py` | Table 2 hybrid row (keys ×4, values ×2) |
| `python3 make_pareto.py` | Figure 2 (`pareto.png`), rate–distortion |
| `python3 suffix_retrieval.py` | Figure 3 (`suffix_retrieval.png`), retrieval |

## What the experiments show

With an exact window of a few attention-sink and recent tokens kept in full
precision and the rest archived through per-head residual vector
quantization, the archived KV cache shrinks 36–54× versus an fp16 cache at
an 11–15% perplexity cost on GPT-2 (1024-token contexts). Quantizing keys
is roughly 4× more damaging than quantizing values, which a bit-asymmetric
hybrid exploits. The compression itself comes from the vector quantizer;
the fractal codec is the serialization layer — on the resulting index
streams its size is comparable to byte-oriented coders, but unlike them it
offers O(1) random access, O(1) append, and in-place substring search over
its own contents (nearest-neighbor distance is a graded suffix similarity).

See the note for full results, baselines, and limitations.

## Repository layout

```
fractal_poc/
  codec.py          # FractalCodec interface + full-mantissa ChaosGameCodec
  chaos_walk.py     # chaos-walk encoder (N-gon walk)
  walk_codec.py     # walk decoder + full codec (adaptive kissing ratio)
  kv_simulator.py   # ToyKVSimulator (sequential growth / random access)
  benchmarks.py     # sanity, compression, speed, incremental benchmarks
  semantic_test.py  # paraphrase-clustering test
run_poc.py          # codec benchmark suite (--walk, or reference)
real_llm_kv.py      # GPT-2 KV cache vs. codec size
vq_bridge.py        # VQ codebooks, quantized-cache eval, storage report
scale_eval.py       # 1024-token pooled vs. residual-VQ sweep
per_head_eval.py    # per-head codebooks + key/value diagnostic
hybrid_eval.py      # keys x4 + values x2 hybrid
make_pareto.py      # rate-distortion figure
suffix_retrieval.py # decay law + in-place retrieval figure
visualize_walk.py   # walk visualizations
visualize_cells.py  # cell-overlap / decodability figure
```

## Citation

If you use this work, please cite the paper:

> Gusev, V. (2026). *Fractal KV-Cache Archives: Lossless Symbolic Storage
> with In-Place Retrieval for Long-Context LLM Inference.*
> arXiv:2607.07144. https://arxiv.org/abs/2607.07144

```bibtex
@misc{gusev2026fractalkv,
  author       = {Gusev, Vladimir},
  title        = {Fractal KV-Cache Archives: Lossless Symbolic Storage
                  with In-Place Retrieval for Long-Context LLM Inference},
  year         = {2026},
  eprint       = {2607.07144},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  doi          = {10.5281/zenodo.21237993},
  url          = {https://arxiv.org/abs/2607.07144}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The
accompanying preprint is licensed CC BY 4.0.
