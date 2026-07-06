#!/usr/bin/env python3
"""The vector-quantization bridge from KV vectors to fractal storage.

Pipeline:
  1. Run GPT-2 over a training text; collect every per-head K and V vector.
  2. k-means them into a codebook per (layer, K/V) pair.
  3. Evaluate a held-out text token by token; after each step, overwrite
     the newly-cached K/V vectors with their nearest centroids, so all
     later tokens attend to QUANTIZED memory.  Perplexity vs. the exact
     cache measures the quality cost.
  4. The centroid-ID stream is a "text" over a k-symbol alphabet: store it
     with the fractal codecs to get the end-to-end bytes/token.

Requires torch + transformers (see real_llm_kv.py).
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("run: python3 -m pip install torch transformers")

from fractal_poc.codec import ChaosGameCodec
from fractal_poc.walk_codec import WalkCodec, precision_span
from fractal_poc.chaos_walk import sierpinski_ratio

TRAIN_TEXT = (
    "The history of data compression begins long before computers. "
    "Morse code, designed in the 1830s, already assigned shorter symbols "
    "to more frequent letters, an idea that would later be formalized by "
    "information theory. When Claude Shannon published his mathematical "
    "theory of communication in 1948, he gave engineers a precise limit: "
    "no lossless code can, on average, use fewer bits than the entropy of "
    "the source. Everything since has been a search for practical ways to "
    "approach that limit. Huffman coding arrived in 1952 as an optimal "
    "prefix code for known symbol frequencies. Arithmetic coding relaxed "
    "the constraint that each symbol must map to a whole number of bits, "
    "and dictionary methods such as Lempel-Ziv replaced explicit "
    "statistics with adaptive pattern matching. By the 1990s these ideas "
    "had merged into the general purpose compressors that still power "
    "file archives today. "
    "Meanwhile, a very different branch of mathematics was producing "
    "images of startling complexity from tiny formulas. Benoit Mandelbrot "
    "argued that clouds, coastlines, and market prices share a property "
    "he called self-similarity: the small parts resemble the whole. "
    "Michael Barnsley turned this observation into an engineering tool "
    "with iterated function systems, showing that a fern leaf could be "
    "reproduced by four affine maps applied at random. His collage "
    "theorem suggested a form of compression in which the stored object "
    "is not the image itself but a set of contractive maps whose fixed "
    "point approximates it. Fractal image compression enjoyed a wave of "
    "commercial interest in the early 1990s, promising resolution "
    "independence and extreme ratios, though the cost of searching for "
    "good maps kept it from displacing transform coding. "
    "Language modeling connects these threads. A neural network trained "
    "to predict the next token is, by Shannon's argument, an estimate of "
    "the entropy of text, and better predictors compress better. Modern "
    "transformer models push this to remarkable levels, but they pay for "
    "it with memory: to generate each new token, the model consults "
    "cached key and value vectors for every previous position in every "
    "layer. For long documents this cache dwarfs the model weights "
    "themselves, and serving systems now spend enormous effort evicting, "
    "quantizing, or offloading it. The question explored here is whether "
    "the cache's history can be stored in a radically smaller symbolic "
    "form and expanded only when needed, trading arithmetic for memory. "
    "Quantization is the standard first step. Instead of sixteen or "
    "thirty-two bits per number, the cache can often survive on eight or "
    "even four, because attention is a weighted average that tolerates "
    "small perturbations. Vector quantization goes further by replacing "
    "whole vectors with entries from a learned codebook, so that each "
    "position stores only an index. The severity of the approximation "
    "depends on the codebook size and on how gracefully the model "
    "degrades when its memories are rounded to the nearest prototype."
)

EVAL_TEXT = (
    "Steve Reich discovered phasing in 1965 while experimenting with tape "
    "loops of a preacher's voice. Two identical loops, played on machines "
    "that ran at very slightly different speeds, drifted out of "
    "synchrony, and the words dissolved into rhythm before reassembling "
    "farther apart. He transferred the idea to live performers in Piano "
    "Phase, where two pianists repeat the same twelve-note pattern while "
    "one gradually accelerates, and to Clapping Music, which reduces the "
    "process to a single rhythmic cell displaced one eighth note at a "
    "time. The music is a gradual process in the composer's own words: "
    "once the system is set in motion it runs by itself, and the listener "
    "hears every intermediate state. What makes the technique striking is "
    "how much structure emerges from so little material. A single bar of "
    "music, combined with a slowly changing offset, produces interference "
    "patterns, phantom melodies, and a sense of large-scale form, much as "
    "a simple iterated map can produce an intricate attractor. The "
    "connection between minimal rules and rich behavior runs through both "
    "music and mathematics, and it is why a short program can sometimes "
    "describe what looks like an elaborate composition."
)


def get_layers(cache):
    """Return [(keys, values), ...] tensor pairs for any cache layout."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]


def kmeans(X: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    centroids[0] = X[rng.integers(n)]
    d2 = np.full(n, np.inf)
    for j in range(1, k):
        d2 = np.minimum(d2, ((X - centroids[j - 1]) ** 2).sum(1))
        p = d2.astype(np.float64)
        p /= p.sum()
        centroids[j] = X[rng.choice(n, p=p)]
    for _ in range(iters):
        idx = assign(X, centroids)
        for j in range(k):
            members = X[idx == j]
            if len(members):
                centroids[j] = members.mean(0)
    return centroids


def assign(X: np.ndarray, C: np.ndarray, chunk: int = 8192) -> np.ndarray:
    out = np.empty(len(X), dtype=np.int64)
    c2 = (C**2).sum(1)
    for s in range(0, len(X), chunk):
        xb = X[s : s + chunk]
        out[s : s + chunk] = (c2 - 2.0 * xb @ C.T).argmin(1)
    return out


def train_codebooks(model, tok, k: int, text: str = TRAIN_TEXT, stages: int = 1):
    """Train per-(layer, K/V) codebooks; long texts are split into as many
    1024-token chunks as they contain, pooling vectors across chunks.

    stages > 1 = residual VQ: each further codebook is trained on the
    reconstruction error left by the previous ones, and a vector is stored
    as `stages` ids whose centroids sum to the reconstruction.
    """
    all_ids = tok(text)["input_ids"]
    chunks = [all_ids[s : s + 1024] for s in range(0, len(all_ids), 1024)]
    chunks = [c for c in chunks if len(c) >= 64]
    pooled: dict = {}
    for chunk in chunks:
        ids = torch.tensor([chunk])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        for l, (keys, values) in enumerate(get_layers(out.past_key_values)):
            for name, t in (("k", keys), ("v", values)):
                # [1, heads, T, head_dim] -> [heads*T, head_dim]
                X = t[0].transpose(0, 1).reshape(-1, t.shape[-1]).numpy()
                pooled.setdefault((name, l), []).append(X.astype(np.float64))
    books = {}
    for key, parts in pooled.items():
        X = np.concatenate(parts)
        residual = X
        stage_books = []
        for _ in range(stages):
            C = kmeans(residual, k)
            stage_books.append(C)
            residual = residual - C[assign(residual, C)]
        books[key] = stage_books
    return books


def train_codebooks_per_head(
    model, tok, k: int, text: str, stages: int = 2,
    max_vectors: int = 12000, iters: int = 10, progress: bool = False,
):
    """Per-(layer, head, K/V) residual-VQ codebooks.

    Returns books keyed by (name, layer, head) -> [C1, ..., C_stages].
    Vectors are subsampled to max_vectors per codebook and clustered in
    float32 to keep 288 x stages k-means runs tractable on CPU.
    """
    all_ids = tok(text)["input_ids"]
    chunks = [all_ids[s : s + 1024] for s in range(0, len(all_ids), 1024)]
    chunks = [c for c in chunks if len(c) >= 64]
    pooled: dict = {}
    for chunk in chunks:
        ids = torch.tensor([chunk])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        for l, (keys, values) in enumerate(get_layers(out.past_key_values)):
            for name, t in (("k", keys), ("v", values)):
                for h in range(t.shape[1]):
                    X = t[0, h].numpy().astype(np.float32)  # [T, head_dim]
                    pooled.setdefault((name, l, h), []).append(X)
    rng = np.random.default_rng(0)
    books = {}
    for i, (key, parts) in enumerate(sorted(pooled.items())):
        X = np.concatenate(parts)
        if len(X) > max_vectors:
            X = X[rng.choice(len(X), max_vectors, replace=False)]
        residual = X
        stage_books = []
        for _ in range(stages):
            C = kmeans(residual, k, iters=iters)
            stage_books.append(C.astype(np.float64))
            residual = residual - C[assign(residual, C)]
        books[key] = stage_books
        if progress and (i + 1) % 48 == 0:
            print(f"  trained {i + 1}/{len(pooled)} codebooks", flush=True)
    return books


def rvq_quantize(vecs: np.ndarray, stage_books: list[np.ndarray]):
    """Residual-VQ encode: returns (ids [stages, n], reconstruction)."""
    recon = np.zeros_like(vecs)
    ids = []
    for C in stage_books:
        idx = assign(vecs - recon, C)
        ids.append(idx)
        recon = recon + C[idx]
    return np.stack(ids), recon


def eval_ppl(model, tok, books=None, keep_sink=0, keep_recent=0,
             text: str = EVAL_TEXT, quantize_types: str = "kv"):
    """Token-by-token eval with quantized cache memory.

    With books given, cache entries are replaced by their nearest centroid.
    Books may be pooled (keyed (name, layer)) or per-head ((name, layer,
    head)); values may be single codebooks or residual-VQ stage lists.
    keep_sink: never quantize the first S positions (attention sinks).
    keep_recent: quantize a position only once it falls out of the last-W
    window (its exact values age out of "working memory" into "archive").
    quantize_types: "kv" (default), "k" (keys only), or "v" (values only).

    Returns (perplexity, id_stream, nlls): centroid ids per archived
    position and the per-position negative log-likelihoods.
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    seq = ids["input_ids"][0]
    nll, count = 0.0, 0
    nlls: list[float] = []
    id_stream: list[int] = []
    cache = None

    def get_stage_books(name: str, l: int, h: int):
        sb = books.get((name, l, h)) or books.get((name, l))
        return sb if isinstance(sb, list) else [sb]

    per_head = books is not None and any(len(k) == 3 for k in books)

    def quantize_position(pos: int) -> None:
        for l, (keys, values) in enumerate(get_layers(cache)):
            for name, tens in (("k", keys), ("v", values)):
                if name not in quantize_types:
                    continue
                vecs = tens[0, :, pos, :].numpy().astype(np.float64)
                if per_head:
                    recon = np.empty_like(vecs)
                    for h in range(vecs.shape[0]):
                        ids_mat, rec = rvq_quantize(
                            vecs[h : h + 1], get_stage_books(name, l, h)
                        )
                        recon[h] = rec[0]
                        id_stream.extend(int(i) for i in ids_mat.ravel())
                else:
                    ids_mat, recon = rvq_quantize(
                        vecs, get_stage_books(name, l, 0)
                    )
                    id_stream.extend(int(i) for i in ids_mat.ravel())
                tens[0, :, pos, :] = torch.from_numpy(recon).to(tens.dtype)

    with torch.no_grad():
        for t in range(len(seq) - 1):
            out = model(input_ids=seq[t : t + 1].unsqueeze(0),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logp = torch.log_softmax(out.logits[0, -1], dim=-1)
            step_nll = -float(logp[seq[t + 1]])
            nll += step_nll
            nlls.append(step_nll)
            count += 1
            if books is not None:
                aged = t - keep_recent  # position leaving the exact window
                if aged >= keep_sink:
                    quantize_position(aged)
    return math.exp(nll / count), id_stream, nlls


def storage_report(id_stream: list[int], n_tokens: int, k: int) -> None:
    text = "".join(chr(i) for i in id_stream)
    ids_per_token = len(id_stream) / n_tokens

    ratio = sierpinski_ratio(k)
    span = precision_span(ratio)
    walk_bpt = math.ceil(len(text) / span) * 16 / n_tokens

    ref = ChaosGameCodec()
    c = ref.compress(text)
    ok = ref.decompress(c) == text
    ref_bpt = c.memory_bytes() / n_tokens

    import zlib

    packed_bits = len(id_stream) * math.ceil(math.log2(k))
    plain_bpt = packed_bits / 8 / n_tokens
    zlib_bpt = len(zlib.compress(bytes(bytearray(
        b for i in id_stream for b in i.to_bytes(2, "little")
    )), 9)) / n_tokens

    fp32 = 2 * 12 * 768 * 4
    fp16 = fp32 // 2
    print(f"\nstorage of the centroid-id stream "
          f"({ids_per_token:.0f} ids/token, codebook k={k}):")
    print(f"  full KV cache fp32:    {fp32:9,.0f} B/token")
    print(f"  full KV cache fp16:    {fp16:9,.0f} B/token")
    print(f"  ids, walk codec:       {walk_bpt:9,.1f} B/token "
          f"(ratio {ratio:.5f}, span {span})")
    print(f"  ids, reference codec:  {ref_bpt:9,.1f} B/token "
          f"(round-trip {'OK' if ok else 'FAILED'})")
    print(f"  ids, bit-packed:       {plain_bpt:9,.1f} B/token")
    print(f"  ids, zlib -9:          {zlib_bpt:9,.1f} B/token")
    print(f"  -> vs fp16 cache: {fp16 / ref_bpt:,.0f}x smaller "
          f"(reference codec)")
    print(f"  codebook overhead (one-time): "
          f"{24 * k * 64 * 4 / 1024:,.0f} KB")


def main() -> None:
    torch.manual_seed(0)
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()

    n_eval = len(tok(EVAL_TEXT)["input_ids"]) - 1
    print(f"model: {name}   eval tokens: {n_eval}")

    t0 = time.time()
    base_ppl, _, _ = eval_ppl(model, tok)
    print(f"baseline perplexity (exact KV): {base_ppl:.3f} "
          f"[{time.time()-t0:.0f}s]")

    sink, recent = 4, 32
    for k in (256, 1024):
        t0 = time.time()
        books = train_codebooks(model, tok, k)
        ppl, id_stream, _ = eval_ppl(model, tok, books)
        print(f"\nVQ k={k}, everything quantized: perplexity {ppl:.3f} "
              f"({(ppl/base_ppl-1)*100:+.1f}% vs baseline) "
              f"[{time.time()-t0:.0f}s]")
        t0 = time.time()
        ppl_w, _, _ = eval_ppl(
            model, tok, books, keep_sink=sink, keep_recent=recent
        )
        print(f"VQ k={k}, keep {sink} sink + {recent} recent exact: "
              f"perplexity {ppl_w:.3f} "
              f"({(ppl_w/base_ppl-1)*100:+.1f}% vs baseline) "
              f"[{time.time()-t0:.0f}s]")
        storage_report(id_stream, n_eval, k)


if __name__ == "__main__":
    main()
