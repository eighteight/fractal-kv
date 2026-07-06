#!/usr/bin/env python3
"""Probability-weighted chaos game == arithmetic coding.

Demonstrates three things from the discussion:

  1. Weighting the contraction maps by a source model (instead of the
     uniform 1/N of the plain chaos game) turns the codec into arithmetic
     coding, whose rate is the model's cross-entropy. Better model ->
     closer to the Shannon entropy floor ("compression is intelligence").

  2. An exact interval coder round-trips losslessly, and its code length
     equals the cross-entropy (to within the final rounding).

  3. TRUNCATING the code point is lossy, and -- for standard (forward)
     arithmetic coding -- the loss falls on the END of the string (the
     symbols that need the finest precision), the OPPOSITE end from the
     recency-weighted chaos WALK. To move the loss to the beginning you
     code the sequence in reverse.

Runnable with the standard library only.
"""

from __future__ import annotations

import lzma
import math
import sys
import zlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def load_corpus(n_chars: int = 60_000) -> str:
    """Real, non-repetitive English prose (The Time Machine, public domain).

    Using genuine prose rather than a repeated paragraph matters: verbatim
    repetition is trivially exploited by LZ77 (zlib/lzma) and would give the
    dictionary coders an unfair advantage over the n-gram models.
    """
    for p in (Path("timemachine.txt"), Path("../timemachine.txt")):
        if p.exists():
            raw = p.read_text(encoding="utf-8", errors="ignore")
            break
    else:  # fetch it once (public domain, Project Gutenberg #35)
        import urllib.request
        url = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
        print(f"downloading {url} ...", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "fractal-kv/1.0"})
        data = urllib.request.urlopen(req).read()
        Path("../timemachine.txt").write_bytes(data)
        raw = data.decode("utf-8", errors="ignore")
    start = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.find("\n", start) + 1 if start != -1 else 0
    end = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = " ".join(raw[start:end if end != -1 else None].split())
    return body[:n_chars]


SAMPLE = load_corpus()


# ------------------------------------------------------------------ models
def adaptive_bits(text: str, order: int) -> float:
    """Prequential (online) add-1 n-gram code length in bits for `text`.

    The model predicts each symbol from counts of PRIOR symbols only, then
    updates -- so this is a valid, leak-free code length (what an adaptive
    arithmetic coder would actually emit). No train/test split needed.
    """
    V = len(set(text))
    counts: dict[str, Counter] = defaultdict(Counter)
    bits = 0.0
    for i, c in enumerate(text):
        ctx = text[max(0, i - order):i] if order else ""
        cnt = counts[ctx]
        tot = sum(cnt.values())
        p = (cnt.get(c, 0) + 1) / (tot + V)  # Laplace add-1 smoothing
        bits += -math.log2(p)
        cnt[c] += 1
    return bits


# ----------------------------------------------- exact interval (arithmetic) coder
def _model0(text: str):
    """Fixed order-0 model: returns (alphabet, cumulative-interval fn)."""
    freq = Counter(text)
    total = sum(freq.values())
    alphabet = sorted(freq)
    cum = {}
    acc = 0
    for c in alphabet:
        cum[c] = (Fraction(acc, total), Fraction(acc + freq[c], total))
        acc += freq[c]
    return alphabet, cum


def encode_point(text: str, cum) -> Fraction:
    """Chaos-game / arithmetic encode -> a single point (exact Fraction)."""
    lo, hi = Fraction(0), Fraction(1)
    for c in text:
        w = hi - lo
        clo, chi = cum[c]
        lo, hi = lo + w * clo, lo + w * chi
    return lo  # any point in [lo, hi) names the whole string


def decode_point(point: Fraction, n: int, alphabet, cum) -> str:
    """Invert the walk: recover n symbols from a (possibly truncated) point."""
    lo, hi = Fraction(0), Fraction(1)
    out = []
    for _ in range(n):
        w = hi - lo
        # locate the symbol whose sub-interval contains `point`
        for c in alphabet:
            clo, chi = cum[c]
            a, b = lo + w * clo, lo + w * chi
            if a <= point < b:
                out.append(c)
                lo, hi = a, b
                break
        else:  # precision exhausted -> pick the last cell (a wrong guess)
            out.append(alphabet[-1])
            clo, chi = cum[alphabet[-1]]
            lo, hi = lo + w * clo, lo + w * chi
    return "".join(out)


def truncate(point: Fraction, bits: int) -> Fraction:
    """Keep only the top `bits` binary digits of the point (lossy)."""
    scaled = int(point * (1 << bits))
    return Fraction(scaled, 1 << bits)


# ----------------------------------------------------------------- experiments
def rate_table(text: str) -> None:
    n = len(text)
    V = len(set(text))
    raw = text.encode("utf-8")
    print(f"corpus: {n} chars, alphabet {V}, raw utf-8 {len(raw)} bytes\n")
    print(f"{'method':38} {'bits/char':>10} {'vs entropy':>12}")
    print("-" * 62)
    uniform = math.log2(V)
    print(f"{'uniform chaos game (log2 N)':38} {uniform:10.3f} {'(the floor a':>12}")
    print(f"{'':38} {'':>10} {'model-free code hits)':>12}")
    best, best_k = None, None
    for k in (0, 1, 2, 3, 4, 5, 6):
        bpc = adaptive_bits(text, k) / n
        if best is None or bpc < best:
            best, best_k = bpc, k
        tag = "  <- probability-weighted chaos game" if k else ""
        print(f"{'adaptive order-' + str(k) + ' n-gram':38} {bpc:10.3f}{tag}")
    zbpc = 8 * len(zlib.compress(raw, 9)) / n
    xbpc = 8 * len(lzma.compress(raw)) / n
    print(f"{'zlib -9':38} {zbpc:10.3f}")
    print(f"{'lzma':38} {xbpc:10.3f}")
    print(f"\nbest n-gram model: order-{best_k} at {best:.3f} bits/char; "
          f"uniform code spends {uniform:.3f}.")
    print(f"redundancy the uniform (fractal) code leaves on the table: "
          f"~{uniform - best:.2f} bits/char "
          f"({uniform / best:.2f}x larger than the best model here).")
    print("(Shannon's estimate for English is ~1-1.5 bits/char; a strong LLM"
          "\n approaches that, so the achievable gap is even larger.)")


def lossless_and_lossy_demo(text: str) -> None:
    s = text[:240]
    alphabet, cum = _model0(text)  # model trained on full sample
    point = encode_point(s, cum)
    exact = decode_point(point, len(s), alphabet, cum)
    ideal_bits = sum(-math.log2(float(cum[c][1] - cum[c][0])) for c in s)
    print("\n--- exact interval coder (order-0 model) ---")
    print(f"lossless round-trip on {len(s)} chars: {exact == s}")
    print(f"ideal code length: {ideal_bits:.0f} bits "
          f"({ideal_bits/len(s):.3f} bits/char)")

    print("\n--- lossy, FORWARD encoding: keep only B bits of the point ---")
    _lossy_sweep(s, cum, alphabet, reverse=False)
    print("=> errors cluster at the END: forward arithmetic coding spends its")
    print("   least-significant bits on the most RECENT symbols.")

    print("\n--- lossy, REVERSED encoding (recency-graded regime) ---")
    print("Encode the string back-to-front, so the OLDEST symbols get the")
    print("least-significant bits. This is the KV-archive regime: recent")
    print("context stays exact, distant past degrades first.")
    _lossy_sweep(s, cum, alphabet, reverse=True)
    print("=> errors now cluster at the BEGINNING: the distant past blurs")
    print("   while recent context is recovered exactly.")


def _lossy_sweep(s: str, cum, alphabet, reverse: bool) -> None:
    """Truncate the code point to B bits and report WHERE the errors land."""
    coded = s[::-1] if reverse else s
    point = encode_point(coded, cum)
    print(f"{'B bits':>8} {'bits/char':>10} {'errors':>8} "
          f"{'first bad':>10} {'last bad':>9}  {'exact region':>22}")
    for B in (200, 400, 800, 1200):
        rec_coded = decode_point(truncate(point, B), len(coded), alphabet, cum)
        rec = rec_coded[::-1] if reverse else rec_coded
        errs = [i for i in range(len(s)) if rec[i] != s[i]]
        if not errs:
            region = "all exact"
            fb = lb = -1
        else:
            fb, lb = errs[0], errs[-1]
            # contiguous exact run at whichever end survived
            if reverse:
                region = f"tail [{lb+1}..{len(s)-1}] exact"
            else:
                region = f"head [0..{fb-1}] exact"
        print(f"{B:>8} {B/len(s):>10.2f} {len(errs):>8} "
              f"{fb:>10} {lb:>9}  {region:>22}")


if __name__ == "__main__":
    rate_table(SAMPLE)
    lossless_and_lossy_demo(SAMPLE)
