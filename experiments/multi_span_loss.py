#!/usr/bin/env python3
"""Where does the loss land when the string is stored as MANY points?

The single-point analysis (weighted_chaos.py) shows truncation damaging one
end of the string. But the actual codec stores one point per fixed-length
SPAN. Truncating every point damages the far end of EVERY span, so the
damage is distributed across the whole string as a periodic "comb" -- one
damaged region per span -- rather than concentrated in the distant past.

That distinction matters for the KV-archive argument: a comb damages recent
context too, which is exactly what the exact-window policy is meant to
protect. The fix is to allocate precision ACROSS spans (recent spans get
more bits) rather than uniformly. This script measures both.
"""

from __future__ import annotations

from fractions import Fraction

from weighted_chaos import (
    SAMPLE, _model0, encode_point, decode_point, truncate,
)


def span_encode(text: str, cum, span: int) -> list[tuple[Fraction, int]]:
    """Encode text as one point per span of `span` characters."""
    out = []
    for s in range(0, len(text), span):
        chunk = text[s:s + span]
        out.append((encode_point(chunk, cum), len(chunk)))
    return out


def span_decode(points, alphabet, cum, bits_per_point) -> str:
    """Decode each span from its (possibly truncated) point."""
    out = []
    for (pt, n), B in zip(points, bits_per_point):
        p = truncate(pt, B) if B is not None else pt
        out.append(decode_point(p, n, alphabet, cum))
    return "".join(out)


def damage_map(orig: str, rec: str, width: int = 72) -> str:
    """Compact visual: '.' = exact, 'X' = wrong, bucketed to `width` cols."""
    n = len(orig)
    bucket = max(1, n // width)
    marks = []
    for b in range(0, n, bucket):
        seg_o, seg_r = orig[b:b + bucket], rec[b:b + bucket]
        bad = sum(1 for a, c in zip(seg_o, seg_r) if a != c)
        marks.append("X" if bad > len(seg_o) // 2 else ("x" if bad else "."))
    return "".join(marks)


def report(label, orig, rec, budget_bits):
    errs = [i for i in range(len(orig)) if rec[i] != orig[i]]
    # count contiguous damaged regions
    regions = 0
    prev = -2
    for i in errs:
        if i != prev + 1:
            regions += 1
        prev = i
    print(f"\n{label}")
    print(f"  total bits: {budget_bits}  ({budget_bits/len(orig):.2f} b/char)")
    print(f"  chars wrong: {len(errs)}/{len(orig)}   damaged regions: {regions}")
    print(f"  map: {damage_map(orig, rec)}")
    return regions


def main() -> None:
    text = SAMPLE[:960]
    alphabet, cum = _model0(SAMPLE)
    print(f"string: {len(text)} chars, alphabet {len(alphabet)}")
    print("map legend: '.' exact   'x' some errors   'X' mostly wrong\n")
    print("=" * 78)
    print("A. ONE span (the single-interval analysis)")
    print("=" * 78)
    pts = span_encode(text, cum, span=len(text))
    total = 1400
    rec = span_decode(pts, alphabet, cum, [total])
    report("  one point, truncated", text, rec, total)

    print("\n" + "=" * 78)
    print("B. MANY spans, UNIFORM precision  <- the user's point")
    print("=" * 78)
    for span in (240, 120, 60):
        pts = span_encode(text, cum, span=span)
        k = len(pts)
        per = total // k
        rec = span_decode(pts, alphabet, cum, [per] * k)
        report(f"  {k} spans of {span} chars, {per} bits each",
               text, rec, per * k)

    print("\n" + "=" * 78)
    print("C. MANY spans, RECENCY-WEIGHTED precision")
    print("   (same total budget; recent spans get more bits)")
    print("=" * 78)
    span = 120
    pts = span_encode(text, cum, span=span)
    k = len(pts)
    # linear ramp: oldest span gets least, newest gets most, same total
    weights = [i + 1 for i in range(k)]
    tot_w = sum(weights)
    alloc = [max(8, total * w // tot_w) for w in weights]
    rec = span_decode(pts, alphabet, cum, alloc)
    report(f"  {k} spans, ramped bits {alloc}", text, rec, sum(alloc))

    print("\n" + "=" * 78)
    print("D. MANY spans, EXACT-WINDOW policy")
    print("   (recent spans lossless, distant spans heavily truncated)")
    print("=" * 78)
    keep_exact = 3  # most recent 3 spans stored exactly
    alloc = [None if i >= k - keep_exact else 60 for i in range(k)]
    spent = sum(a for a in alloc if a) + sum(
        900 for a in alloc if a is None)  # ~900b for an exact 120-char span
    rec = span_decode(pts, alphabet, cum, alloc)
    report(f"  {k} spans, last {keep_exact} exact / rest 60 bits",
           text, rec, spent)


if __name__ == "__main__":
    main()
