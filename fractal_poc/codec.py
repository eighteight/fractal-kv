"""String -> 2D-float-vector codec interface, plus a reference
implementation based on the chaos-game / IFS construction.

A string is stored as a (short) sequence of fixed-dimension float vectors,

    "the cat sat"                                   -> [(0.11, 0.22)]
    "the cat sat and the dog jumped while ..."      -> [(0.11, 0.22), (0.304, 0.123)]

where the number of vectors grows linearly-but-slowly with string length.

The `FractalCodec` base class defines the interface (compress / decompress,
plus optional char_at / append for O(1) random access and updates); the
chaos-walk codec in `walk_codec.py` is the primary implementation and
`ChaosGameCodec` below is the full-mantissa reference. The KV simulator,
benchmarks, and semantic test are all codec-agnostic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CompressedText:
    """A string compressed to a sequence of 2D float vectors.

    `vectors` is the payload the KV cache would hold: num_vectors x 2 float64.
    `meta` is small per-document metadata a codec may need to invert the
    mapping (alphabet, length, ...); it is counted in memory_bytes().
    """

    vectors: list[tuple[float, float]]
    length: int  # original string length in characters
    meta: dict = field(default_factory=dict)

    def memory_bytes(self, float_bytes: int = 8) -> int:
        meta_bytes = sum(len(str(k)) + len(str(v)) for k, v in self.meta.items())
        return len(self.vectors) * 2 * float_bytes + meta_bytes

    @property
    def num_vectors(self) -> int:
        return len(self.vectors)


class FractalCodec(ABC):
    """Interface a codec implements to run against the harness.

    random access (char_at) and incremental update (append) are the two
    architectural requirements the KV cache imposes; a codec that supports
    them natively overrides the default implementations, which fall back to
    full decompression / full recompression.
    """

    name = "abstract"

    @abstractmethod
    def compress(self, text: str) -> CompressedText: ...

    @abstractmethod
    def decompress(self, compressed: CompressedText) -> str: ...

    def char_at(self, compressed: CompressedText, pos: int) -> str:
        """Random access. Default: decompress everything (slow)."""
        return self.decompress(compressed)[pos]

    def append(self, compressed: CompressedText, more: str) -> CompressedText:
        """Incremental update. Default: recompress everything (slow)."""
        return self.compress(self.decompress(compressed) + more)


class ChaosGameCodec(FractalCodec):
    """Reference stand-in codec built on the chaos-game (IFS) construction.

    Each character of the alphabet selects a contractive affine map
    w_i(p) = (p + i) / A on the unit interval; the orbit endpoint after
    feeding a chunk of characters through the maps encodes the whole chunk
    exactly (it is the base-A expansion 0.c_k ... c_2 c_1).  The endpoint
    is split across the two coordinates of a 2D vector so that each vector
    carries up to 104 bits (2 x 52-bit float64 mantissas) of payload.

    Properties: lossless, O(n) compression, O(1) random access, O(1)
    incremental append.  It is NOT semantic: similar meanings do not map
    to nearby points (see semantic_test.py).  Swap in the real algorithm
    to test that property.
    """

    name = "chaos-game (reference)"
    MANTISSA_BITS = 52  # exact dyadic rationals in float64
    CAPACITY = 1 << (2 * MANTISSA_BITS)  # payload per 2D vector

    def __init__(self, alphabet: str | None = None):
        self.fixed_alphabet = alphabet

    # -- helpers -----------------------------------------------------------

    def _alphabet_for(self, text: str) -> str:
        if self.fixed_alphabet is not None:
            return self.fixed_alphabet
        return "".join(sorted(set(text)))

    @classmethod
    def chars_per_vector(cls, alphabet_size: int) -> int:
        if alphabet_size < 2:
            return 2 * cls.MANTISSA_BITS  # degenerate single-symbol alphabet
        k, cap = 0, 1
        while cap * alphabet_size <= cls.CAPACITY:
            cap *= alphabet_size
            k += 1
        return k

    @classmethod
    def _pack(cls, indices: list[int], base: int) -> tuple[float, float]:
        n = 0
        for idx in reversed(indices):  # first char = least-significant digit
            n = n * base + idx
        x = n & ((1 << cls.MANTISSA_BITS) - 1)
        y = n >> cls.MANTISSA_BITS
        scale = float(1 << cls.MANTISSA_BITS)
        return (x / scale, y / scale)

    @classmethod
    def _unpack(cls, vec: tuple[float, float], base: int, count: int) -> list[int]:
        scale = 1 << cls.MANTISSA_BITS
        x = round(vec[0] * scale)
        y = round(vec[1] * scale)
        n = (y << cls.MANTISSA_BITS) | x
        out = []
        for _ in range(count):
            n, idx = divmod(n, base)
            out.append(idx)
        return out

    # -- FractalCodec interface --------------------------------------------

    def compress(self, text: str) -> CompressedText:
        alphabet = self._alphabet_for(text)
        base = max(len(alphabet), 2)
        cpv = self.chars_per_vector(base)
        index = {c: i for i, c in enumerate(alphabet)}
        vectors = []
        for start in range(0, len(text), cpv):
            chunk = [index[c] for c in text[start:start + cpv]]
            vectors.append(self._pack(chunk, base))
        return CompressedText(
            vectors=vectors,
            length=len(text),
            meta={"alphabet": alphabet},
        )

    def decompress(self, compressed: CompressedText) -> str:
        alphabet = compressed.meta["alphabet"]
        base = max(len(alphabet), 2)
        cpv = self.chars_per_vector(base)
        out = []
        remaining = compressed.length
        for vec in compressed.vectors:
            count = min(cpv, remaining)
            out.extend(alphabet[i] for i in self._unpack(vec, base, count))
            remaining -= count
        return "".join(out)

    def char_at(self, compressed: CompressedText, pos: int) -> str:
        """O(1): decode only the vector containing position `pos`."""
        if not 0 <= pos < compressed.length:
            raise IndexError(pos)
        alphabet = compressed.meta["alphabet"]
        base = max(len(alphabet), 2)
        cpv = self.chars_per_vector(base)
        vec = compressed.vectors[pos // cpv]
        indices = self._unpack(vec, base, pos % cpv + 1)
        return alphabet[indices[pos % cpv]]

    def append(self, compressed: CompressedText, more: str) -> CompressedText:
        """O(len(more)): only the trailing partial vector is re-encoded."""
        alphabet = compressed.meta["alphabet"]
        if any(c not in alphabet for c in more):
            # alphabet must grow -> chunk geometry changes -> recompress
            merged = self.decompress(compressed) + more
            return self.compress(merged)
        base = max(len(alphabet), 2)
        cpv = self.chars_per_vector(base)
        index = {c: i for i, c in enumerate(alphabet)}

        vectors = list(compressed.vectors)
        tail_len = compressed.length % cpv
        tail = ""
        if tail_len:
            last = vectors.pop()
            tail = "".join(
                alphabet[i] for i in self._unpack(last, base, tail_len)
            )
        text = tail + more
        for start in range(0, len(text), cpv):
            chunk = [index[c] for c in text[start:start + cpv]]
            vectors.append(self._pack(chunk, base))
        return CompressedText(
            vectors=vectors,
            length=compressed.length + len(more),
            meta=dict(compressed.meta),
        )
