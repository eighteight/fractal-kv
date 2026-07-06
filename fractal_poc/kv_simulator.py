"""Toy KV-cache simulator.

Mimics the three constraints a real LLM KV cache imposes on any
compression scheme, without needing a real LLM:

  1. sequential growth  -- tokens arrive one at a time (add_token)
  2. random access      -- any past token must be retrievable (token_at)
  3. bounded memory     -- memory_bytes() is what we are trying to shrink

The simulator holds the *entire* history only in compressed form, so
correctness of token_at / all_tokens is proof the codec really works.
"""

from __future__ import annotations

from .codec import FractalCodec

SEP = "\x1f"  # unit separator: joins tokens inside the compressed stream


class ToyKVSimulator:
    def __init__(self, codec: FractalCodec):
        self.codec = codec
        self.compressed = None
        self.num_tokens = 0
        self._offsets: list[int] = []  # char offset of each token (tiny index)

    def add_token(self, token: str) -> None:
        piece = token if self.num_tokens == 0 else SEP + token
        if self.compressed is None:
            self._offsets.append(0)
            self.compressed = self.codec.compress(piece)
        else:
            self._offsets.append(self.compressed.length + 1)
            self.compressed = self.codec.append(self.compressed, piece)
        self.num_tokens += 1

    def token_at(self, pos: int) -> str:
        """Random access to one past token via per-character random access."""
        start = self._offsets[pos]
        end = (
            self._offsets[pos + 1] - 1
            if pos + 1 < self.num_tokens
            else self.compressed.length
        )
        return "".join(
            self.codec.char_at(self.compressed, i) for i in range(start, end)
        )

    def all_tokens(self) -> list[str]:
        return self.codec.decompress(self.compressed).split(SEP)

    def memory_bytes(self) -> int:
        payload = self.compressed.memory_bytes() if self.compressed else 0
        return payload + 4 * len(self._offsets)  # offsets as uint32 index

    def raw_bytes(self) -> int:
        """What the uncompressed token-string history would occupy."""
        return self.compressed.length if self.compressed else 0
