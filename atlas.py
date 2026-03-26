#!/usr/bin/env python3
"""atlas — Adaptive Token-Lexicon Arithmetic Stream compression.

Uses a public BPE tokenizer as shared topology and adaptive PPM-3
for sequential prediction. Beats LZMA by 12-22% on English text
with zero training and zero shared state beyond the public tokenizer.

Usage:
    python atlas.py compress   input.txt output.atlas
    python atlas.py decompress output.atlas restored.txt
    python atlas.py verify     input.txt
    python atlas.py bench      input.txt [--order N]
"""

import struct
import sys
import time
from pathlib import Path

import tiktoken

# --------------------------------------------------------------------------- #
#  Backend selection                                                            #
# --------------------------------------------------------------------------- #

try:
    import atlas_core

    BACKEND = "rust"
except ImportError:
    atlas_core = None
    BACKEND = "python"

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #

MAGIC = b"ATLAS2"  # v2: chunk-parallel format
VERSION = 2
DEFAULT_ORDER = 3

# --------------------------------------------------------------------------- #
#  Tokenizer                                                                    #
# --------------------------------------------------------------------------- #

_encoder_cache = {}


def _get_tokenizer():
    if "enc" not in _encoder_cache:
        _encoder_cache["enc"] = tiktoken.get_encoding("cl100k_base")
    return _encoder_cache["enc"]


# --------------------------------------------------------------------------- #
#  Compress / Decompress                                                        #
# --------------------------------------------------------------------------- #


def compress(
    data: bytes, order: int = DEFAULT_ORDER, verbose: bool = False, parallel: bool = True
) -> bytes:
    """Compress bytes to atlas format."""
    enc = _get_tokenizer()
    tokens = enc.encode_ordinary(data.decode("utf-8", errors="replace"))
    n_tokens = len(tokens)

    if verbose:
        print(f"  Tokenized: {len(data):,} bytes -> {n_tokens:,} tokens ({BACKEND} backend)")

    theory_bits = 0.0
    if atlas_core is not None:
        compressed, theory_bits = atlas_core.compress_token_stream(
            tokens, enc.n_vocab, order, parallel
        )
        compressed = bytes(compressed)
    else:
        compressed = _python_compress(tokens, enc.n_vocab, order)

    if verbose:
        print(f"  Compressed: {len(compressed):,} bytes")
        if theory_bits > 0:
            theory_bytes = theory_bits / 8
            print(f"  Theoretical: {theory_bytes:,.0f} bytes ({theory_bytes / len(data) * 100:.1f}%)")
            print(f"  Arith overhead: {(len(compressed) - theory_bytes) / theory_bytes * 100:.2f}%")

    # Header: magic(6) + version(1) + order(1) + original_size(8) + n_tokens(4)
    header = bytearray()
    header.extend(MAGIC)
    header.append(VERSION)
    header.append(order)
    header.extend(struct.pack("<Q", len(data)))
    header.extend(struct.pack("<I", n_tokens))

    return bytes(header) + compressed


def decompress(data: bytes, verbose: bool = False) -> bytes:
    """Decompress atlas format to bytes."""
    if data[:6] != MAGIC:
        raise ValueError("Not an atlas file (bad magic)")
    version = data[6]
    if version != VERSION:
        raise ValueError(f"Unsupported atlas version: {version}")
    order = data[7]
    original_size = struct.unpack("<Q", data[8:16])[0]
    n_tokens = struct.unpack("<I", data[16:20])[0]
    payload = data[20:]

    if verbose:
        print(f"  Header: version={version}, order={order}, {n_tokens:,} tokens ({BACKEND} backend)")

    enc = _get_tokenizer()

    if atlas_core is not None:
        tokens = atlas_core.decompress_token_stream(
            payload, enc.n_vocab, order, True
        )
    else:
        tokens = _python_decompress(payload, n_tokens, enc.n_vocab, order)

    result = enc.decode(tokens)
    result_bytes = result.encode("utf-8")

    if len(result_bytes) != original_size:
        raise ValueError(
            f"Size mismatch: expected {original_size}, got {len(result_bytes)}"
        )

    if verbose:
        print(f"  Restored: {len(result_bytes):,} bytes")

    return result_bytes


# --------------------------------------------------------------------------- #
#  Python fallback (slow, for systems without Rust)                             #
# --------------------------------------------------------------------------- #

ARITH_BITS = 48
ARITH_FULL = 1 << ARITH_BITS
ARITH_HALF = ARITH_FULL >> 1
ARITH_QTR = ARITH_HALF >> 1
MAX_TOTAL = 65535


class _PPMContext:
    __slots__ = ("counts", "total")

    def __init__(self):
        self.counts = {}
        self.total = 0

    @property
    def escape_count(self):
        return max(len(self.counts), 1)

    @property
    def grand_total(self):
        return self.total + self.escape_count

    def rescale(self):
        new_total = 0
        for sym in self.counts:
            self.counts[sym] = max(self.counts[sym] >> 1, 1)
            new_total += self.counts[sym]
        self.total = new_total

    def get_range(self, symbol):
        grand = self.grand_total
        cum = 0
        for sym in sorted(self.counts):
            cnt = self.counts[sym]
            if sym == symbol:
                return cum, cum + cnt, grand
            cum += cnt
        raise KeyError(symbol)

    def get_escape_range(self):
        grand = self.grand_total
        return grand - self.escape_count, grand, grand

    def decode_symbol(self, target):
        cum = 0
        for sym in sorted(self.counts):
            cnt = self.counts[sym]
            if target < cum + cnt:
                return sym, cum, cum + cnt
            cum += cnt
        grand = self.grand_total
        return None, grand - self.escape_count, grand


class _PPMModel:
    def __init__(self, order, vocab_size):
        self.order = order
        self.vocab_size = vocab_size
        self.contexts = [{} for _ in range(order + 1)]

    def _ctx(self, history, order):
        key = tuple(history[-order:]) if order > 0 else ()
        return self.contexts[order].get(key)

    def update(self, history, symbol):
        for k in range(self.order + 1):
            key = tuple(history[-k:]) if k > 0 else ()
            if key not in self.contexts[k]:
                self.contexts[k][key] = _PPMContext()
            ctx = self.contexts[k][key]
            if symbol not in ctx.counts:
                ctx.counts[symbol] = 0
            ctx.counts[symbol] += 1
            ctx.total += 1
            if ctx.grand_total > MAX_TOTAL:
                ctx.rescale()

    def encode_symbol(self, encoder, history, symbol):
        for order in range(self.order, -1, -1):
            ctx = self._ctx(history, order)
            if ctx is None or not ctx.counts:
                continue
            if symbol in ctx.counts:
                lo, hi, total = ctx.get_range(symbol)
                encoder.encode_range(lo, hi, total)
                self.update(history, symbol)
                return
            else:
                lo, hi, total = ctx.get_escape_range()
                encoder.encode_range(lo, hi, total)
        encoder.encode_range(symbol, symbol + 1, self.vocab_size)
        self.update(history, symbol)

    def decode_symbol(self, decoder, history):
        for order in range(self.order, -1, -1):
            ctx = self._ctx(history, order)
            if ctx is None or not ctx.counts:
                continue
            target = decoder.get_target(ctx.grand_total)
            sym, lo, hi = ctx.decode_symbol(target)
            decoder.narrow(lo, hi, ctx.grand_total)
            if sym is not None:
                self.update(history, sym)
                return sym
        target = decoder.get_target(self.vocab_size)
        decoder.narrow(target, target + 1, self.vocab_size)
        self.update(history, target)
        return target


class _ArithEncoder:
    def __init__(self):
        self.low = 0
        self.high = ARITH_FULL - 1
        self.pending = 0
        self.bits = []

    def encode_range(self, cum_low, cum_high, total):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high // total) - 1
        self.low = self.low + (rng * cum_low // total)
        while True:
            if self.high < ARITH_HALF:
                self.bits.append(0)
                self.bits.extend([1] * self.pending)
                self.pending = 0
            elif self.low >= ARITH_HALF:
                self.bits.append(1)
                self.bits.extend([0] * self.pending)
                self.pending = 0
                self.low -= ARITH_HALF
                self.high -= ARITH_HALF
            elif self.low >= ARITH_QTR and self.high < 3 * ARITH_QTR:
                self.pending += 1
                self.low -= ARITH_QTR
                self.high -= ARITH_QTR
            else:
                break
            self.low = (self.low << 1) & (ARITH_FULL - 1)
            self.high = ((self.high << 1) | 1) & (ARITH_FULL - 1)

    def finish(self):
        self.pending += 1
        if self.low < ARITH_QTR:
            self.bits.append(0)
            self.bits.extend([1] * self.pending)
        else:
            self.bits.append(1)
            self.bits.extend([0] * self.pending)
        while len(self.bits) % 8:
            self.bits.append(0)
        result = bytearray()
        for i in range(0, len(self.bits), 8):
            b = 0
            for j in range(8):
                b = (b << 1) | self.bits[i + j]
            result.append(b)
        return bytes(result)


class _ArithDecoder:
    def __init__(self, data):
        self.data = data
        self.bit_pos = 0
        self.low = 0
        self.high = ARITH_FULL - 1
        self.value = 0
        for _ in range(ARITH_BITS):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self):
        if self.bit_pos >= len(self.data) * 8:
            return 0
        byte_idx = self.bit_pos >> 3
        bit_idx = 7 - (self.bit_pos & 7)
        self.bit_pos += 1
        return (self.data[byte_idx] >> bit_idx) & 1

    def get_target(self, total):
        rng = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // rng

    def narrow(self, cum_low, cum_high, total):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_high // total) - 1
        self.low = self.low + (rng * cum_low // total)
        while True:
            if self.high < ARITH_HALF:
                pass
            elif self.low >= ARITH_HALF:
                self.low -= ARITH_HALF
                self.high -= ARITH_HALF
                self.value -= ARITH_HALF
            elif self.low >= ARITH_QTR and self.high < 3 * ARITH_QTR:
                self.low -= ARITH_QTR
                self.high -= ARITH_QTR
                self.value -= ARITH_QTR
            else:
                break
            self.low = (self.low << 1) & (ARITH_FULL - 1)
            self.high = ((self.high << 1) | 1) & (ARITH_FULL - 1)
            self.value = ((self.value << 1) | self._read_bit()) & (ARITH_FULL - 1)


def _python_compress(tokens, vocab_size, order):
    """Pure Python compression fallback."""
    model = _PPMModel(order, vocab_size)
    encoder = _ArithEncoder()
    history = []
    for tok in tokens:
        model.encode_symbol(encoder, history, tok)
        history.append(tok)
        if len(history) > order:
            history = history[-order:]
    compressed = encoder.finish()
    # Wrap in chunk format for compatibility
    result = bytearray()
    result.extend(struct.pack("<I", 1))  # 1 chunk
    result.extend(struct.pack("<I", len(tokens)))
    result.extend(struct.pack("<I", len(compressed)))
    result.extend(compressed)
    return bytes(result)


def _python_decompress(data, n_tokens, vocab_size, order):
    """Pure Python decompression fallback."""
    # Parse chunk format
    n_chunks = struct.unpack("<I", data[0:4])[0]
    offset = 4
    tokens = []
    for _ in range(n_chunks):
        chunk_n = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        comp_len = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        comp_data = data[offset : offset + comp_len]
        offset += comp_len

        model = _PPMModel(order, vocab_size)
        decoder = _ArithDecoder(comp_data)
        history = []
        for _ in range(chunk_n):
            tok = model.decode_symbol(decoder, history)
            tokens.append(tok)
            history.append(tok)
            if len(history) > order:
                history = history[-order:]
    return tokens


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #


def cmd_compress(args):
    input_path = Path(args[0])
    output_path = Path(args[1]) if len(args) > 1 else input_path.with_suffix(".atlas")
    order = _parse_order(args)

    data = input_path.read_bytes()
    print(f"Input:  {input_path} ({len(data):,} bytes)")

    t0 = time.time()
    compressed = compress(data, order=order, verbose=True)
    elapsed = time.time() - t0

    output_path.write_bytes(compressed)
    ratio = len(compressed) / len(data) * 100
    print(f"Output: {output_path} ({len(compressed):,} bytes, {ratio:.1f}%)")
    print(f"Time:   {elapsed:.1f}s ({len(data) / elapsed / 1024:.0f} KB/s)")


def cmd_decompress(args):
    input_path = Path(args[0])
    output_path = Path(args[1]) if len(args) > 1 else input_path.with_suffix(".txt")

    data = input_path.read_bytes()
    print(f"Input:  {input_path} ({len(data):,} bytes)")

    t0 = time.time()
    restored = decompress(data, verbose=True)
    elapsed = time.time() - t0

    output_path.write_bytes(restored)
    print(f"Output: {output_path} ({len(restored):,} bytes)")
    print(f"Time:   {elapsed:.1f}s")


def cmd_verify(args):
    input_path = Path(args[0])
    order = _parse_order(args)

    data = input_path.read_bytes()
    print(f"Input: {input_path} ({len(data):,} bytes)")

    t0 = time.time()
    compressed = compress(data, order=order, verbose=True)
    t1 = time.time()
    restored = decompress(compressed, verbose=True)
    t2 = time.time()

    ratio = len(compressed) / len(data) * 100
    if data == restored:
        print(f"\nROUND-TRIP VERIFIED: lossless")
        print(f"  Ratio:    {ratio:.1f}% ({len(data):,} -> {len(compressed):,} bytes)")
        print(f"  Compress: {t1 - t0:.1f}s")
        print(f"  Decomp:   {t2 - t1:.1f}s")
    else:
        print(f"\nROUND-TRIP FAILED!")
        for i in range(min(len(data), len(restored))):
            if data[i] != restored[i]:
                print(
                    f"  First diff at byte {i}: "
                    f"original=0x{data[i]:02x} restored=0x{restored[i]:02x}"
                )
                break
        if len(data) != len(restored):
            print(f"  Size: original={len(data)}, restored={len(restored)}")
        sys.exit(1)


def cmd_bench(args):
    import lzma

    input_path = Path(args[0])
    order = _parse_order(args)

    data = input_path.read_bytes()
    print(f"Input: {input_path} ({len(data):,} bytes)")
    print(f"Backend: {BACKEND}\n")

    # LZMA
    t0 = time.time()
    lzma_out = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    t1 = time.time()
    lzma_ratio = len(lzma_out) / len(data) * 100
    print(f"LZMA:  {len(lzma_out):,} bytes ({lzma_ratio:.1f}%) in {t1 - t0:.1f}s")

    # Atlas
    t0 = time.time()
    atlas_out = compress(data, order=order, verbose=False)
    t1 = time.time()
    atlas_ratio = len(atlas_out) / len(data) * 100
    improvement = (1 - len(atlas_out) / len(lzma_out)) * 100
    print(f"Atlas: {len(atlas_out):,} bytes ({atlas_ratio:.1f}%) in {t1 - t0:.1f}s")
    print(f"\nAtlas vs LZMA: {improvement:+.1f}%")

    # Verify
    t0 = time.time()
    restored = decompress(atlas_out, verbose=False)
    t1 = time.time()
    if data == restored:
        print(f"Round-trip: VERIFIED ({t1 - t0:.1f}s)")
    else:
        print("Round-trip: FAILED")
        sys.exit(1)


def _parse_order(args):
    for i, a in enumerate(args):
        if a == "--order" and i + 1 < len(args):
            return int(args[i + 1])
    return DEFAULT_ORDER


COMMANDS = {
    "compress": cmd_compress,
    "decompress": cmd_decompress,
    "verify": cmd_verify,
    "bench": cmd_bench,
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print(f"Backend: {BACKEND}")
        print("Commands:", ", ".join(COMMANDS.keys()))
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print("Commands:", ", ".join(COMMANDS.keys()))
        sys.exit(1)

    COMMANDS[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
