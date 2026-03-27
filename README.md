# atlas

Lossless text compression that beats LZMA by using a public BPE tokenizer as shared structure.

## The core idea

A BPE tokenizer like tiktoken reduces the effective alphabet from 256 raw bytes to ~14K commonly used tokens. Each token represents 4-5 bytes of text on average. This is a massive alphabet reduction: instead of predicting one byte at a time from 256 possibilities, atlas predicts one token at a time — and each correct prediction compresses 4-5 bytes at once.

The tokenizer is public (both encoder and decoder have it). No training, no shared model, no corpus dependency. The compression comes from the fact that the tokenizer was built from internet-scale data and already knows which byte patterns are common in English text. Atlas exploits this pre-existing structure.

## How it works

1. **Tokenize** the input with [tiktoken](https://github.com/openai/tiktoken) (cl100k_base, 100K vocabulary, ~14K tokens used in practice)
2. **Predict** each token from context using adaptive PPM order-3 (Prediction by Partial Matching)
3. **Encode** predictions with arithmetic coding (62-bit precision, 0.01% overhead)

The PPM learns token transition statistics as it processes the file. The longer the file, the better the predictions — the model never forgets a pattern, unlike LZMA's fixed 64MB sliding window.

## Results

Tested on [enwik8](http://mattmahoney.net/dc/textdata.html) — the first 100MB of English Wikipedia, a standard compression benchmark. All results are lossless and verified via round-trip (compress → decompress → diff against original).

### Compression ratio

| Input size | LZMA | atlas | Improvement |
|---|---|---|---|
| 75 KB | 33.2% | **28.3%** | **+14.8%** |
| 1 MB | 29.1% | **26.9%** | **+7.6%** |
| 10 MB | 27.2% | **25.7%** | **+5.6%** |
| 100 MB (enwik8) | 24.8% | **23.1%** | **+6.9%** |

LZMA tested at maximum compression (`preset=9 | PRESET_EXTREME`).

### Speed

| Input size | LZMA compress | atlas compress | LZMA decompress | atlas decompress |
|---|---|---|---|---|
| 1 MB | 0.2s | 0.4s | — | 0.4s |
| 10 MB | 2.9s | 3.5s | — | 4.7s |
| 100 MB (enwik8) | 49s | 40s | — | 49s |

Atlas is slightly slower than LZMA at small scales but **faster at 100MB** (40s vs 49s compress). The Rust implementation is 2 days old — there is significant optimization headroom in the data structures and memory layout. LZMA has had 30 years of engineering.

## Install

Requires Python 3.9+ and Rust (for the native backend).

```bash
pip install tiktoken
cd atlas_core && maturin develop --release
```

Falls back to a pure Python implementation if the Rust extension isn't available (much slower).

## Usage

```bash
# Compress
python atlas.py compress input.txt output.atlas

# Decompress
python atlas.py decompress output.atlas restored.txt

# Verify round-trip (compress + decompress + diff)
python atlas.py verify input.txt

# Benchmark against LZMA
python atlas.py bench input.txt
```

### Python API

```python
from atlas import compress, decompress

data = open("paper.txt", "rb").read()
compressed = compress(data)
restored = decompress(compressed)
assert data == restored
```

## Architecture

```
input bytes
    |
    v
tiktoken tokenizer (public, 100K vocab)
    |
    v
adaptive PPM-3 (learns token transitions on the fly)
    |
    v
arithmetic coder (62-bit precision, 0.01% overhead)
    |
    v
.atlas file
```

The PPM model starts empty and learns from the data as it encodes. The decoder builds an identical model from the decoded tokens, so no model needs to be transmitted. The tokenizer vocabulary is public and identical on both sides.

## Why it works

Two advantages compound:

**1. Token-level alphabet reduction.** Raw bytes have 256 possible values per position. Tokens have ~14K used values, but each token covers 4-5 bytes. The information density per prediction is much higher — one correct token prediction compresses 4-5 bytes at once. The tokenizer's vocabulary was optimized on internet-scale text to maximize this effect.

**2. Unlimited context window.** LZMA uses a sliding window dictionary (64MB max). It must rediscover patterns that fall outside its window. Atlas's PPM has unlimited effective memory — its model contains statistics from the entire file history. Every pattern ever seen contributes to predictions, regardless of distance. The PPM never forgets.

For English text, common patterns ("the", "of the", "in the") are learned in the first few thousand tokens and predict correctly for the rest of the file. LZMA can only match these patterns if they're within its 64MB window.

## Limitations

- **English text only** (for now). The cl100k_base tokenizer is optimized for English. Other languages or binary data may not benefit.
- **Sequential compression**. The PPM model is inherently sequential — each prediction depends on all previous tokens. Parallel mode (chunked) trades compression for speed.
- **Speed**. Comparable to LZMA at 100MB, but the Rust implementation is young. There's optimization headroom in the data structures and memory layout.

## License

MIT
