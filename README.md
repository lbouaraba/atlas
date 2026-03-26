# atlas

Lossless text compression that beats LZMA by using a public BPE tokenizer as shared structure.

## How it works

1. **Tokenize** the input with [tiktoken](https://github.com/openai/tiktoken) (cl100k_base, 100K vocabulary)
2. **Predict** each token from context using adaptive PPM order-3 (Prediction by Partial Matching)
3. **Encode** predictions with arithmetic coding (62-bit precision)

The tokenizer provides a pre-computed vocabulary of English patterns. The PPM learns token transition statistics as it processes the file. Together they compress better than LZMA because the tokenizer gives a structural head start that dictionary-based compressors must discover on their own.

No training. No shared model. No corpus dependency. Just a public tokenizer + adaptive prediction.

## Results

Tested on [enwik](http://mattmahoney.net/dc/textdata.html) (Wikipedia XML), lossless and verified at every scale:

| Input size | LZMA | atlas | Improvement |
|---|---|---|---|
| 75 KB | 33.2% | **28.3%** | **+14.8%** |
| 1 MB | 29.1% | **26.9%** | **+7.6%** |
| 10 MB | 27.2% | **25.7%** | **+5.6%** |
| 100 MB (enwiki8) | 24.8% | **23.1%** | **+6.9%** |

LZMA tested at maximum compression (`preset=9 | PRESET_EXTREME`).

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

LZMA uses a sliding window dictionary (64MB max). It can only reference patterns within that window. Atlas's PPM has unlimited effective memory — every pattern ever seen contributes to predictions, regardless of distance. For English text, common patterns ("the", "of the", "in the") are learned in the first few thousand tokens and predict correctly for the rest of the file.

The tokenizer amplifies this: instead of predicting one byte at a time (256 possible values), the PPM predicts one token at a time (~14K used values). Each correct prediction saves 4-5 bytes. The tokenizer's vocabulary was optimized for exactly this — maximizing predictability of English text.

## Limitations

- **English text only** (for now). The cl100k_base tokenizer is optimized for English. Other languages or binary data may not benefit.
- **Sequential compression**. The PPM model is inherently sequential — each prediction depends on all previous tokens. Parallel mode (chunked) trades compression for speed.
- **Speed**. Comparable to LZMA at 100MB, but the Rust implementation is young. There's optimization headroom in the data structures and memory layout.

## License

MIT
