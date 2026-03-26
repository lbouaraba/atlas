use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::collections::HashMap;

// =========================================================================
//  Constants
// =========================================================================

const ARITH_BITS: u64 = 62;
const ARITH_FULL: u64 = 1u64 << ARITH_BITS;
const ARITH_HALF: u64 = ARITH_FULL >> 1;
const ARITH_QTR: u64 = ARITH_HALF >> 1;
const MAX_TOTAL: u32 = u32::MAX;

// =========================================================================
//  PPM Context — with prefix sums for O(1) range lookup
// =========================================================================

struct PPMContext {
    symbols: Vec<u32>,    // sorted symbol IDs
    counts: Vec<u32>,     // counts[i] = count for symbols[i]
    total: u32,
}

impl PPMContext {
    fn new() -> Self {
        PPMContext {
            symbols: Vec::new(),
            counts: Vec::new(),
            total: 0,
        }
    }

    #[inline]
    fn escape_count(&self) -> u32 {
        std::cmp::max(self.symbols.len() as u32, 1)
    }

    #[inline]
    fn grand_total(&self) -> u32 {
        self.total + self.escape_count()
    }

    #[inline]
    fn find_symbol(&self, symbol: u32) -> Option<usize> {
        self.symbols.binary_search(&symbol).ok()
    }

    fn add_symbol(&mut self, symbol: u32) {
        match self.symbols.binary_search(&symbol) {
            Ok(idx) => {
                self.counts[idx] += 1;
            }
            Err(idx) => {
                self.symbols.insert(idx, symbol);
                self.counts.insert(idx, 1);
            }
        }
        self.total += 1;
    }

    #[inline]
    fn get_range(&self, sym_idx: usize) -> (u32, u32, u32) {
        let grand = self.grand_total();
        let mut cum = 0u32;
        for i in 0..sym_idx {
            cum += self.counts[i];
        }
        (cum, cum + self.counts[sym_idx], grand)
    }

    #[inline]
    fn get_escape_range(&self) -> (u32, u32, u32) {
        let grand = self.grand_total();
        let esc = self.escape_count();
        (grand - esc, grand, grand)
    }

    fn decode_symbol(&self, target: u32) -> (Option<u32>, u32, u32) {
        let mut cum = 0u32;
        for i in 0..self.symbols.len() {
            if target < cum + self.counts[i] {
                return (Some(self.symbols[i]), cum, cum + self.counts[i]);
            }
            cum += self.counts[i];
        }
        let grand = self.grand_total();
        let esc = self.escape_count();
        (None, grand - esc, grand)
    }
}

// =========================================================================
//  Context key — fixed-size array to avoid Vec allocation
// =========================================================================

#[derive(Hash, Eq, PartialEq, Clone)]
struct CtxKey {
    data: [u32; 8], // supports up to order 8
    len: u8,
}

impl CtxKey {
    #[inline]
    fn from_history(history: &[u32], order: usize) -> Self {
        let mut key = CtxKey {
            data: [0; 8],
            len: 0,
        };
        if order > 0 && !history.is_empty() {
            let start = if history.len() >= order {
                history.len() - order
            } else {
                0
            };
            let slice = &history[start..];
            key.len = slice.len() as u8;
            key.data[..slice.len()].copy_from_slice(slice);
        }
        key
    }
}

// =========================================================================
//  PPM Model
// =========================================================================

struct PPMModel {
    order: usize,
    vocab_size: u32,
    contexts: Vec<HashMap<CtxKey, PPMContext>>,
}

impl PPMModel {
    fn new(order: usize, vocab_size: u32) -> Self {
        let mut contexts = Vec::with_capacity(order + 1);
        for _ in 0..=order {
            contexts.push(HashMap::new());
        }
        PPMModel {
            order,
            vocab_size,
            contexts,
        }
    }

    fn update(&mut self, history: &[u32], symbol: u32) {
        for k in 0..=self.order {
            let key = CtxKey::from_history(history, k);
            let ctx = self.contexts[k].entry(key).or_insert_with(PPMContext::new);
            ctx.add_symbol(symbol);
            if ctx.grand_total() > MAX_TOTAL {
                // With u32::MAX this never triggers, but keep for safety
                break;
            }
        }
    }

    fn encode_symbol(&mut self, encoder: &mut ArithEncoder, history: &[u32], symbol: u32, theory_bits: &mut f64) {
        // Standard PPMC with cascading escapes — this is optimal and decodable.
        for order in (0..=self.order).rev() {
            let key = CtxKey::from_history(history, order);
            if let Some(ctx) = self.contexts[order].get(&key) {
                if ctx.symbols.is_empty() {
                    continue;
                }
                if let Some(idx) = ctx.find_symbol(symbol) {
                    let (lo, hi, total) = ctx.get_range(idx);
                    *theory_bits += -((hi - lo) as f64 / total as f64).log2();
                    encoder.encode_range(lo as u64, hi as u64, total as u64);
                    self.update(history, symbol);
                    return;
                } else {
                    let (lo, hi, total) = ctx.get_escape_range();
                    *theory_bits += -((hi - lo) as f64 / total as f64).log2();
                    encoder.encode_range(lo as u64, hi as u64, total as u64);
                }
            }
        }
        *theory_bits += (self.vocab_size as f64).log2();
        encoder.encode_range(symbol as u64, symbol as u64 + 1, self.vocab_size as u64);
        self.update(history, symbol);
    }

    fn decode_symbol(&mut self, decoder: &mut ArithDecoder, history: &[u32]) -> u32 {
        // Standard PPMC cascading — mirrors encoder exactly.
        for order in (0..=self.order).rev() {
            let key = CtxKey::from_history(history, order);
            if let Some(ctx) = self.contexts[order].get(&key) {
                if ctx.symbols.is_empty() {
                    continue;
                }
                let grand = ctx.grand_total();
                let target = decoder.get_target(grand as u64);
                let (sym, lo, hi) = ctx.decode_symbol(target as u32);
                decoder.narrow(lo as u64, hi as u64, grand as u64);
                if let Some(s) = sym {
                    self.update(history, s);
                    return s;
                }
            }
        }
        let target = decoder.get_target(self.vocab_size as u64);
        let sym = target as u32;
        decoder.narrow(sym as u64, sym as u64 + 1, self.vocab_size as u64);
        self.update(history, sym);
        sym
    }
}

// =========================================================================
//  Arithmetic Encoder — batched bit output
// =========================================================================

struct ArithEncoder {
    low: u64,
    high: u64,
    pending: u64,
    output: Vec<u8>,
    bit_buf: u64,
    bit_count: u32,
}

impl ArithEncoder {
    fn new() -> Self {
        ArithEncoder {
            low: 0,
            high: ARITH_FULL - 1,
            pending: 0,
            output: Vec::with_capacity(1 << 20),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    #[inline]
    fn write_bit(&mut self, bit: u8) {
        self.bit_buf = (self.bit_buf << 1) | bit as u64;
        self.bit_count += 1;
        if self.bit_count == 64 {
            self.flush_bits();
        }
    }

    #[inline]
    fn flush_bits(&mut self) {
        if self.bit_count == 0 {
            return;
        }
        // Left-align remaining bits
        let shift = 64 - self.bit_count;
        let aligned = self.bit_buf << shift;
        let bytes_needed = (self.bit_count + 7) / 8;
        for i in 0..bytes_needed {
            self.output.push((aligned >> (56 - i * 8)) as u8);
        }
        self.bit_buf = 0;
        self.bit_count = 0;
    }

    #[inline]
    fn write_bit_plus_pending(&mut self, bit: u8) {
        self.write_bit(bit);
        let opposite = bit ^ 1;
        for _ in 0..self.pending {
            self.write_bit(opposite);
        }
        self.pending = 0;
    }

    #[inline]
    fn encode_range(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high =
            self.low + ((range as u128 * cum_high as u128 / total as u128) as u64) - 1;
        self.low = self.low + ((range as u128 * cum_low as u128 / total as u128) as u64);
        self.renormalize();
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < ARITH_HALF {
                self.write_bit_plus_pending(0);
            } else if self.low >= ARITH_HALF {
                self.write_bit_plus_pending(1);
                self.low -= ARITH_HALF;
                self.high -= ARITH_HALF;
            } else if self.low >= ARITH_QTR && self.high < 3 * ARITH_QTR {
                self.pending += 1;
                self.low -= ARITH_QTR;
                self.high -= ARITH_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.low &= ARITH_FULL - 1;
            self.high &= ARITH_FULL - 1;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < ARITH_QTR {
            self.write_bit_plus_pending(0);
        } else {
            self.write_bit_plus_pending(1);
        }
        // Pad to byte boundary
        while self.bit_count % 8 != 0 {
            self.write_bit(0);
        }
        self.flush_bits();
        self.output
    }
}

// =========================================================================
//  Arithmetic Decoder
// =========================================================================

struct ArithDecoder {
    data: Vec<u8>,
    bit_pos: usize,
    low: u64,
    high: u64,
    value: u64,
}

impl ArithDecoder {
    fn new(data: Vec<u8>) -> Self {
        let mut dec = ArithDecoder {
            data,
            bit_pos: 0,
            low: 0,
            high: ARITH_FULL - 1,
            value: 0,
        };
        for _ in 0..ARITH_BITS {
            dec.value = (dec.value << 1) | dec.read_bit() as u64;
        }
        dec
    }

    #[inline]
    fn read_bit(&mut self) -> u8 {
        if self.bit_pos >= self.data.len() * 8 {
            return 0;
        }
        let byte_idx = self.bit_pos >> 3;
        let bit_idx = 7 - (self.bit_pos & 7);
        self.bit_pos += 1;
        (self.data[byte_idx] >> bit_idx) & 1
    }

    #[inline]
    fn get_target(&self, total: u64) -> u64 {
        let range = self.high - self.low + 1;
        (((self.value - self.low + 1) as u128 * total as u128 - 1) / range as u128) as u64
    }

    #[inline]
    fn narrow(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high =
            self.low + ((range as u128 * cum_high as u128 / total as u128) as u64) - 1;
        self.low = self.low + ((range as u128 * cum_low as u128 / total as u128) as u64);
        self.renormalize();
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < ARITH_HALF {
                // pass
            } else if self.low >= ARITH_HALF {
                self.low -= ARITH_HALF;
                self.high -= ARITH_HALF;
                self.value -= ARITH_HALF;
            } else if self.low >= ARITH_QTR && self.high < 3 * ARITH_QTR {
                self.low -= ARITH_QTR;
                self.high -= ARITH_QTR;
                self.value -= ARITH_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | self.read_bit() as u64;
            self.low &= ARITH_FULL - 1;
            self.high &= ARITH_FULL - 1;
            self.value &= ARITH_FULL - 1;
        }
    }
}

// =========================================================================
//  Core compress/decompress
// =========================================================================

fn compress_tokens(tokens: &[u32], vocab_size: u32, order: usize) -> (Vec<u8>, f64) {
    let mut model = PPMModel::new(order, vocab_size);
    let mut encoder = ArithEncoder::new();
    let mut history: Vec<u32> = Vec::with_capacity(order + 1);
    let mut theory_bits: f64 = 0.0;

    for &tok in tokens {
        model.encode_symbol(&mut encoder, &history, tok, &mut theory_bits);
        history.push(tok);
        if history.len() > order {
            history.remove(0);
        }
    }

    (encoder.finish(), theory_bits)
}

fn decompress_tokens(data: &[u8], n_tokens: usize, vocab_size: u32, order: usize) -> Vec<u32> {
    let mut model = PPMModel::new(order, vocab_size);
    let mut decoder = ArithDecoder::new(data.to_vec());
    let mut history: Vec<u32> = Vec::with_capacity(order + 1);
    let mut tokens = Vec::with_capacity(n_tokens);

    for _ in 0..n_tokens {
        let tok = model.decode_symbol(&mut decoder, &history);
        tokens.push(tok);
        history.push(tok);
        if history.len() > order {
            history.remove(0);
        }
    }

    tokens
}

// =========================================================================
//  Chunk-parallel
// =========================================================================

const CHUNK_SIZE: usize = 65536;

fn compress_chunks_parallel(tokens: &[u32], vocab_size: u32, order: usize) -> (Vec<u8>, f64) {
    if tokens.len() <= CHUNK_SIZE {
        let (compressed, theory) = compress_tokens(tokens, vocab_size, order);
        let mut result = Vec::new();
        result.extend_from_slice(&1u32.to_le_bytes());
        result.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        result.extend_from_slice(&compressed);
        return (result, theory);
    }

    let chunks: Vec<&[u32]> = tokens.chunks(CHUNK_SIZE).collect();
    let n_chunks = chunks.len();

    let compressed_chunks: Vec<(Vec<u8>, f64)> = chunks
        .par_iter()
        .map(|chunk| compress_tokens(chunk, vocab_size, order))
        .collect();

    let mut result = Vec::new();
    let mut total_theory = 0.0f64;
    result.extend_from_slice(&(n_chunks as u32).to_le_bytes());
    for (i, (compressed, theory)) in compressed_chunks.iter().enumerate() {
        let n_tok = chunks[i].len() as u32;
        result.extend_from_slice(&n_tok.to_le_bytes());
        result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        result.extend_from_slice(compressed);
        total_theory += theory;
    }
    (result, total_theory)
}

fn decompress_chunks_parallel(data: &[u8], vocab_size: u32, order: usize) -> Vec<u32> {
    let n_chunks = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;

    let mut offset = 4;
    let mut chunk_info: Vec<(usize, Vec<u8>)> = Vec::new();
    for _ in 0..n_chunks {
        let n_tok = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let comp_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let comp_data = data[offset..offset + comp_len].to_vec();
        offset += comp_len;
        chunk_info.push((n_tok, comp_data));
    }

    let decoded_chunks: Vec<Vec<u32>> = chunk_info
        .par_iter()
        .map(|(n_tok, comp_data)| decompress_tokens(comp_data, *n_tok, vocab_size, order))
        .collect();

    let mut result = Vec::new();
    for chunk in decoded_chunks {
        result.extend_from_slice(&chunk);
    }
    result
}

// =========================================================================
//  Python bindings
// =========================================================================

#[pyfunction]
fn compress_token_stream(
    py: Python<'_>,
    tokens: Vec<u32>,
    vocab_size: u32,
    order: usize,
    parallel: bool,
) -> (Vec<u8>, f64) {
    if parallel {
        py.allow_threads(|| compress_chunks_parallel(&tokens, vocab_size, order))
    } else {
        let (compressed, theory) = py.allow_threads(|| compress_tokens(&tokens, vocab_size, order));
        let mut result = Vec::new();
        result.extend_from_slice(&1u32.to_le_bytes());
        result.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        result.extend_from_slice(&compressed);
        (result, theory)
    }
}

#[pyfunction]
fn decompress_token_stream(
    py: Python<'_>,
    data: &[u8],
    vocab_size: u32,
    order: usize,
    _parallel: bool,
) -> Vec<u32> {
    py.allow_threads(|| decompress_chunks_parallel(data, vocab_size, order))
}

#[pymodule]
fn atlas_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_token_stream, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_token_stream, m)?)?;
    Ok(())
}
