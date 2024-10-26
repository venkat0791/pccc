# pccc

## Parallel-concatenated convolutional code (PCCC)

This crate implements encoding and decoding functionality for a _parallel-concatenated
convolutional code_ (PCCC), commonly referred to as a turbo code. The encoder for such a code
comprises the parallel concatenation of two identical _recursive systematic convolutional_ (RSC)
encoders, separated by an internal interleaver. The decoder is based on iterations between two
corresponding soft-input/soft-output _a posteriori probability_ (APP) decoders, separated by an
interleaver and deinterleaver.

The [`encoder`] and [`decoder`] functions handle PCCC encoding and decoding, respectively,
while the [`Interleaver`] struct models the internal interleaver. The [`Bit`] enum represents
binary symbol values, and the [`DecodingAlgo`] enum lists the supported decoding algorithms for
the constituent RSC code; each variant of the latter holds a [`u32`] representing the desired
number of turbo iterations.

## Examples

```rust
use pccc::{decoder, encoder, Bit, DecodingAlgo, Interleaver};
use Bit::{One, Zero};

// Rate-1/3 PCCC in LTE, 4 information bits
let code_polynomials = [0o13, 0o15];
let interleaver = Interleaver::new(&[3, 0, 1, 2])?;
// Encoding
let info_bits = [One, Zero, Zero, One];
let code_bits = encoder(&info_bits, &interleaver, &code_polynomials)?;
assert_eq!(
    code_bits,
    [
        One, One, One, Zero, One, Zero, Zero, One, Zero, One, Zero, Zero, One, Zero, One, One,
        Zero, Zero, Zero, One, One, One, Zero, Zero
    ]
);
// Decoding
let code_bits_llr = [
    -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
];
let info_bits_hat = decoder(
    &code_bits_llr,
    &interleaver,
    &code_polynomials,
    DecodingAlgo::LogMAP(8),
)?; // Log-MAP decoding with 8 turbo iterations
assert_eq!(info_bits_hat, [One, Zero, Zero, One]);
```

License: MIT
