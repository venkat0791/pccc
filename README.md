# pccc

This crate implements encoding and decoding functionality for a parallel-concatenated
convolutional code (PCCC), commonly referred to as a turbo code. The encoder for such a code
comprises the parallel concatenation of two identical recursive systematic convolutional (RSC)
encoders, separated by an internal interleaver (the systematic bits from the second encoder are
discarded). The decoder is based on iterations between two corresponding soft-input/soft-output
BCJR decoders, separated by an interleaver and deinterleaver.

The [`encoder`] and [`decoder`] functions handle PCCC encoding and decoding, respectively,
while the [`Interleaver`] struct models the internal interleaver. The [`Bit`] enum represents
binary symbol values, and the [`DecodingAlgo`] enum lists the supported decoding algorithms for
the constituent RSC code; each variant of the latter holds a [`u32`] representing the desired
number of turbo iterations.

## Examples

```rust
use pccc::{decoder, encoder, Bit, DecodingAlgo, Interleaver};
use Bit::{One, Zero};

let code_polynomials = [0o13, 0o15];
let interleaver = Interleaver::new(&[0, 3, 1, 2])?;
// Encoding
let info_bits = [Zero, One, One, Zero];
let code_bits = encoder(&info_bits, &interleaver, &code_polynomials)?;
assert_eq!(
    code_bits,
    [
        Zero, Zero, Zero, One, One, Zero, One, Zero, One, Zero, Zero, Zero, Zero, Zero, Zero,
        One, One, One, One, One, Zero, One, One, One,
    ]
);
// Decoding
let code_bits_llr = [
    10.0, 10.0, 10.0, -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0,
];
let info_bits_hat = decoder(
    &code_bits_llr,
    &interleaver,
    &code_polynomials,
    DecodingAlgo::LinearLogMAP(8),
)?;
assert_eq!(info_bits_hat, [Zero, One, One, Zero]);
```
