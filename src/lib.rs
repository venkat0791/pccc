//! This crate implements encoding and decoding functionality for a parallel-concatenated
//! convolutional code (PCCC), commonly referred to as a turbo code. The encoder for such a code
//! comprises the parallel concatenation of two identical recursive systematic convolutional (RSC)
//! encoders, separated by an internal interleaver (the systematic bits from the second encoder are
//! discarded). The decoder is based on iterations between two corresponding soft-input/soft-output
//! BCJR decoders, separated by an interleaver and deinterleaver.

#![warn(
    clippy::complexity,
    clippy::pedantic,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_allocation,
    unused_import_braces,
    unused_qualifications
)]

