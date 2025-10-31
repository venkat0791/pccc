//! # Parallel-concatenated convolutional code (PCCC)
//!
//! This crate implements encoding and decoding functionality for a _parallel-concatenated
//! convolutional code_ (PCCC), commonly referred to as a
//! [turbo code](https://en.wikipedia.org/wiki/Turbo_code). An instance of such a code is
//! used for forward error correction in the 4G LTE standard for wireless broadband communication
//! (see [3GPP TS 36.212](https://www.3gpp.org/ftp/Specs/archive/36_series/36.212/)).
//!
//! The encoder for a PCCC comprises the parallel concatenation of two identical _recursive
//! systematic convolutional_ (RSC) encoders, separated by an internal interleaver. The decoder is
//! based on "turbo" iterations between two corresponding soft-input/soft-output _a posteriori
//! probability_ (APP) decoders, separated by an interleaver and deinterleaver.
//!
//! The [`encoder`] and [`decoder`] functions handle PCCC encoding and decoding, respectively,
//! while the [`Interleaver`] struct models the internal interleaver and associated deinterleaver.
//! The [`Bit`] enum represents binary symbol values, and the [`DecodingAlgo`] enum the supported
//! decoding algorithms; each variant of the latter holds a [`u32`] representing the number of
//! turbo iterations. The code below illustrates their usage through a toy example.
//!
//! # Examples
//!
//! ```
//! use pccc::{decoder, encoder, Bit, DecodingAlgo, Interleaver};
//! use Bit::{One, Zero};
//!
//! let code_polynomials = [0o13, 0o15]; // Rate-1/3 PCCC in LTE
//! let interleaver = Interleaver::new(&[3, 0, 1, 2])?; // Interleaver for 4 information bits
//! let decoding_algo = DecodingAlgo::LogMAP(8); // Log-MAP decoding, 8 turbo iterations
//!
//! // Encoding
//! let info_bits = [One, Zero, Zero, One];
//! let code_bits = encoder(&info_bits, &interleaver, &code_polynomials)?;
//! assert_eq!(
//!     code_bits,
//!     [
//!         One, One, One, Zero, One, Zero, Zero, One, Zero, One, Zero, Zero, One, Zero, One, One,
//!         Zero, Zero, Zero, One, One, One, Zero, Zero
//!     ]
//! );
//!
//! // Decoding
//! let code_bits_llr = [
//!     -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
//!     1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
//! ];
//! let info_bits_hat = decoder(
//!     &code_bits_llr,
//!     &interleaver,
//!     &code_polynomials,
//!     decoding_algo,
//! )?;
//! assert_eq!(info_bits_hat, [One, Zero, Zero, One]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The [`lte`] module contains encoder and decoder functions specifically for the rate-1/3 PCCC
//! used in the LTE standard; the block size is restricted to the values in Table 5.1.3-3 of 3GPP
//! TS 36.212 (from 40 to 6144 bits), and the interleaver for each of these block sizes is
//! automatically set to the _quadratic permutation polynomial_ (QPP) interleaver prescribed
//! therein. This module also has functions to evaluate the performance of the LTE PCCC over a
//! BPSK-AWGN channel by simulation. Finally, the [`utils`] module has some useful functions for
//! such a simulation.

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

mod common;
mod interleaver;
pub mod lte;
mod pccc;
mod rsc;
pub mod utils;

pub use common::{Bit, Error};
pub use interleaver::Interleaver;
pub use pccc::{decoder, encoder, DecodingAlgo};
