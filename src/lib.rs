//! This crate implements encoding and decoding functionality for a parallel-concatenated
//! convolutional code (PCCC), commonly referred to as a turbo code. The encoder for such a code
//! comprises the parallel concatenation of two identical recursive systematic convolutional (RSC)
//! encoders, separated by an internal interleaver (the systematic bits from the second encoder are
//! discarded). The decoder is based on iterations between two corresponding soft-input/soft-output
//! BCJR decoders, separated by an interleaver and deinterleaver.
//!
//! The [`encoder`] and [`decoder`] functions handle PCCC encoding and decoding, respectively,
//! while the [`Interleaver`] struct models the internal interleaver. The [`Bit`] enum represents
//! binary symbol values, and the [`DecodingAlgo`] enum lists the supported decoding algorithms for
//! the constituent RSC code; each variant of the latter holds a [`u32`] representing the desired
//! number of turbo iterations.
//!
//! # Examples
//!
//! ```
//! use pccc::{decoder, encoder, Bit, DecodingAlgo, Interleaver};
//! use Bit::{One, Zero};
//!
//! let code_polynomials = [0o13, 0o15];
//! let interleaver = Interleaver::new(&[0, 3, 1, 2])?;
//! // Encoding
//! let info_bits = [Zero, One, One, Zero];
//! let code_bits = encoder(&info_bits, &interleaver, &code_polynomials)?;
//! assert_eq!(
//!     code_bits,
//!     [
//!         Zero, Zero, Zero, One, One, Zero, One, Zero, One, Zero, Zero, Zero, Zero, Zero, Zero,
//!         One, One, One, One, One, Zero, One, One, One,
//!     ]
//! );
//! // Decoding
//! let code_bits_llr = [
//!     10.0, 10.0, 10.0, -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
//!     10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0,
//! ];
//! let info_bits_hat = decoder(
//!     &code_bits_llr,
//!     &interleaver,
//!     &code_polynomials,
//!     DecodingAlgo::LinearLogMAP(8),
//! )?;
//! assert_eq!(info_bits_hat, [Zero, One, One, Zero]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

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

use std::io;

use rand::{rngs::ThreadRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod lte;
mod rsc;

/// Custom error type
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid input error
    #[error("{0}")]
    InvalidInput(String),
    /// File read/write error
    #[error("{0}")]
    FileReadWriteError(#[from] io::Error),
    /// Serde read/write error
    #[error("{0}")]
    SerdeReadWriteError(#[from] serde_json::Error),
    /// Unknown error
    #[error("Unknown error")]
    Unknown,
}

/// Enumeration of binary symbol values
#[derive(Clone, Eq, PartialEq, Debug, Copy)]
pub enum Bit {
    /// Binary symbol `0`
    Zero = 0,
    /// Binary symbol `1`
    One = 1,
}

/// Enumeration of decoding algorithms, with each variant holding the number of turbo iterations
#[derive(Clone, Eq, PartialEq, Debug, Copy, Deserialize, Serialize)]
pub enum DecodingAlgo {
    /// Max-Log-MAP (lowest complexity, worst performance)
    MaxLogMAP(u32),
    /// Linear-Log-MAP (see Valenti & Sun, 2001)
    LinearLogMAP(u32),
    /// Log-MAP (highest complexity, best performance)
    LogMAP(u32),
}

impl DecodingAlgo {
    /// Returns number of turbo iterations held in the variant.
    fn num_iter(self) -> u32 {
        match self {
            DecodingAlgo::MaxLogMAP(n)
            | DecodingAlgo::LinearLogMAP(n)
            | DecodingAlgo::LogMAP(n) => n,
        }
    }
}

/// Interleaver for sequences of a given length
#[derive(Eq, PartialEq, Debug)]
pub struct Interleaver {
    /// Length of input/output sequence
    length: usize,
    /// Input index for each output index (needed in interleaving)
    all_in_index_given_out_index: Vec<usize>,
    /// Output index for each input index (needed in deinterleaving)
    all_out_index_given_in_index: Vec<usize>,
}

impl Interleaver {
    /// Returns interleaver corresponding to a given permutation.
    ///
    /// # Parameters
    ///
    /// - `perm`: Permutation of integers in `[0, L)` for some positive integer `L`. If the
    ///   interleaver input is the sequence `x[0], x[1], ..., x[L-1]`, then its output is the
    ///   sequence `x[perm[0]], x[perm[1]], ..., x[perm[L-1]]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `perm` is not a permutation of the integers in `[0, L)` for some
    /// positive integer `L`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pccc::Interleaver;
    ///
    /// let perm = [0, 3, 2, 5, 4, 7, 6, 1];
    /// let interleaver = Interleaver::new(&perm)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(perm: &[usize]) -> Result<Self, Error> {
        if perm.is_empty() {
            return Err(Error::InvalidInput(
                "Permutation cannot be empty".to_string(),
            ));
        }
        let perm_vec = perm.to_vec();
        let mut perm_vec_sorted = perm.to_vec();
        perm_vec_sorted.sort_unstable();
        if !perm_vec_sorted.into_iter().eq(0 .. perm_vec.len()) {
            return Err(Error::InvalidInput(format!(
                "Expected permutation of all integers in the range [0, {})",
                perm_vec.len()
            )));
        }
        Ok(Self::from_valid_perm(perm_vec))
    }

    /// Returns random interleaver for sequences of a given length.
    ///
    /// # Parameters
    ///
    /// - `length`: Length of input/output sequence.
    ///
    /// - `rng`: Random number generator to be used.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` is `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pccc::Interleaver;
    ///
    /// let mut rng = rand::thread_rng();
    /// let length = 8;
    /// let interleaver = Interleaver::random(length, &mut rng);
    /// ```
    pub fn random(length: usize, rng: &mut ThreadRng) -> Result<Self, Error> {
        if length == 0 {
            return Err(Error::InvalidInput(
                "Length must be a positive integer".to_string(),
            ));
        }
        let mut perm_vec: Vec<usize> = (0 .. length).collect();
        perm_vec.shuffle(rng);
        Ok(Self::from_valid_perm(perm_vec))
    }

    /// Generates interleaver output given its input.
    ///
    /// # Parameters
    ///
    /// - `input`: Interleaver input.
    ///
    /// - `output`: Buffer for interleaver output (any pre-existing contents will be cleared).
    ///
    /// # Errors
    ///
    /// Returns an error if `input.len()` is not equal to `self.length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pccc::Interleaver;
    ///
    /// let perm = [0, 3, 2, 5, 4, 7, 6, 1];
    /// let interleaver = Interleaver::new(&perm)?;
    /// let input = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    /// let mut output = Vec::new();
    /// interleaver.interleave(&input, &mut output)?;
    /// assert_eq!(output, ['a', 'd', 'c', 'f', 'e', 'h', 'g', 'b']);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn interleave<T: Copy>(&self, input: &[T], output: &mut Vec<T>) -> Result<(), Error> {
        if input.len() != self.length {
            return Err(Error::InvalidInput(format!(
                "Interleaver input has length {} (expected {})",
                input.len(),
                self.length,
            )));
        }
        output.clear();
        for out_index in 0 .. self.length {
            output.push(input[self.all_in_index_given_out_index[out_index]]);
        }
        Ok(())
    }

    /// Generates interleaver input given its output.
    ///
    /// # Parameters
    ///
    /// - `output`: Interleaver output.
    ///
    /// - `input`: Buffer for interleaver input (any pre-existing contents will be cleared).
    ///
    /// # Errors
    ///
    /// Returns an error if `output.len()` is not equal to `self.length`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pccc::Interleaver;
    ///
    /// let perm = [0, 3, 2, 5, 4, 7, 6, 1];
    /// let interleaver = Interleaver::new(&perm)?;
    /// let output = ['a', 'd', 'c', 'f', 'e', 'h', 'g', 'b'];
    /// let mut input = Vec::new();
    /// interleaver.deinterleave(&output, &mut input)?;
    /// assert_eq!(input, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn deinterleave<T: Copy>(&self, output: &[T], input: &mut Vec<T>) -> Result<(), Error> {
        if output.len() != self.length {
            return Err(Error::InvalidInput(format!(
                "Interleaver output has length {} (expected {})",
                output.len(),
                self.length,
            )));
        }
        input.clear();
        for in_index in 0 .. self.length {
            input.push(output[self.all_out_index_given_in_index[in_index]]);
        }
        Ok(())
    }

    /// Returns interleaver corresponding to a valid permutation.
    fn from_valid_perm(perm_vec: Vec<usize>) -> Self {
        let length = perm_vec.len();
        let all_in_index_given_out_index: Vec<usize> = perm_vec;
        let mut all_out_index_given_in_index: Vec<usize> = (0 .. length).collect();
        all_out_index_given_in_index.sort_by_key(|&k| all_in_index_given_out_index[k]);
        Self {
            length,
            all_in_index_given_out_index,
            all_out_index_given_in_index,
        }
    }
}

/// Returns code bits from PCCC encoder for given information bits.
///
/// # Parameters
///
/// - `info_bits`: Information bits to be encoded.
///
/// - `interleaver`: Internal interleaver for the code.
///
/// - `code_polynomials`: Integer representations of the generator polynomials for the code. Must
///   have length `N >= 2` for a PCCC code of rate `1/(2*N-1)`. The first element is taken as the
///   feedback polynomial (this corresponds to the systematic bit), and all subsequent ones as the
///   feedforward polynomials (these correspond to the parity bits from each RSC encoder). For a
///   code of constraint length `L`, the feedback polynomial must be in the range `(2^(L-1), 2^L)`,
///   and each feedforward polynomial must be in the range `[1, 2^L)` and different from the
///   feedback polynomial.
///
/// # Returns
///
/// - `code_bits`: Code bits from the PCCC encoder.
///
/// # Errors
///
/// Returns an error if `info_bits.len()` is not equal to `interleaver.length` or if
/// `code_polynomials` is invalid.
///
/// # Examples
/// ```
/// use pccc::{encoder, Bit, Interleaver};
/// use Bit::{One, Zero};
///
/// let code_polynomials = [0o13, 0o15];
/// let interleaver = Interleaver::new(&[0, 3, 1, 2])?;
/// let info_bits = [Zero, One, One, Zero];
/// let code_bits = encoder(&info_bits, &interleaver, &code_polynomials)?;
/// assert_eq!(
///     code_bits,
///     [
///         Zero, Zero, Zero, One, One, Zero, One, Zero, One, Zero, Zero, Zero, Zero, Zero, Zero,
///         One, One, One, One, One, Zero, One, One, One,
///     ]
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn encoder(
    info_bits: &[Bit],
    interleaver: &Interleaver,
    code_polynomials: &[usize],
) -> Result<Vec<Bit>, Error> {
    let mut sm = rsc::StateMachine::new(code_polynomials)?;
    let num_info_bits = info_bits.len();
    let num_code_bits_per_rsc = (num_info_bits + sm.memory_len) * sm.num_output_bits;
    // Top RSC encoder output
    let mut top_code_bits = Vec::with_capacity(num_code_bits_per_rsc);
    rsc::encode(info_bits, &mut sm, &mut top_code_bits);
    // Bottom RSC encoder output
    let mut bottom_code_bits = Vec::with_capacity(num_code_bits_per_rsc);
    let mut interleaved_info_bits = Vec::with_capacity(num_info_bits);
    interleaver.interleave(info_bits, &mut interleaved_info_bits)?;
    rsc::encode(&interleaved_info_bits, &mut sm, &mut bottom_code_bits);
    // PCCC encoder output
    let mut code_bits = Vec::with_capacity(2 * num_code_bits_per_rsc - num_info_bits);
    let i_first_rsc_tail_code_bit = num_info_bits * sm.num_output_bits;
    // Code bits corresponding to information bits
    for (top_chunk, bottom_chunk) in top_code_bits[.. i_first_rsc_tail_code_bit]
        .chunks_exact(sm.num_output_bits)
        .zip(bottom_code_bits[.. i_first_rsc_tail_code_bit].chunks_exact(sm.num_output_bits))
    {
        code_bits.extend(top_chunk.iter());
        code_bits.extend(bottom_chunk.iter().skip(1));
    }
    // Code bits corresponding to tail bits
    code_bits.extend(top_code_bits[i_first_rsc_tail_code_bit ..].iter());
    code_bits.extend(bottom_code_bits[i_first_rsc_tail_code_bit ..].iter());
    Ok(code_bits)
}

/// Returns information bit decisions from PCCC decoder for given code bit LLR values.
///
/// # Parameters
///
/// - `code_bits_llr`: Log-likelihood-ratio (LLR) values for the code bits.
///
/// - `interleaver`: Internal interleaver for the code.
///
/// - `code_polynomials`: Integer representations of the generator polynomials for the code. Must
///   have length `N >= 2` for a PCCC code of rate `1/(2*N-1)`. The first element is taken as the
///   feedback polynomial (this corresponds to the systematic bit), and all subsequent ones as the
///   feedforward polynomials (these correspond to the parity bits from each RSC encoder). For a
///   code of constraint length `L`, the feedback polynomial must be in the range `(2^(L-1), 2^L)`,
///   and each feedforward polynomial must be in the range `[1, 2^L)` and different from the
///   feedback polynomial.
///
/// - `decoding_algo`: Decoding algorithm to use, and associated number of turbo iterations.
///
/// # Returns
///
/// - `info_bits_hat`: Decisions on the information bits.
///
/// # Errors
///
/// Returns an error if `code_bits_llr.len()` is incompatible with `interleaver.length` or if
/// `code_polynomials` is invalid.
///
/// # Examples
/// ```
/// use pccc::{decoder, Bit, DecodingAlgo, Interleaver};
/// use Bit::{One, Zero};
///
/// let decoding_algo = DecodingAlgo::LinearLogMAP(8);
/// let code_polynomials = [0o13, 0o15];
/// let interleaver = Interleaver::new(&[0, 3, 1, 2])?;
/// let code_bits_llr = [
///     10.0, 10.0, 10.0, -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
///     10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0,
/// ];
/// let info_bits_hat = decoder(
///     &code_bits_llr,
///     &interleaver,
///     &code_polynomials,
///     decoding_algo,
/// )?;
/// assert_eq!(info_bits_hat, [Zero, One, One, Zero]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decoder(
    code_bits_llr: &[f64],
    interleaver: &Interleaver,
    code_polynomials: &[usize],
    decoding_algo: DecodingAlgo,
) -> Result<Vec<Bit>, Error> {
    let mut sm = rsc::StateMachine::new(code_polynomials)?;
    check_decoder_inputs(code_bits_llr, interleaver, &sm)?;
    let num_info_bits = interleaver.length;
    if decoding_algo.num_iter() == 0 {
        return Ok(vec![Bit::Zero; num_info_bits]);
    }
    let mut ws = rsc::DecoderWorkspace::new(sm.num_states, num_info_bits, decoding_algo);
    let (top_code_bits_llr, bottom_code_bits_llr) = bcjr_inputs(code_bits_llr, interleaver, &sm);
    let mut info_bits_llr_prior = vec![0.0; num_info_bits];
    for i_iter in 0 .. decoding_algo.num_iter() {
        // Bottom RSC decoder
        if i_iter > 0 {
            interleaver.interleave(&ws.extrinsic_info, &mut info_bits_llr_prior)?;
        }
        rsc::decode(
            &bottom_code_bits_llr,
            &info_bits_llr_prior,
            &mut sm,
            &mut ws,
        )?;
        // Top RSC decoder
        interleaver.deinterleave(&ws.extrinsic_info, &mut info_bits_llr_prior)?;
        rsc::decode(&top_code_bits_llr, &info_bits_llr_prior, &mut sm, &mut ws)?;
    }
    Ok(bpsk_slicer(&ws.llr_posterior))
}

/// Checks validity of decoder inputs.
fn check_decoder_inputs(
    code_bits_llr: &[f64],
    interleaver: &Interleaver,
    sm: &rsc::StateMachine,
) -> Result<(), Error> {
    let num_info_bits = interleaver.length;
    let num_code_bits_per_rsc = (num_info_bits + sm.memory_len) * sm.num_output_bits;
    let expected_code_bits_llr_len = 2 * num_code_bits_per_rsc - num_info_bits;
    if code_bits_llr.len() == expected_code_bits_llr_len {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "For interleaver of length {}, expected {} code bit LLR values (found {})",
            num_info_bits,
            expected_code_bits_llr_len,
            code_bits_llr.len(),
        )))
    }
}

/// Returns code bit LLR values to be input to top and bottom BCJR decoders.
fn bcjr_inputs(
    code_bits_llr: &[f64],
    interleaver: &Interleaver,
    sm: &rsc::StateMachine,
) -> (Vec<f64>, Vec<f64>) {
    let num_info_bits = interleaver.length;
    let inverse_code_rate = 2 * sm.num_output_bits - 1;
    let num_code_bits_per_rsc = (num_info_bits + sm.memory_len) * sm.num_output_bits;
    let mut top_code_bits_llr = Vec::with_capacity(num_code_bits_per_rsc);
    let mut bottom_code_bits_llr = Vec::with_capacity(num_code_bits_per_rsc);
    // Information bits
    for k in 0 .. num_info_bits {
        let i_top = k * inverse_code_rate;
        top_code_bits_llr.extend_from_slice(&code_bits_llr[i_top .. i_top + sm.num_output_bits]);
        let i_bottom = interleaver.all_in_index_given_out_index[k] * inverse_code_rate;
        bottom_code_bits_llr.push(code_bits_llr[i_bottom]);
        bottom_code_bits_llr.extend_from_slice(
            &code_bits_llr[i_bottom + sm.num_output_bits .. i_bottom + inverse_code_rate],
        );
    }
    // Tail bits
    top_code_bits_llr.extend_from_slice(
        &code_bits_llr[num_info_bits * inverse_code_rate
            .. num_info_bits * inverse_code_rate + sm.memory_len * sm.num_output_bits],
    );
    bottom_code_bits_llr.extend_from_slice(
        &code_bits_llr[num_info_bits * inverse_code_rate + sm.memory_len * sm.num_output_bits ..],
    );
    (top_code_bits_llr, bottom_code_bits_llr)
}

/// Returns BPSK slicer output.
#[must_use]
pub(crate) fn bpsk_slicer(syms: &[f64]) -> Vec<Bit> {
    syms.iter()
        .map(|&x| if x >= 0.0 { Bit::Zero } else { Bit::One })
        .collect()
}

#[cfg(test)]
mod tests_of_interleaver {
    use super::*;

    #[test]
    fn test_new() {
        // Invalid input
        assert!(Interleaver::new(&[]).is_err());
        assert!(Interleaver::new(&[1, 2, 3, 4]).is_err());
        assert!(Interleaver::new(&[0, 1, 2, 4]).is_err());
        assert!(Interleaver::new(&[0, 0, 1, 2]).is_err());
        // Valid input
        let interleaver = Interleaver::new(&[0, 3, 2, 5, 4, 7, 6, 1]).unwrap();
        assert_eq!(interleaver.length, 8);
        assert_eq!(
            interleaver.all_in_index_given_out_index,
            [0, 3, 2, 5, 4, 7, 6, 1]
        );
        assert_eq!(
            interleaver.all_out_index_given_in_index,
            [0, 7, 2, 1, 4, 3, 6, 5]
        );
    }

    #[test]
    fn test_random() {
        let mut rng = rand::thread_rng();
        // Invalid input
        assert!(Interleaver::random(0, &mut rng).is_err());
        // Valid input
        let length = 8;
        let interleaver = Interleaver::random(length, &mut rng).unwrap();
        let mut o2i = interleaver.all_in_index_given_out_index;
        o2i.sort_unstable();
        assert!(o2i == (0 .. length).collect::<Vec<usize>>());
    }

    #[test]
    fn test_interleave() {
        let interleaver = Interleaver::new(&[0, 3, 2, 5, 4, 7, 6, 1]).unwrap();
        let mut output = Vec::new();
        // Invalid input
        let input = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        assert!(interleaver.interleave(&input, &mut output).is_err());
        // Valid input
        let input = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        interleaver.interleave(&input, &mut output).unwrap();
        assert_eq!(output, ['a', 'd', 'c', 'f', 'e', 'h', 'g', 'b']);
    }

    #[test]
    fn test_deinterleave() {
        let interleaver = Interleaver::new(&[0, 3, 2, 5, 4, 7, 6, 1]).unwrap();
        let mut input = Vec::new();
        // Invalid output
        let output = ['a', 'd', 'c', 'f', 'e', 'h', 'g'];
        assert!(interleaver.deinterleave(&output, &mut input).is_err());
        // Valid output
        let output = ['a', 'd', 'c', 'f', 'e', 'h', 'g', 'b'];
        interleaver.deinterleave(&output, &mut input).unwrap();
        assert_eq!(input, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    }

    #[test]
    fn test_from_valid_perm() {
        let interleaver = Interleaver::from_valid_perm(vec![0, 3, 2, 5, 4, 7, 6, 1]);
        assert_eq!(interleaver.length, 8);
        assert_eq!(
            interleaver.all_in_index_given_out_index,
            [0, 3, 2, 5, 4, 7, 6, 1]
        );
        assert_eq!(
            interleaver.all_out_index_given_in_index,
            [0, 7, 2, 1, 4, 3, 6, 5]
        );
    }
}

#[cfg(test)]
mod tests_of_functions {
    use super::*;
    use Bit::{One, Zero};

    #[test]
    fn test_encoder() {
        let code_polynomials = [0o13, 0o15];
        let interleaver = Interleaver::new(&[0, 3, 1, 2]).unwrap();
        // Invalid inputs
        let info_bits = [Zero, One, One];
        assert!(encoder(&info_bits, &interleaver, &code_polynomials).is_err());
        // Valid inputs
        let info_bits = [Zero, One, One, Zero];
        let code_bits = encoder(&info_bits, &interleaver, &code_polynomials).unwrap();
        assert_eq!(
            code_bits,
            [
                Zero, Zero, Zero, One, One, Zero, One, Zero, One, Zero, Zero, Zero, Zero, Zero,
                Zero, One, One, One, One, One, Zero, One, One, One,
            ]
        );
    }

    #[test]
    fn test_decoder() {
        let code_polynomials = [0o13, 0o15];
        let interleaver = Interleaver::new(&[0, 3, 1, 2]).unwrap();
        // Invalid inputs
        let code_bits_llr = [10.0, -10.0, -10.0];
        assert!(decoder(
            &code_bits_llr,
            &interleaver,
            &code_polynomials,
            DecodingAlgo::LinearLogMAP(8),
        )
        .is_err());
        // Valid inputs
        let code_bits_llr = [
            10.0, 10.0, 10.0, -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0,
        ];
        let info_bits_hat = decoder(
            &code_bits_llr,
            &interleaver,
            &code_polynomials,
            DecodingAlgo::LinearLogMAP(8),
        )
        .unwrap();
        assert_eq!(info_bits_hat, [Zero, One, One, Zero]);
    }

    #[test]
    fn test_check_decoder_inputs() {
        let sm = rsc::StateMachine::new(&[0o13, 0o15]).unwrap();
        let interleaver = Interleaver::new(&[0, 3, 1, 2]).unwrap();
        // Invalid inputs
        let code_bits_llr = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        ];
        assert!(check_decoder_inputs(&code_bits_llr, &interleaver, &sm).is_err());
        let code_bits_llr = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ];
        assert!(check_decoder_inputs(&code_bits_llr, &interleaver, &sm).is_err());
        // Valid inputs
        let code_bits_llr = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        ];
        assert!(check_decoder_inputs(&code_bits_llr, &interleaver, &sm).is_ok());
    }

    #[test]
    fn test_bcjr_inputs() {
        let sm = rsc::StateMachine::new(&[0o13, 0o15]).unwrap();
        let interleaver = Interleaver::new(&[0, 3, 1, 2]).unwrap();
        let code_bits_llr = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        ];
        let (top_code_bits_llr, bottom_code_bits_llr) =
            bcjr_inputs(&code_bits_llr, &interleaver, &sm);
        assert_eq!(
            top_code_bits_llr,
            [0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        );
        assert_eq!(
            bottom_code_bits_llr,
            [0.0, 2.0, 9.0, 11.0, 3.0, 5.0, 6.0, 8.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn test_bpsk_slicer() {
        assert_eq!(bpsk_slicer(&[0.0, 0.1, -0.2]), [Zero, Zero, One]);
    }
}
