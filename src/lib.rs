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

use rand::{rngs::ThreadRng, seq::SliceRandom};
use thiserror::Error;

mod rsc;

/// Custom error type
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid input error
    #[error("{0}")]
    InvalidInput(String),
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
#[derive(Clone, Eq, PartialEq, Debug, Copy)]
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
