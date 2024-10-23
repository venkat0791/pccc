//! Collection of useful functions for simulating code performance
//!
//! # Examples
//!
//! ```
//! use pccc::utils;
//!
//! let mut rng = rand::thread_rng();
//! let num_bits = 40;
//! let es_over_n0_db = 10.0;
//! let bits = utils::random_bits(num_bits, &mut rng);
//! let bits_llr = utils::bpsk_awgn_channel(&bits, es_over_n0_db, &mut rng);
//! let bits_hat = utils::bpsk_slicer(&bits_llr);
//! let err_count = utils::error_count(&bits_hat, &bits);
//! ```

use crate::Bit;
use rand::{prelude::ThreadRng, Rng};
use rand_distr::StandardNormal;

/// Returns given number of random bits.
///
/// # Parameters
///
/// - `num_bits`: Number of random bits to be generated.
///
/// - `rng`: Random number generator to be used.
///
/// # Returns
///
/// - `bits`: Random bits.
pub fn random_bits(num_bits: usize, rng: &mut ThreadRng) -> Vec<Bit> {
    (0 .. num_bits)
        .map(|_| {
            if rng.gen_bool(0.5) {
                Bit::One
            } else {
                Bit::Zero
            }
        })
        .collect()
}

/// Returns LLR values at BPSK-AWGN channel output corresponding to given input bits.
///
/// # Parameters
///
/// - `bits`: Bits to be transmitted over the BPSK-AWGN channel.
///
/// - `es_over_n0_db`: Ratio (dB) of symbol energy to noise power spectral density at the BPSK-AWGN
///   channel output (if the BPSK symbols are `+1.0` and `-1.0`, then the noise variance is
///   `0.5 / 10f64.powf(0.1 * es_over_n0_db)`).
///
/// - `rng`: Random number generator to be used.
///
/// # Returns
///
/// - `bits_llr`: Log-likelihood-ratio (LLR) values at the BPSK-AWGN channel output corresponding
///   to the transmitted bits.
pub fn bpsk_awgn_channel(bits: &[Bit], es_over_n0_db: f64, rng: &mut ThreadRng) -> Vec<f64> {
    let es_over_n0 = 10f64.powf(0.1 * es_over_n0_db);
    let noise_var = 0.5 / es_over_n0;
    bits.iter()
        .map(|b| match b {
            Bit::Zero => 1f64,
            Bit::One => -1f64,
        })
        .map(|x| 4.0 * es_over_n0 * (x + noise_var.sqrt() * rng.sample::<f64, _>(StandardNormal)))
        .collect()
}

/// Returns BPSK slicer output.
///
/// # Parameters
///
/// - `syms`: Symbols to be sliced.
///
/// # Returns
///
/// - `bits_hat`: Bits obtained by slicing the given symbols.
#[must_use]
pub fn bpsk_slicer(syms: &[f64]) -> Vec<Bit> {
    syms.iter()
        .map(|&x| if x >= 0.0 { Bit::Zero } else { Bit::One })
        .collect()
}

/// Returns number of errors in a sequence with respect to a reference sequence.
///
/// # Parameters
///
/// - `seq`: Sequence in which errors must be counted.
///
/// - `ref_seq`: Reference sequence to which the given sequence is compared.
///
/// # Returns
///
/// - `err_count`: Number of positions in which the two sequences differ. If they are of different
///   lengths, then the longer sequence is effectively truncated to the length of the shorter one.
pub fn error_count<T: PartialEq>(seq: &[T], ref_seq: &[T]) -> usize {
    ref_seq
        .iter()
        .zip(seq.iter())
        .filter(|&(x, y)| x != y)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use Bit::{One, Zero};

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn test_random_bits() {
        let mut rng = rand::thread_rng();
        let num_bits = 0;
        assert!(random_bits(num_bits, &mut rng).is_empty());
        let num_bits = 10000;
        let bits = random_bits(num_bits, &mut rng);
        let num_zeros = bits.iter().filter(|&b| *b == Zero).count();
        let num_ones = bits.iter().filter(|&b| *b == One).count();
        assert!(num_zeros > 9 * num_bits / 20 && num_ones > 9 * num_bits / 20);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    fn test_bpsk_awgn_channel() {
        let mut rng = rand::thread_rng();
        assert!(bpsk_awgn_channel(&random_bits(0, &mut rng), 0.0, &mut rng).is_empty());
        let es_over_n0_db = 20f64;
        let num_bits = 10000;
        let bits = random_bits(num_bits, &mut rng);
        let bits_llr = bpsk_awgn_channel(&bits, es_over_n0_db, &mut rng);
        let es_over_n0 = 10f64.powf(0.1 * es_over_n0_db);
        let noise_var_est = bits_llr
            .iter()
            .zip(bits)
            .map(|(y, b)| match b {
                Zero => y - 4.0 * es_over_n0,
                One => y + 4.0 * es_over_n0,
            })
            .map(|x| x * x)
            .sum::<f64>()
            / f64::from(num_bits as u32);
        assert!(noise_var_est > 7.2 * es_over_n0 && noise_var_est < 8.8 * es_over_n0);
    }

    #[test]
    fn test_bpsk_slicer() {
        assert!(bpsk_slicer(&[]).is_empty());
        assert_eq!(bpsk_slicer(&[0.0, 0.01, -0.01]), [Zero, Zero, One]);
    }

    #[test]
    fn test_error_count() {
        assert_eq!(error_count(&[], &[One, Zero]), 0);
        assert_eq!(error_count(&[One, Zero], &[]), 0);
        // Longer `seq`
        let ref_seq = [One, Zero, Zero, One, One, One, Zero, Zero];
        let seq = [One, One, Zero, Zero, One, One, Zero, Zero, Zero, One];
        assert_eq!(error_count(&seq, &ref_seq), 2);
        // Shorter `seq`
        let ref_seq = [One, Zero, Zero, One, One, One, Zero, Zero, Zero, One];
        let seq = [One, One, Zero, Zero, One, One, Zero, Zero];
        assert_eq!(error_count(&seq, &ref_seq), 2);
    }
}
