//! Encoder and decoder for a parallel concatenated convolutional code

use serde::{Deserialize, Serialize};

use crate::{rsc, utils, Bit, Error, Interleaver};

/// Enumeration of PCCC decoding algorithms
#[derive(Clone, Eq, Hash, PartialEq, Debug, Copy, Deserialize, Serialize)]
pub enum DecodingAlgo {
    /// Log-MAP decoding with given number of turbo iterations
    LogMAP(u32),
    /// Max-Log-MAP decoding with given number of turbo iterations
    MaxLogMAP(u32),
    /// Linear-Log-MAP decoding (Valenti & Sun, 2001) with given number of turbo iterations
    LinearLogMAP(u32),
}

impl DecodingAlgo {
    /// Returns the name of the variant.
    fn name(&self) -> &str {
        match self {
            DecodingAlgo::LogMAP(_) => "Log-MAP",
            DecodingAlgo::MaxLogMAP(_) => "Max-Log-MAP",
            DecodingAlgo::LinearLogMAP(_) => "Linear-Log-MAP",
        }
    }

    /// Returns the number of turbo iterations held in the variant.
    fn num_iter(self) -> u32 {
        match self {
            DecodingAlgo::LogMAP(n)
            | DecodingAlgo::MaxLogMAP(n)
            | DecodingAlgo::LinearLogMAP(n) => n,
        }
    }
}

impl std::fmt::Display for DecodingAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} decoding, {} turbo iterations",
            self.name(),
            self.num_iter()
        )
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
/// Returns an error if the number of information bits is not equal to `interleaver.length` or if
/// `code_polynomials` is invalid. The latter condition holds if the number of code polynomials is
/// less than `2`, if the first code polynomial is either `0` or a power of `2`, or if any
/// subsequent code polynomial is either not in the range `[1, 2^L)` or equals the first code
/// polynomial; here, `L` is the positive integer such that the first code polynomial is in the
/// range `(2^(L-1), 2^L)`.
///
/// # Examples
/// ```
/// use pccc::{encoder, Bit, Interleaver};
/// use Bit::{One, Zero};
///
/// let code_polynomials = [0o13, 0o15]; // Rate-1/3 PCCC in LTE
/// let interleaver = Interleaver::new(&[3, 0, 1, 2])?; // Interleaver for 4 information bits
/// let info_bits = [One, Zero, Zero, One];
/// let code_bits = encoder(&info_bits, &interleaver, &code_polynomials)?;
/// assert_eq!(
///     code_bits,
///     [
///         One, One, One, Zero, One, Zero, Zero, One, Zero, One, Zero, Zero, One, Zero, One, One,
///         Zero, Zero, Zero, One, One, One, Zero, Zero
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
    let num_info_bits = interleaver.length;
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
/// - `code_bits_llr`: Log-likelihood-ratio (LLR) values for the code bits, with positive values
///   indicating that `Zero` is more likely.
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
/// Returns an error if the number of code bit LLR values is incompatible with `interleaver.length`
/// or if `code_polynomials` is invalid. The latter condition holds if the number of code
/// polynomials is less than `2`, if the first code polynomial is either `0` or a power of `2`, or
/// if any subsequent code polynomial is either not in the range `[1, 2^L)` or equals the first
/// code polynomial; here, `L` is the positive integer such that the first code polynomial is in
/// the range `(2^(L-1), 2^L)`.
///
/// # Examples
/// ```
/// use pccc::{decoder, Bit, DecodingAlgo, Interleaver};
/// use Bit::{One, Zero};
///
/// let code_polynomials = [0o13, 0o15]; // Rate-1/3 PCCC in LTE
/// let interleaver = Interleaver::new(&[3, 0, 1, 2])?; // Interleaver for 4 information bits
/// let decoding_algo = DecodingAlgo::LogMAP(8); // Log-MAP decoding, 8 turbo iterations
/// let code_bits_llr = [
///     -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
///     1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
/// ];
/// let info_bits_hat = decoder(
///     &code_bits_llr,
///     &interleaver,
///     &code_polynomials,
///     decoding_algo,
/// )?;
/// assert_eq!(info_bits_hat, [One, Zero, Zero, One]);
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
    Ok(utils::bpsk_slicer(&ws.llr_posterior))
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
            code_bits_llr.len()
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
            &code_bits_llr[i_top + sm.num_output_bits .. i_top + inverse_code_rate],
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

#[cfg(test)]
mod tests_of_functions {
    use float_eq::assert_float_eq;

    use super::*;
    use Bit::{One, Zero};

    #[test]
    fn test_encoder() {
        let code_polynomials = [0o13, 0o15];
        let interleaver = Interleaver::new(&[3, 0, 1, 2]).unwrap();
        // Invalid inputs
        let info_bits = [Zero, One, One];
        assert!(encoder(&info_bits, &interleaver, &code_polynomials).is_err());
        // Valid inputs
        let info_bits = [One, Zero, Zero, One];
        let code_bits = encoder(&info_bits, &interleaver, &code_polynomials).unwrap();
        assert_eq!(
            code_bits,
            [
                One, One, One, Zero, One, Zero, Zero, One, Zero, One, Zero, Zero, One, Zero, One,
                One, Zero, Zero, Zero, One, One, One, Zero, Zero
            ]
        );
    }

    #[test]
    fn test_decoder() {
        let code_polynomials = [0o13, 0o15];
        let interleaver = Interleaver::new(&[3, 0, 1, 2]).unwrap();
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
            -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
            -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
        ];
        let info_bits_hat = decoder(
            &code_bits_llr,
            &interleaver,
            &code_polynomials,
            DecodingAlgo::LinearLogMAP(8),
        )
        .unwrap();
        assert_eq!(info_bits_hat, [One, Zero, Zero, One]);
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
        assert_float_eq!(
            top_code_bits_llr,
            [0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0].to_vec(),
            abs_all <= 1e-8
        );
        assert_float_eq!(
            bottom_code_bits_llr,
            [0.0, 2.0, 9.0, 5.0, 3.0, 8.0, 6.0, 11.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0].to_vec(),
            abs_all <= 1e-8
        );
    }
}
