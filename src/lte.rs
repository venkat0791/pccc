//! Simulator to evaluate performance of LTE rate-1/3 PCCC over BPSK-AWGN channel

use serde::{Deserialize, Serialize};

use crate::{DecodingAlgo, Error, Interleaver};

/// Parameters for parallel-concatenated convolutional code simulation over BPSK-AWGN channel
#[derive(Clone, PartialEq, Debug, Copy, Deserialize, Serialize)]
pub struct SimParams {
    /// Number of information bits per block
    pub num_info_bits_per_block: u32,
    /// Ratio (dB) of symbol energy to noise power spectral density at BPSK-AWGN channel output
    pub es_over_n0_db: f64,
    /// Decoding algorithm to be used
    pub decoding_algo: DecodingAlgo,
    /// Desired minimum number of block errors
    pub num_block_errors_min: u32,
    /// Number of blocks to be transmitted per run
    pub num_blocks_per_run: u32,
    /// Minimum number of runs of blocks to be simulated
    pub num_runs_min: u32,
    /// Maximum number of runs of blocks to be simulated
    pub num_runs_max: u32,
}

/// Returns quadratic permutation polynomial (QPP) interleaver for LTE rate-1/3 PCCC.
///
/// # Parameters
///
/// - `num_info_bits`: Number of information bits. Must be one of the values specified in Table
///   5.1.3-3 of 3GPP TS 36.212.
///
/// # Errors
///
/// Returns an error if `num_info_bits` is invalid.
///
/// # Examples
///
/// ```
/// use pccc::lte;
///
/// let num_info_bits = 40;
/// let interleaver = lte::interleaver(num_info_bits)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn interleaver(num_info_bits: usize) -> Result<Interleaver, Error> {
    let (coeff1, coeff2) = if num_info_bits <= 128 {
        qpp_coefficients_40_to_128_bits(num_info_bits)?
    } else if num_info_bits <= 256 {
        qpp_coefficients_136_to_256_bits(num_info_bits)?
    } else if num_info_bits <= 512 {
        qpp_coefficients_264_to_512_bits(num_info_bits)?
    } else if num_info_bits <= 1024 {
        qpp_coefficients_528_to_1024_bits(num_info_bits)?
    } else if num_info_bits <= 2048 {
        qpp_coefficients_1056_to_2048_bits(num_info_bits)?
    } else if num_info_bits <= 4096 {
        qpp_coefficients_2112_to_4096_bits(num_info_bits)?
    } else {
        qpp_coefficients_4160_to_6144_bits(num_info_bits)?
    };
    let perm: Vec<usize> = (0 .. num_info_bits)
        .map(|out_index| ((coeff1 + coeff2 * out_index) * out_index) % num_info_bits)
        .collect();
    Interleaver::new(&perm)
}

/// Checks validity of simulation parameters.
fn check_sim_params(params: &SimParams) -> Result<(), Error> {
    if params.num_blocks_per_run == 0 {
        return Err(Error::InvalidInput(
            "Number of blocks per run cannot be zero".to_string(),
        ));
    }
    if params.num_runs_min > params.num_runs_max {
        return Err(Error::InvalidInput(format!(
            "Minimum number of runs ({}) exceeds maximum number of runs ({})",
            params.num_runs_min, params.num_runs_max
        )));
    }
    Ok(())
}

/// Returns QPP coefficients for `40 <= num_info_bits <= 128`.
fn qpp_coefficients_40_to_128_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        40 => Ok((3, 10)),
        48 => Ok((7, 12)),
        56 => Ok((19, 42)),
        64 => Ok((7, 16)),
        72 => Ok((7, 18)),
        80 => Ok((11, 20)),
        88 => Ok((5, 22)),
        96 => Ok((11, 24)),
        104 => Ok((7, 26)),
        112 => Ok((41, 84)),
        120 => Ok((103, 90)),
        128 => Ok((15, 32)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns QPP coefficients for `136 <= num_info_bits <= 256`.
fn qpp_coefficients_136_to_256_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        136 => Ok((9, 34)),
        144 => Ok((17, 108)),
        152 => Ok((9, 38)),
        160 => Ok((21, 120)),
        168 => Ok((101, 84)),
        176 => Ok((21, 44)),
        184 => Ok((57, 46)),
        192 => Ok((23, 48)),
        200 => Ok((13, 50)),
        208 => Ok((27, 52)),
        216 => Ok((11, 36)),
        224 => Ok((27, 56)),
        232 => Ok((85, 58)),
        240 => Ok((29, 60)),
        248 => Ok((33, 62)),
        256 => Ok((15, 32)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns QPP coefficients for `264 <= num_info_bits <= 512`.
fn qpp_coefficients_264_to_512_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        264 => Ok((17, 198)),
        272 => Ok((33, 68)),
        280 => Ok((103, 210)),
        288 => Ok((19, 36)),
        296 => Ok((19, 74)),
        304 => Ok((37, 76)),
        312 => Ok((19, 78)),
        320 => Ok((21, 120)),
        328 => Ok((21, 82)),
        336 => Ok((115, 84)),
        344 => Ok((193, 86)),
        352 => Ok((21, 44)),
        360 => Ok((133, 90)),
        368 => Ok((81, 46)),
        376 => Ok((45, 94)),
        384 => Ok((23, 48)),
        392 => Ok((243, 98)),
        400 => Ok((151, 40)),
        408 => Ok((155, 102)),
        416 => Ok((25, 52)),
        424 => Ok((51, 106)),
        432 => Ok((47, 72)),
        440 => Ok((91, 110)),
        448 => Ok((29, 168)),
        456 => Ok((29, 114)),
        464 => Ok((247, 58)),
        472 => Ok((29, 118)),
        480 => Ok((89, 180)),
        488 => Ok((91, 122)),
        496 => Ok((157, 62)),
        504 => Ok((55, 84)),
        512 => Ok((31, 64)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns QPP coefficients for `528 <= num_info_bits <= 1024`.
fn qpp_coefficients_528_to_1024_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        528 => Ok((17, 66)),
        544 => Ok((35, 68)),
        560 => Ok((227, 420)),
        576 => Ok((65, 96)),
        592 => Ok((19, 74)),
        608 => Ok((37, 76)),
        624 => Ok((41, 234)),
        640 => Ok((39, 80)),
        656 => Ok((185, 82)),
        672 => Ok((43, 252)),
        688 => Ok((21, 86)),
        704 => Ok((155, 44)),
        720 => Ok((79, 120)),
        736 => Ok((139, 92)),
        752 => Ok((23, 94)),
        768 => Ok((217, 48)),
        784 => Ok((25, 98)),
        800 => Ok((17, 80)),
        816 => Ok((127, 102)),
        832 => Ok((25, 52)),
        848 => Ok((239, 106)),
        864 => Ok((17, 48)),
        880 => Ok((137, 110)),
        896 => Ok((215, 112)),
        912 => Ok((29, 114)),
        928 => Ok((15, 58)),
        944 => Ok((147, 118)),
        960 => Ok((29, 60)),
        976 => Ok((59, 122)),
        992 => Ok((65, 124)),
        1008 => Ok((55, 84)),
        1024 => Ok((31, 64)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns QPP coefficients for `1056 <= num_info_bits <= 2048`.
fn qpp_coefficients_1056_to_2048_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        1056 => Ok((17, 66)),
        1088 => Ok((171, 204)),
        1120 => Ok((67, 140)),
        1152 => Ok((35, 72)),
        1184 => Ok((19, 74)),
        1216 => Ok((39, 76)),
        1248 => Ok((19, 78)),
        1280 => Ok((199, 240)),
        1312 => Ok((21, 82)),
        1344 => Ok((211, 252)),
        1376 => Ok((21, 86)),
        1408 => Ok((43, 88)),
        1440 => Ok((149, 60)),
        1472 => Ok((45, 92)),
        1504 => Ok((49, 846)),
        1536 => Ok((71, 48)),
        1568 => Ok((13, 28)),
        1600 => Ok((17, 80)),
        1632 => Ok((25, 102)),
        1664 => Ok((183, 104)),
        1696 => Ok((55, 954)),
        1728 => Ok((127, 96)),
        1760 => Ok((27, 110)),
        1792 => Ok((29, 112)),
        1824 => Ok((29, 114)),
        1856 => Ok((57, 116)),
        1888 => Ok((45, 354)),
        1920 => Ok((31, 120)),
        1952 => Ok((59, 610)),
        1984 => Ok((185, 124)),
        2016 => Ok((113, 420)),
        2048 => Ok((31, 64)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns QPP coefficients for `2112 <= num_info_bits <= 4096`.
fn qpp_coefficients_2112_to_4096_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        2112 => Ok((17, 66)),
        2176 => Ok((171, 136)),
        2240 => Ok((209, 420)),
        2304 => Ok((253, 216)),
        2368 => Ok((367, 444)),
        2432 => Ok((265, 456)),
        2496 => Ok((181, 468)),
        2560 => Ok((39, 80)),
        2624 => Ok((27, 164)),
        2688 => Ok((127, 504)),
        2752 => Ok((143, 172)),
        2816 => Ok((43, 88)),
        2880 => Ok((29, 300)),
        2944 => Ok((45, 92)),
        3008 => Ok((157, 188)),
        3072 => Ok((47, 96)),
        3136 => Ok((13, 28)),
        3200 => Ok((111, 240)),
        3264 => Ok((443, 204)),
        3328 => Ok((51, 104)),
        3392 => Ok((51, 212)),
        3456 => Ok((451, 192)),
        3520 => Ok((257, 220)),
        3584 => Ok((57, 336)),
        3648 => Ok((313, 228)),
        3712 => Ok((271, 232)),
        3776 => Ok((179, 236)),
        3840 => Ok((331, 120)),
        3904 => Ok((363, 244)),
        3968 => Ok((375, 248)),
        4032 => Ok((127, 168)),
        4096 => Ok((31, 64)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns QPP coefficients for `4160 <= num_info_bits <= 6144`.
fn qpp_coefficients_4160_to_6144_bits(num_info_bits: usize) -> Result<(usize, usize), Error> {
    match num_info_bits {
        4160 => Ok((33, 130)),
        4224 => Ok((43, 264)),
        4288 => Ok((33, 134)),
        4352 => Ok((477, 408)),
        4416 => Ok((35, 138)),
        4480 => Ok((233, 280)),
        4544 => Ok((357, 142)),
        4608 => Ok((337, 480)),
        4672 => Ok((37, 146)),
        4736 => Ok((71, 444)),
        4800 => Ok((71, 120)),
        4864 => Ok((37, 152)),
        4928 => Ok((39, 462)),
        4992 => Ok((127, 234)),
        5056 => Ok((39, 158)),
        5120 => Ok((39, 80)),
        5184 => Ok((31, 96)),
        5248 => Ok((113, 902)),
        5312 => Ok((41, 166)),
        5376 => Ok((251, 336)),
        5440 => Ok((43, 170)),
        5504 => Ok((21, 86)),
        5568 => Ok((43, 174)),
        5632 => Ok((45, 176)),
        5696 => Ok((45, 178)),
        5760 => Ok((161, 120)),
        5824 => Ok((89, 182)),
        5888 => Ok((323, 184)),
        5952 => Ok((47, 186)),
        6016 => Ok((23, 94)),
        6080 => Ok((47, 190)),
        6144 => Ok((263, 480)),
        _ => Err(Error::InvalidInput(invalid_num_info_bits_message(
            num_info_bits,
        ))),
    }
}

/// Returns error message for invalid number of information bits.
fn invalid_num_info_bits_message(num_info_bits: usize) -> String {
    format!("Number of information bits {num_info_bits} is not in Table 5.1.3-3 of 3GPP TS 36.212")
}

#[cfg(test)]
mod tests_of_functions {
    use super::*;

    #[test]
    fn test_interleaver() {
        // Invalid input
        assert!(interleaver(32).is_err());
        // Valid input
        for num in 0 .. 60 {
            let num_info_bits = 40 + 8 * num;
            assert!(interleaver(num_info_bits).is_ok());
        }
        for num in 0 .. 32 {
            let num_info_bits = 528 + 16 * num;
            assert!(interleaver(num_info_bits).is_ok());
        }
        for num in 0 .. 32 {
            let num_info_bits = 1056 + 32 * num;
            assert!(interleaver(num_info_bits).is_ok());
        }
        for num in 0 .. 64 {
            let num_info_bits = 2112 + 64 * num;
            assert!(interleaver(num_info_bits).is_ok());
        }
        // Invalid input
        assert!(interleaver(6208).is_err());
    }

    #[test]
    fn test_check_sim_params() {
        // Invalid input
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 0,
            num_runs_min: 1,
            num_runs_max: 2,
        };
        assert!(check_sim_params(&params).is_err());
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 1,
            num_runs_min: 2,
            num_runs_max: 1,
        };
        assert!(check_sim_params(&params).is_err());
        // Valid input
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 1,
            num_runs_min: 1,
            num_runs_max: 2,
        };
        assert!(check_sim_params(&params).is_ok());
    }

    #[test]
    fn test_qpp_coefficients_40_to_128_bits() {
        // Invalid input
        assert!(qpp_coefficients_40_to_128_bits(32).is_err());
        assert!(qpp_coefficients_40_to_128_bits(84).is_err());
        assert!(qpp_coefficients_40_to_128_bits(136).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_40_to_128_bits(40).unwrap(), (3, 10));
        assert_eq!(qpp_coefficients_40_to_128_bits(128).unwrap(), (15, 32));
    }

    #[test]
    fn test_qpp_coefficients_136_to_256_bits() {
        // Invalid input
        assert!(qpp_coefficients_136_to_256_bits(128).is_err());
        assert!(qpp_coefficients_136_to_256_bits(196).is_err());
        assert!(qpp_coefficients_136_to_256_bits(264).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_136_to_256_bits(136).unwrap(), (9, 34));
        assert_eq!(qpp_coefficients_136_to_256_bits(256).unwrap(), (15, 32));
    }

    #[test]
    fn test_qpp_coefficients_264_to_512_bits() {
        // Invalid input
        assert!(qpp_coefficients_264_to_512_bits(256).is_err());
        assert!(qpp_coefficients_264_to_512_bits(388).is_err());
        assert!(qpp_coefficients_264_to_512_bits(520).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_264_to_512_bits(264).unwrap(), (17, 198));
        assert_eq!(qpp_coefficients_264_to_512_bits(512).unwrap(), (31, 64));
    }

    #[test]
    fn test_qpp_coefficients_528_to_1024_bits() {
        // Invalid input
        assert!(qpp_coefficients_528_to_1024_bits(512).is_err());
        assert!(qpp_coefficients_528_to_1024_bits(776).is_err());
        assert!(qpp_coefficients_528_to_1024_bits(1040).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_528_to_1024_bits(528).unwrap(), (17, 66));
        assert_eq!(qpp_coefficients_528_to_1024_bits(1024).unwrap(), (31, 64));
    }

    #[test]
    fn test_qpp_coefficients_1056_to_2048_bits() {
        // Invalid input
        assert!(qpp_coefficients_1056_to_2048_bits(1024).is_err());
        assert!(qpp_coefficients_1056_to_2048_bits(1552).is_err());
        assert!(qpp_coefficients_1056_to_2048_bits(2080).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_1056_to_2048_bits(1056).unwrap(), (17, 66));
        assert_eq!(qpp_coefficients_1056_to_2048_bits(2048).unwrap(), (31, 64));
    }

    #[test]
    fn test_qpp_coefficients_2112_to_4096_bits() {
        // Invalid input
        assert!(qpp_coefficients_2112_to_4096_bits(2048).is_err());
        assert!(qpp_coefficients_2112_to_4096_bits(3104).is_err());
        assert!(qpp_coefficients_2112_to_4096_bits(4160).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_2112_to_4096_bits(2112).unwrap(), (17, 66));
        assert_eq!(qpp_coefficients_2112_to_4096_bits(4096).unwrap(), (31, 64));
    }

    #[test]
    fn test_qpp_coefficients_4160_to_6144_bits() {
        // Invalid input
        assert!(qpp_coefficients_4160_to_6144_bits(4096).is_err());
        assert!(qpp_coefficients_4160_to_6144_bits(5152).is_err());
        assert!(qpp_coefficients_4160_to_6144_bits(6208).is_err());
        // Valid input
        assert_eq!(qpp_coefficients_4160_to_6144_bits(4160).unwrap(), (33, 130));
        assert_eq!(
            qpp_coefficients_4160_to_6144_bits(6144).unwrap(),
            (263, 480)
        );
    }
}
