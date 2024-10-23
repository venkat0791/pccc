//! # Encoder and decoder for the rate-1/3 PCCC in LTE
//!
//! The encoder and decoder for the LTE PCCC are implemented in [`encoder`] and [`decoder`],
//! respectively. The [`bpsk_awgn_sim`] function simulates the performance of this code over a
//! BPSK-AWGN channel. The parameters of the simulation and the results from it are captured in the
//! [`SimParams`] and [`SimResults`] structs, respectively. Finally, the [`run_bpsk_awgn_sims`]
//! function can run multiple such simulations and save the results to a JSON file.
//!
//! # Examples
//!
//! This example shows how to encode 40 information bits with the LTE PCCC, pass the resulting code
//! bits through a BPSK-AWGN channel at an SNR of -3 dB, and decode the resulting channel output
//! with 8 iterations of the Log-MAP algorithm:
//! ```
//! use pccc::{lte, DecodingAlgo, utils};
//! let mut rng = rand::thread_rng();
//! let num_info_bits = 40;
//! let es_over_n0_db = -3.0;
//! let info_bits = utils::random_bits(num_info_bits, &mut rng);
//! let code_bits = lte::encoder(&info_bits)?;
//! let code_bits_llr = utils::bpsk_awgn_channel(&code_bits, es_over_n0_db, &mut rng);
//! let info_bits_hat = lte::decoder(&code_bits_llr, DecodingAlgo::LogMAP(8))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;

use crate::{utils, Bit, DecodingAlgo, Error, Interleaver};

const CODE_POLYNOMIALS: [usize; 2] = [0o13, 0o15];
const INVERSE_CODE_RATE: usize = 3;
const NUM_TAIL_CODE_BITS: usize = 12;

/// Parameters for LTE PCCC simulation over BPSK-AWGN channel
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

impl SimParams {
    /// Checks validity of simulation parameters.
    fn check(&self) -> Result<(), Error> {
        if qpp_coefficients(self.num_info_bits_per_block as usize).is_err() {
            return Err(Error::InvalidInput(format!(
                "{} is not a valid number of information bits per block",
                self.num_info_bits_per_block,
            )));
        }
        if self.num_blocks_per_run == 0 {
            return Err(Error::InvalidInput(
                "Number of blocks per run cannot be zero".to_string(),
            ));
        }
        if self.num_runs_min > self.num_runs_max {
            return Err(Error::InvalidInput(format!(
                "Minimum number of runs ({}) exceeds maximum number of runs ({})",
                self.num_runs_min, self.num_runs_max,
            )));
        }
        Ok(())
    }

    /// Prints simulation parameters.
    fn print(&self) {
        eprintln!();
        self.print_num_info_bits_per_block();
        self.print_es_over_n0_db();
        self.print_decoding_algo();
        self.print_num_block_errors_min();
        self.print_num_blocks_per_run();
        self.print_num_runs_min();
        self.print_num_runs_max();
    }

    /// Prints information block size.
    fn print_num_info_bits_per_block(&self) {
        eprintln!(
            "{:?} information bits per block",
            self.num_info_bits_per_block,
        );
    }

    /// Prints Es/N0 (dB) value.
    fn print_es_over_n0_db(&self) {
        eprintln!("Es/N0 of {} dB", self.es_over_n0_db);
    }

    /// Prints decoding algorithm to use.
    fn print_decoding_algo(&self) {
        match self.decoding_algo {
            DecodingAlgo::LogMAP(n) => eprintln!("Log-MAP decoding, {n} iterations"),
            DecodingAlgo::MaxLogMAP(n) => eprintln!("Max-Log-MAP decoding, {n} iterations"),
            DecodingAlgo::LinearLogMAP(n) => eprintln!("Linear-Log-MAP decoding, {n} iterations"),
        };
    }

    /// Prints desired minimum number of block errors.
    fn print_num_block_errors_min(&self) {
        eprintln!("Minimum of {} block errors", self.num_block_errors_min);
    }

    /// Prints number of blocks to be transmitted per run.
    fn print_num_blocks_per_run(&self) {
        eprintln!("{} blocks per run", self.num_blocks_per_run);
    }

    /// Prints minimum number of runs of blocks to be simulated.
    fn print_num_runs_min(&self) {
        eprintln!("Minimum of {} runs", self.num_runs_min);
    }

    /// Prints maximum number of runs of blocks to be simulated.
    fn print_num_runs_max(&self) {
        eprintln!("Maximum of {} runs", self.num_runs_max);
    }
}

/// Results from LTE PCCC simulation over BPSK-AWGN channel
#[derive(Clone, PartialEq, Debug, Copy, Deserialize, Serialize)]
pub struct SimResults {
    /// Simulation parameters
    pub params: SimParams,
    /// Number of blocks transmitted
    pub num_blocks: u32,
    /// Number of information bits transmitted
    pub num_info_bits: u32,
    /// Number of block errors
    pub num_block_errors: u32,
    /// Number of information bit errors
    pub num_info_bit_errors: u32,
}

impl SimResults {
    /// Returns initialized simulation results.
    #[must_use]
    fn new(params: &SimParams) -> Self {
        Self {
            params: *params,
            num_blocks: 0,
            num_info_bits: 0,
            num_block_errors: 0,
            num_info_bit_errors: 0,
        }
    }

    /// Returns block error rate.
    #[must_use]
    pub fn block_error_rate(&self) -> f64 {
        if self.num_blocks > 0 {
            f64::from(self.num_block_errors) / f64::from(self.num_blocks)
        } else {
            0.0
        }
    }

    /// Returns information bit error rate.
    #[must_use]
    pub fn info_bit_error_rate(&self) -> f64 {
        if self.num_info_bits > 0 {
            f64::from(self.num_info_bit_errors) / f64::from(self.num_info_bits)
        } else {
            0.0
        }
    }

    /// Prints progress message.
    fn print_progress_message(&self) {
        if self.run_complete() {
            eprint!(
                "\r{:5} bits/block, Es/N0 = {:6.3} dB: \
                 BER = {:9.4e}, BLER = {:9.4e} ({}/{}, {}/{})",
                self.params.num_info_bits_per_block,
                self.params.es_over_n0_db,
                self.info_bit_error_rate(),
                self.block_error_rate(),
                self.num_info_bit_errors,
                self.num_info_bits,
                self.num_block_errors,
                self.num_blocks,
            );
            if self.sim_complete() {
                eprintln!();
            }
        }
    }

    /// Returns `true` iff a run of blocks is now complete.
    fn run_complete(&self) -> bool {
        self.num_blocks % self.params.num_blocks_per_run == 0
    }

    /// Returns `true` iff the simulation is now complete.
    fn sim_complete(&self) -> bool {
        self.run_complete()
            && self.num_blocks >= self.params.num_runs_min * self.params.num_blocks_per_run
            && (self.num_block_errors >= self.params.num_block_errors_min
                || self.num_blocks >= self.params.num_runs_max * self.params.num_blocks_per_run)
    }

    /// Updates simulation results after a block.
    fn update_after_block(&mut self, num_info_bit_errors_this_block: u32) {
        self.num_blocks += 1;
        self.num_info_bits += self.params.num_info_bits_per_block;
        if num_info_bit_errors_this_block > 0 {
            self.num_block_errors += 1;
            self.num_info_bit_errors += num_info_bit_errors_this_block;
        }
    }
}

/// Returns code bits from rate-1/3 LTE PCCC encoder for given information bits.
///
/// # Parameters
///
/// - `info_bits`: Information bits to be encoded.
///
/// # Returns
///
/// - `code_bits`: Code bits from the encoder.
///
/// # Errors
///
/// Returns an error if the number of information bits does not equal any of the block sizes in
/// Table 5.1.3-3 of 3GPP TS 36.212.
pub fn encoder(info_bits: &[Bit]) -> Result<Vec<Bit>, Error> {
    let qpp_interleaver = interleaver(info_bits.len())?;
    crate::encoder(info_bits, &qpp_interleaver, &CODE_POLYNOMIALS)
}

/// Returns information bit decisions from rate-1/3 LTE PCCC decoder for given code bit LLR values.
///
/// # Parameters
///
/// - `code_bits_llr`: Log-likelihood-ratio (LLR) values for the code bits.
///
/// - `decoding_algo`: Decoding algorithm to use, and associated number of turbo iterations.
///
/// # Returns
///
/// - `info_bits_hat`: Information bit decisions from the decoder.
///
/// # Errors
///
/// Returns an error if the number of code bit LLR values does not correspond to any of the block
/// sizes in Table 5.1.3-3 of 3GPP TS 36.212.
pub fn decoder(code_bits_llr: &[f64], decoding_algo: DecodingAlgo) -> Result<Vec<Bit>, Error> {
    if code_bits_llr.len() < NUM_TAIL_CODE_BITS {
        return Err(Error::InvalidInput(format!(
            "Must have a minimum of {NUM_TAIL_CODE_BITS} code bit LLR values",
        )));
    }
    let num_info_bits = (code_bits_llr.len() - NUM_TAIL_CODE_BITS) / INVERSE_CODE_RATE;
    let qpp_interleaver = interleaver(num_info_bits)?;
    crate::decoder(
        code_bits_llr,
        &qpp_interleaver,
        &CODE_POLYNOMIALS,
        decoding_algo,
    )
}

/// Runs simulation of rate-1/3 LTE PCCC over a BPSK-AWGN channel.
///
/// # Parameters
///
/// - `params`: Parameters for the simulation.
///
/// - `rng`: Random number generator for the simulation.
///
/// # Returns
///
/// - `results`: Results from the simulation.
///
/// # Errors
///
/// Returns an error if `params.num_info_bits_per_block` is not one of the values specified in
/// Table 5.1.3-3 of 3GPP TS 36.212, if `params.num_blocks_per_run` is `0`, or if
/// `params.num_runs_min` exceeds `params.num_runs_max`.
///
/// # Examples
///
/// ```
/// use pccc::{lte, DecodingAlgo};
///
/// let mut rng = rand::thread_rng();
/// let params = lte::SimParams {
///     num_info_bits_per_block: 40,
///     es_over_n0_db: -3.0,
///     decoding_algo: DecodingAlgo::LogMAP(8),
///     num_block_errors_min: 10,
///     num_blocks_per_run: 10,
///     num_runs_min: 1,
///     num_runs_max: 2,
/// };
/// let results: lte::SimResults = lte::bpsk_awgn_sim(&params, &mut rng)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::cast_possible_truncation)]
pub fn bpsk_awgn_sim(params: &SimParams, rng: &mut ThreadRng) -> Result<SimResults, Error> {
    params.check()?;
    let mut results = SimResults::new(params);
    while !results.sim_complete() {
        let info_bits = utils::random_bits(params.num_info_bits_per_block as usize, rng);
        let code_bits = encoder(&info_bits)?;
        let code_bits_llr = utils::bpsk_awgn_channel(&code_bits, params.es_over_n0_db, rng);
        let info_bits_hat = decoder(&code_bits_llr, params.decoding_algo)?;
        let num_info_bit_errors_this_block = utils::error_count(&info_bits_hat, &info_bits);
        results.update_after_block(num_info_bit_errors_this_block as u32);
        results.print_progress_message();
    }
    Ok(results)
}

/// Runs simulations of rate-1/3 LTE PCCC over a BPSK-AWGN channel and saves results to JSON file.
///
/// # Parameters
///
/// - `all_params`: Parameters for each simulation scenario of interest.
///
/// -  `rng`: Random number generator for the simulations.
///
/// - `json_filename`: Name of the JSON file to which all simulation results must be written.
///
/// # Errors
///
/// Returns an error if any invalid simulation parameters are encountered, or if there is an error
/// in creating or writing to the JSON file for the simulation results.
///
/// # Examples
///
/// ```
/// use pccc::{lte, DecodingAlgo};
///
/// let mut rng = rand::thread_rng();
/// let mut all_params = Vec::new();
/// for num_info_bits_per_block in [40, 48] {
///     all_params.push(lte::SimParams {
///         num_info_bits_per_block,
///         es_over_n0_db: -3.0,
///         decoding_algo: DecodingAlgo::LogMAP(8),
///         num_block_errors_min: 100,
///         num_blocks_per_run: 100,
///         num_runs_min: 1,
///         num_runs_max: 100,
///     });
/// }
/// // lte::run_bpsk_awgn_sims(&all_params, &mut rng, "results.json")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn run_bpsk_awgn_sims(
    all_params: &[SimParams],
    rng: &mut ThreadRng,
    json_filename: &str,
) -> Result<(), Error> {
    let mut all_results = Vec::with_capacity(all_params.len());
    for params in all_params {
        params.print();
        if let Ok(results) = bpsk_awgn_sim(params, rng) {
            all_results.push(results);
        } else {
            eprintln!("WARNING: Invalid simulation parameters");
        }
    }
    save_all_sim_results_to_file(&all_results, json_filename)?;
    Ok(())
}

/// Saves all simulation results to a JSON file.
///
/// # Parameters
///
/// - `all_results`: All simulation results to be saved.
///
/// - `json_filename`: Name of the JSON file to which all simulation results must be written.
///
/// # Errors
///
/// Returns an error if creating or writing to the JSON file fails.
pub fn save_all_sim_results_to_file(
    all_results: &[SimResults],
    json_filename: &str,
) -> Result<(), Error> {
    let writer = BufWriter::new(File::create(json_filename)?);
    serde_json::to_writer_pretty(writer, all_results)?;
    Ok(())
}

/// Returns all simulation results from a JSON file.
///
/// # Parameters
///
/// - `json_filename`: Name of the JSON file from which all simulation results must be read.
///
/// # Errors
///
/// Returns an error if opening or reading from the JSON file fails.
pub fn all_sim_results_from_file(json_filename: &str) -> Result<Vec<SimResults>, Error> {
    let reader = BufReader::new(File::open(json_filename)?);
    let all_results = serde_json::from_reader(reader)?;
    Ok(all_results)
}

/// Returns quadratic permutation polynomial (QPP) interleaver for rate-1/3 LTE PCCC.
fn interleaver(num_info_bits: usize) -> Result<Interleaver, Error> {
    let (coeff1, coeff2) = qpp_coefficients(num_info_bits)?;
    let perm: Vec<usize> = (0 .. num_info_bits)
        .map(|out_index| ((coeff1 + coeff2 * out_index) * out_index) % num_info_bits)
        .collect();
    Interleaver::new(&perm)
}

/// Returns QPP coefficients for given number of information bits.
fn qpp_coefficients(num_info_bits: usize) -> Result<(usize, usize), Error> {
    if num_info_bits <= 128 {
        qpp_coefficients_40_to_128_bits(num_info_bits)
    } else if num_info_bits <= 256 {
        qpp_coefficients_136_to_256_bits(num_info_bits)
    } else if num_info_bits <= 512 {
        qpp_coefficients_264_to_512_bits(num_info_bits)
    } else if num_info_bits <= 1024 {
        qpp_coefficients_528_to_1024_bits(num_info_bits)
    } else if num_info_bits <= 2048 {
        qpp_coefficients_1056_to_2048_bits(num_info_bits)
    } else if num_info_bits <= 4096 {
        qpp_coefficients_2112_to_4096_bits(num_info_bits)
    } else {
        qpp_coefficients_4160_to_6144_bits(num_info_bits)
    }
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
mod tests_of_simparams {
    use super::*;

    #[test]
    fn test_check() {
        // Invalid input
        let params = SimParams {
            num_info_bits_per_block: 32,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 1,
            num_runs_min: 1,
            num_runs_max: 2,
        };
        assert!(params.check().is_err());
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 0,
            num_runs_min: 1,
            num_runs_max: 2,
        };
        assert!(params.check().is_err());
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 1,
            num_runs_min: 2,
            num_runs_max: 1,
        };
        assert!(params.check().is_err());
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
        assert!(params.check().is_ok());
    }

    #[test]
    fn test_print() {
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 100,
            num_blocks_per_run: 1000,
            num_runs_min: 1,
            num_runs_max: 10,
        };
        params.print();
    }
}

#[cfg(test)]
mod tests_of_simresults {
    use super::*;
    use float_eq::assert_float_eq;

    fn params_for_test() -> SimParams {
        SimParams {
            num_info_bits_per_block: 48,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 100,
            num_blocks_per_run: 1000,
            num_runs_min: 1,
            num_runs_max: 10,
        }
    }

    fn results_for_test(num_blocks: u32, num_block_errors: u32) -> SimResults {
        let params = params_for_test();
        let num_info_bits = num_blocks * params.num_info_bits_per_block;
        let num_info_bit_errors = num_block_errors * 10;
        SimResults {
            params,
            num_blocks,
            num_info_bits,
            num_block_errors,
            num_info_bit_errors,
        }
    }

    #[test]
    fn test_new() {
        let results = SimResults::new(&params_for_test());
        assert_eq!(results.num_blocks, 0);
        assert_eq!(results.num_info_bits, 0);
        assert_eq!(results.num_block_errors, 0);
        assert_eq!(results.num_info_bit_errors, 0);
    }

    #[test]
    fn test_block_error_rate() {
        let results = results_for_test(2000, 10);
        assert_float_eq!(results.block_error_rate(), 10.0 / 2000.0, abs <= 1e-8);
    }

    #[test]
    fn test_info_bit_error_rate() {
        let results = results_for_test(2000, 10);
        assert_float_eq!(results.info_bit_error_rate(), 100.0 / 96000.0, abs <= 1e-8);
    }

    #[test]
    fn test_print_progress_message() {
        let results = results_for_test(1000, 10);
        results.print_progress_message();
    }

    #[test]
    fn test_run_complete() {
        let results = results_for_test(999, 0);
        assert!(!results.run_complete());
        let results = results_for_test(1000, 0);
        assert!(results.run_complete());
        let results = results_for_test(1001, 0);
        assert!(!results.run_complete());
    }

    #[test]
    fn test_sim_complete() {
        // Test of `num_block_errors_min`
        let results = results_for_test(5000, 99);
        assert!(!results.sim_complete());
        let results = results_for_test(5000, 100);
        assert!(results.sim_complete());
        // Test of `num_blocks_per_run`
        let results = results_for_test(4999, 500);
        assert!(!results.sim_complete());
        let results = results_for_test(5000, 500);
        assert!(results.sim_complete());
        let results = results_for_test(5001, 500);
        assert!(!results.sim_complete());
        // Test of `num_runs_min`
        let results = results_for_test(999, 500);
        assert!(!results.sim_complete());
        let results = results_for_test(1000, 500);
        assert!(results.sim_complete());
        // Test of `num_runs_max`
        let results = results_for_test(9999, 50);
        assert!(!results.sim_complete());
        let results = results_for_test(10000, 50);
        assert!(results.sim_complete());
    }

    #[test]
    fn test_update_after_block() {
        // Zero info bit errors
        let mut results = results_for_test(100, 10);
        results.update_after_block(0);
        assert_eq!(results.num_blocks, 101);
        assert_eq!(results.num_info_bits, 4848);
        assert_eq!(results.num_block_errors, 10);
        assert_eq!(results.num_info_bit_errors, 100);
        // Nonzero info bit errors
        let mut results = results_for_test(100, 10);
        results.update_after_block(10);
        assert_eq!(results.num_blocks, 101);
        assert_eq!(results.num_info_bits, 4848);
        assert_eq!(results.num_block_errors, 11);
        assert_eq!(results.num_info_bit_errors, 110);
    }
}

#[cfg(test)]
mod tests_of_functions {
    use super::*;

    #[test]
    fn test_encoder() {
        let mut rng = rand::thread_rng();
        let info_bits = utils::random_bits(40, &mut rng);
        assert!(encoder(&info_bits).is_ok());
    }

    #[test]
    fn test_decoder() {
        let mut rng = rand::thread_rng();
        let code_bits_llr =
            utils::bpsk_awgn_channel(&utils::random_bits(132, &mut rng), 10.0, &mut rng);
        assert!(decoder(&code_bits_llr, DecodingAlgo::LinearLogMAP(8)).is_ok());
    }

    #[test]
    fn test_bpsk_awgn_sim() {
        let mut rng = rand::thread_rng();
        // Invalid input
        let params = SimParams {
            num_info_bits_per_block: 32,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 10,
            num_runs_min: 1,
            num_runs_max: 2,
        };
        assert!(bpsk_awgn_sim(&params, &mut rng).is_err());
        // Valid input
        let params = SimParams {
            num_info_bits_per_block: 40,
            es_over_n0_db: -3.0,
            decoding_algo: DecodingAlgo::LinearLogMAP(8),
            num_block_errors_min: 10,
            num_blocks_per_run: 10,
            num_runs_min: 1,
            num_runs_max: 2,
        };
        assert!(bpsk_awgn_sim(&params, &mut rng).is_ok());
    }

    #[test]
    fn test_run_bpsk_awgn_sims() {
        let mut rng = rand::thread_rng();
        let all_params = vec![
            SimParams {
                num_info_bits_per_block: 32,
                es_over_n0_db: -3.0,
                decoding_algo: DecodingAlgo::LinearLogMAP(8),
                num_block_errors_min: 20,
                num_blocks_per_run: 10,
                num_runs_min: 1,
                num_runs_max: 2,
            },
            SimParams {
                num_info_bits_per_block: 40,
                es_over_n0_db: -3.0,
                decoding_algo: DecodingAlgo::LinearLogMAP(8),
                num_block_errors_min: 10,
                num_blocks_per_run: 10,
                num_runs_min: 1,
                num_runs_max: 2,
            },
        ];
        run_bpsk_awgn_sims(&all_params, &mut rng, "results.json").unwrap();
    }

    fn all_sim_params_for_test() -> Vec<SimParams> {
        let mut all_params = Vec::new();
        for num_info_bits_per_block in [40, 48] {
            for es_over_n0_db in [-3.5, -3.0] {
                all_params.push(SimParams {
                    num_info_bits_per_block,
                    es_over_n0_db,
                    decoding_algo: DecodingAlgo::LinearLogMAP(8),
                    num_block_errors_min: 20,
                    num_blocks_per_run: 10,
                    num_runs_min: 1,
                    num_runs_max: 10,
                });
            }
        }
        all_params
    }

    fn all_sim_results_for_test(all_params: &[SimParams]) -> Vec<SimResults> {
        let mut all_results = Vec::new();
        for params in all_params {
            all_results.push(SimResults::new(params));
        }
        all_results
    }

    #[test]
    fn test_save_all_sim_results_to_file() {
        let all_results = all_sim_results_for_test(&all_sim_params_for_test());
        let json_filename = "results.json";
        save_all_sim_results_to_file(&all_results, json_filename).unwrap();
        let all_results_saved = all_sim_results_from_file(json_filename).unwrap();
        assert_eq!(all_results, all_results_saved);
    }

    #[test]
    fn test_all_sim_results_from_file() {
        let all_results = all_sim_results_for_test(&all_sim_params_for_test());
        let json_filename = "results.json";
        save_all_sim_results_to_file(&all_results, json_filename).unwrap();
        let all_results_saved = all_sim_results_from_file(json_filename).unwrap();
        assert_eq!(all_results, all_results_saved);
    }

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
    fn test_qpp_coefficients() {
        // Invalid input
        assert!(qpp_coefficients(32).is_err());
        // Valid input
        for num in 0 .. 60 {
            let num_info_bits = 40 + 8 * num;
            assert!(qpp_coefficients(num_info_bits).is_ok());
        }
        for num in 0 .. 32 {
            let num_info_bits = 528 + 16 * num;
            assert!(qpp_coefficients(num_info_bits).is_ok());
        }
        for num in 0 .. 32 {
            let num_info_bits = 1056 + 32 * num;
            assert!(qpp_coefficients(num_info_bits).is_ok());
        }
        for num in 0 .. 64 {
            let num_info_bits = 2112 + 64 * num;
            assert!(qpp_coefficients(num_info_bits).is_ok());
        }
        // Invalid input
        assert!(qpp_coefficients(6208).is_err());
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
