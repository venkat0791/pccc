//! This crate simulates the BER-versus-SNR and BLER-versus-SNR performance of the rate-1/3
//! parallel-concatenated convolutional code (PCCC) in LTE over a BPSK-AWGN channel. Simulation
//! parameters are specified on the command line, and simulation results are saved to a JSON file.
//!
//! Build the executable with `cargo build --release` and then run `./target/release/pccc -h` for
//! help on the command-line interface.

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

use anyhow::Result;
use clap::parser::ValueSource;
use clap::{crate_name, crate_version, value_parser, Arg, ArgMatches, Command};
use pccc::{lte, DecodingAlgo};
use std::time::Instant;

/// Main function
fn main() -> Result<()> {
    let timer = Instant::now();
    let mut rng = rand::rng();
    let matches = command_line_parser().get_matches();
    let json_filename = &json_filename_from_matches(&matches);
    lte::run_bpsk_awgn_sims(&all_sim_params(&matches), &mut rng, json_filename)?;
    eprintln!("Elapsed time: {:.3?}", timer.elapsed());
    Ok(())
}

/// Returns command line parser.
fn command_line_parser() -> Command {
    Command::new(crate_name!())
        .version(crate_version!())
        .about("Evaluates the performance of the rate-1/3 LTE PCCC over a BPSK-AWGN channel")
        .arg(num_info_bits_per_block())
        .arg(decoding_algo_name())
        .arg(num_turbo_iter())
        .arg(first_snr_db())
        .arg(snr_step_db())
        .arg(num_snr())
        .arg(num_block_errors_min())
        .arg(num_blocks_per_run())
        .arg(num_runs_min())
        .arg(num_runs_max())
        .arg(json_filename())
}

/// Returns argument for number of information bits per block.
fn num_info_bits_per_block() -> Arg {
    Arg::new("num_info_bits_per_block")
        .short('i')
        .value_parser(value_parser!(u32))
        .default_value("40")
        .help("Number of information bits per block")
}

/// Returns argument for decoding algorithm name.
fn decoding_algo_name() -> Arg {
    Arg::new("decoding_algo_name")
        .short('a')
        .value_parser(["LogMAP", "MaxLogMAP", "LinearLogMAP"])
        .default_value("LogMAP")
        .help("Decoding algorithm name")
}

/// Returns argument for number of turbo iterations.
fn num_turbo_iter() -> Arg {
    Arg::new("num_turbo_iter")
        .short('t')
        .value_parser(value_parser!(u32))
        .default_value("8")
        .help("Number of turbo iterations")
}

/// Returns argument for first Es/N0 (dB).
fn first_snr_db() -> Arg {
    Arg::new("first_snr_db")
        .short('r')
        .value_parser(value_parser!(f64))
        .allow_negative_numbers(true)
        .default_value("-5.0")
        .help("First Es/N0 (dB)")
}

/// Returns argument for Es/N0 step (dB).
fn snr_step_db() -> Arg {
    Arg::new("snr_step_db")
        .short('p')
        .value_parser(value_parser!(f64))
        .allow_negative_numbers(true)
        .default_value("1.0")
        .help("Es/N0 step (dB)")
}

/// Returns argument for number of Es/N0 values.
fn num_snr() -> Arg {
    Arg::new("num_snr")
        .short('s')
        .value_parser(value_parser!(u32))
        .default_value("4")
        .help("Number of Es/N0 values")
}

/// Returns argument for desired minimum number of block errors.
fn num_block_errors_min() -> Arg {
    Arg::new("num_block_errors_min")
        .short('e')
        .value_parser(value_parser!(u32))
        .default_value("500")
        .help("Desired minimum number of block errors")
}

/// Returns argument for number of blocks to be transmitted per run.
fn num_blocks_per_run() -> Arg {
    Arg::new("num_blocks_per_run")
        .short('b')
        .value_parser(value_parser!(u32))
        .default_value("1000")
        .help("Number of blocks to be transmitted per run")
}

/// Returns argument for minimum number of runs of blocks to be simulated.
fn num_runs_min() -> Arg {
    Arg::new("num_runs_min")
        .short('n')
        .value_parser(value_parser!(u32))
        .default_value("10")
        .help("Minimum number of runs of blocks to be simulated")
}

/// Returns argument for maximum number of runs of blocks to be simulated.
fn num_runs_max() -> Arg {
    Arg::new("num_runs_max")
        .short('x')
        .value_parser(value_parser!(u32))
        .default_value("100")
        .help("Maximum number of runs of blocks to be simulated")
}

/// Returns argument for name of JSON file to which results must be saved.
fn json_filename() -> Arg {
    Arg::new("json_filename")
        .short('f')
        .default_value("results.json")
        .help("Name of JSON file to which results must be saved")
}

/// Returns simulation parameters based on command-line arguments.
fn all_sim_params(matches: &ArgMatches) -> Vec<lte::SimParams> {
    let mut num_runs_min = num_runs_min_from_matches(matches);
    let mut num_runs_max = num_runs_max_from_matches(matches);
    if num_runs_min > num_runs_max {
        if let Some(ValueSource::DefaultValue) = matches.value_source("num_runs_min") {
            num_runs_min = num_runs_max;
        }
        if let Some(ValueSource::DefaultValue) = matches.value_source("num_runs_max") {
            num_runs_max = num_runs_min;
        }
    }
    let mut all_params = Vec::new();
    for es_over_n0_db in all_es_over_n0_db_from_matches(matches) {
        all_params.push(lte::SimParams {
            num_info_bits_per_block: num_info_bits_per_block_from_matches(matches),
            decoding_algo: decoding_algo_from_matches(matches),
            es_over_n0_db,
            num_block_errors_min: num_block_errors_min_from_matches(matches),
            num_blocks_per_run: num_blocks_per_run_from_matches(matches),
            num_runs_min,
            num_runs_max,
        });
    }
    // OK to unwrap: All command-line arguments have default values, so an error cannot occur
    // in any of the associated functions called above.
    all_params
}

/// Returns number of information bits per block.
fn num_info_bits_per_block_from_matches(matches: &ArgMatches) -> u32 {
    *matches.get_one("num_info_bits_per_block").unwrap()
}

/// Returns decoding algorithm.
fn decoding_algo_from_matches(matches: &ArgMatches) -> DecodingAlgo {
    let num_turbo_iter = *matches.get_one("num_turbo_iter").unwrap();
    match matches
        .get_one::<String>("decoding_algo_name")
        .unwrap()
        .as_str()
    {
        "LogMAP" => DecodingAlgo::LogMAP(num_turbo_iter),
        "MaxLogMAP" => DecodingAlgo::MaxLogMAP(num_turbo_iter),
        "LinearLogMAP" => DecodingAlgo::LinearLogMAP(num_turbo_iter),
        _ => panic!("Invalid decoding algorithm name"),
    }
}

/// Returns all Es/N0 (dB) values.
fn all_es_over_n0_db_from_matches(matches: &ArgMatches) -> Vec<f64> {
    let first_snr_db: f64 = *matches.get_one("first_snr_db").unwrap();
    let snr_step_db: f64 = *matches.get_one("snr_step_db").unwrap();
    let num_snr: u32 = *matches.get_one("num_snr").unwrap();
    (0 .. num_snr)
        .map(|n| first_snr_db + snr_step_db * f64::from(n))
        .collect()
}

/// Returns desired minimum number of block errors.
fn num_block_errors_min_from_matches(matches: &ArgMatches) -> u32 {
    *matches.get_one("num_block_errors_min").unwrap()
}

/// Returns number of blocks to be transmitted per run.
fn num_blocks_per_run_from_matches(matches: &ArgMatches) -> u32 {
    *matches.get_one("num_blocks_per_run").unwrap()
}

/// Returns minimum number of runs of blocks to be simulated.
fn num_runs_min_from_matches(matches: &ArgMatches) -> u32 {
    *matches.get_one("num_runs_min").unwrap()
}

/// Returns maximum number of runs of blocks to be simulated.
fn num_runs_max_from_matches(matches: &ArgMatches) -> u32 {
    *matches.get_one("num_runs_max").unwrap()
}

/// Returns name of JSON file to which simulation results must be saved.
fn json_filename_from_matches(matches: &ArgMatches) -> String {
    matches
        .get_one::<String>("json_filename")
        .unwrap()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_line_for_test() -> Vec<&'static str> {
        vec![
            crate_name!(),
            "-i",
            "40",
            "-a",
            "LogMAP",
            "-t",
            "8",
            "-r",
            "-4.0",
            "-p",
            "0.2",
            "-s",
            "6",
            "-e",
            "50",
            "-b",
            "100",
            "-n",
            "10",
            "-x",
            "20",
            "-f",
            "results.json",
        ]
    }

    #[test]
    fn test_command_line_parser() {
        assert!(command_line_parser()
            .try_get_matches_from(command_line_for_test())
            .is_ok());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_all_sim_params() {
        let matches = command_line_parser().get_matches_from(command_line_for_test());
        let all_params = all_sim_params(&matches);
        let all_es_over_n0_db = [-4.0, -3.8, -3.6, -3.4, -3.2, -3.0];
        assert_eq!(all_params.len(), 6);
        for (idx, &params) in all_params.iter().enumerate() {
            assert_eq!(params.num_info_bits_per_block, 40);
            assert_eq!(params.decoding_algo, DecodingAlgo::LogMAP(8));
            assert_eq!(params.es_over_n0_db, all_es_over_n0_db[idx]);
            assert_eq!(params.num_block_errors_min, 50);
            assert_eq!(params.num_blocks_per_run, 100);
            assert_eq!(params.num_runs_min, 10);
            assert_eq!(params.num_runs_max, 20);
        }
    }
}
