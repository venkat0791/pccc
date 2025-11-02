# pccc

[![crate](https://img.shields.io/crates/v/pccc.svg)](https://crates.io/crates/pccc)
[![documentation](https://docs.rs/pccc/badge.svg)](https://docs.rs/pccc)

This package implements encoding and decoding functionality for a _parallel-concatenated convolutional code_ (PCCC), commonly referred to as a [turbo code](https://en.wikipedia.org/wiki/Turbo_code). An instance of such a code is used for forward error correction in the 4G LTE standard for wireless broadband communication (see [3GPP TS 36.212](https://www.3gpp.org/ftp/Specs/archive/36_series/36.212/)).

The package comprises a library crate and a binary crate. Refer to the [API documentation](https://docs.rs/pccc) for details on the former (including illustrative examples), and read on for more on the latter.

## Binary crate usage

The included binary crate can be used to run a Monte Carlo simulation that evaluates the error rate performance over a BPSK-AWGN channel of the rate-1/3 PCCC used in LTE. The parameters for such a simulation are specified on the command line, and the results from it are saved to a JSON file.

Execute `cargo run -- -h` for help on the command-line interface.

```console
$ cargo run -- -h
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.05s
     Running `target/debug/pccc -h`
Evaluates the performance of the rate-1/3 LTE PCCC over a BPSK-AWGN channel

Usage: pccc [OPTIONS]

Options:
  -i <num_info_bits_per_block>      Number of information bits per block [default: 40]
  -a <decoding_algo_name>           Decoding algorithm name [default: LogMAP] [possible values: LogMAP, MaxLogMAP, LinearLogMAP]
  -t <num_turbo_iter>               Number of turbo iterations [default: 8]
  -r <first_snr_db>                 First Es/N0 (dB) [default: -5.0]
  -p <snr_step_db>                  Es/N0 step (dB) [default: 1.0]
  -s <num_snr>                      Number of Es/N0 values [default: 4]
  -e <num_block_errors_min>         Desired minimum number of block errors [default: 1000]
  -b <num_blocks_per_run>           Number of blocks to be transmitted per run [default: 1000]
  -n <num_runs_min>                 Minimum number of runs of blocks to be simulated [default: 10]
  -x <num_runs_max>                 Maximum number of runs of blocks to be simulated [default: 1000]
  -f <json_filename>                Name of JSON file to which results must be saved [default: results.json]
  -h, --help                        Print help
  -V, --version                     Print version
```

Execute `cargo run --release -- [OPTIONS]` to run a simulation with desired parameters. For example, the following command evaluates the bit error rate (BER) and block error rate (BLER) of the LTE PCCC for a block size of 6144 bits, with Log-MAP decoding, 8 turbo iterations, and the signal-to-noise ratio (Es/N0) ranging from -4.95 dB to -4.35 dB in 0.1 dB steps. Results are saved to a file named _results6144.json_. (Some parameters are left at their default values.)

```console
cargo run --release -- -i 6144 -a LogMAP -t 8 -r -4.95 -p 0.1 -s 7 -f results6144.json
```

For reference, the BER and BLER from a simulation with the above parameters are tabulated below:

 | Es/N0 (dB) |    BER    |    BLER    |
 |:----------:|:---------:|:----------:|
 |  -4.95     |  1.05e-1  |    1.00e0  |
 |  -4.85     |  7.86e-2  |   9.89e-1  |
 |  -4.75     |  4.71e-2  |   9.07e-1  |
 |  -4.65     |  1.89e-2  |   6.31e-1  |
 |  -4.55     |  3.86e-3  |   2.47e-1  |
 |  -4.45     |  4.39e-4  |   5.03e-2  |
 |  -4.35     |  2.86e-5  |   5.17e-3  |

License: MIT
