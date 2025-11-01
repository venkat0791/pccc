# pccc

[![documentation](https://docs.rs/pccc/badge.svg)](https://docs.rs/pccc)
[![crate](https://img.shields.io/crates/v/pccc.svg)](https://crates.io/crates/pccc)

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
  -e <num_block_errors_min>         Desired minimum number of block errors [default: 500]
  -b <num_blocks_per_run>           Number of blocks to be transmitted per run [default: 1000]
  -n <num_runs_min>                 Minimum number of runs of blocks to be simulated [default: 10]
  -x <num_runs_max>                 Maximum number of runs of blocks to be simulated [default: 100]
  -f <json_filename>                Name of JSON file to which results must be saved [default: results.json]
  -h, --help                        Print help
  -V, --version                     Print version
```

Execute `cargo run --release -- [OPTIONS]` to run a simulation (in release mode) with desired parameters. For example, the following command evaluates the bit error rate (BER) and block error rate (BLER) of the LTE PCCC for a block size of 1536 bits, with Log-MAP decoding and 8 turbo iterations. The signal-to-noise ratio (Es/N0) ranges from -5 dB to -4 dB in 0.1 dB steps. Results are saved to a file named _results1536.json_. (Some parameters are left at their default values.)

```console
cargo run --release -- -i 1536 -a LogMAP -t 8 -r -5.0 -p 0.1 -s 11 -f results1536.json
```

For reference, the BER and BLER from a simulation with the above parameters are tabulated below:

 | Es/N0 (dB) |    BER    |    BLER    |
 |:----------:|:---------:|:----------:|
 |  -5.0      |  1.09e-1  |   9.65e-1  |
 |  -4.9      |  8.79e-2  |   9.00e-1  |
 |  -4.8      |  6.54e-2  |   7.86e-1  |
 |  -4.7      |  4.27e-2  |   6.12e-1  |
 |  -4.6      |  2.48e-2  |   4.15e-1  |
 |  -4.5      |  1.20e-2  |   2.44e-1  |
 |  -4.4      |  4.97e-3  |   1.15e-1  |
 |  -4.3      |  1.67e-3  |   4.23e-2  |
 |  -4.2      |  5.05e-4  |   1.55e-2  |
 |  -4.1      |  1.31e-4  |   4.39e-3  |
 |  -4.0      |  2.78e-5  |   1.06e-3  |

License: MIT
