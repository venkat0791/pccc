# pccc

## Overview

This package implements encoding and decoding functionality for a _parallel-concatenated convolutional code_ (PCCC), commonly referred to as a [turbo code](https://en.wikipedia.org/wiki/Turbo_code). An instance of such a code is used for forward error correction in the 4G LTE standard for wireless broadband communication (see [3GPP TS 36.212](https://www.3gpp.org/ftp/Specs/archive/36_series/36.212/)).

The package contains both a library crate and a binary crate. Refer to the [API documentation](https://docs.rs/pccc) for more details on the library crate (including illustrative examples). The binary crate evaluates (by simulation) the error rate performance of the rate-1/3 PCCC in LTE over a BPSK-AWGN channel. Simulation parameters are specified on the command line, and simulation results are saved to a JSON file. See below for further details.

## Usage

Build the executable with `cargo build --release` and then run `./target/release/pccc -h` for help on the command-line interface.

```console
$ ./target/release/pccc -h
Evaluates performance of rate-1/3 LTE PCCC over BPSK-AWGN channel

Usage: pccc [OPTIONS]

Options:
  -i <num_info_bits_per_block>      Number of information bits per block [default: 40]
  -a <decoding_algo_name>           Decoding algorithm name [default: LogMAP] [possible values: LogMAP, MaxLogMAP, LinearLogMAP]
  -t <num_decoding_iter>            Number of decoding iterations [default: 8]
  -r <first_snr_db>                 First Es/N0 (dB) [default: -5.0]
  -p <snr_step_db>                  Es/N0 step (dB) [default: 0.1]
  -s <num_snr>                      Number of Es/N0 values [default: 11]
  -e <num_block_errors_min>         Desired minimum number of block errors [default: 500]
  -b <num_blocks_per_run>           Number of blocks to be transmitted per run [default: 1000]
  -n <num_runs_min>                 Minimum number of runs of blocks to be simulated [default: 10]
  -x <num_runs_max>                 Maximum number of runs of blocks to be simulated [default: 100]
  -f <json_filename>                Name of JSON file to which results must be saved [default: results.json]
  -h, --help                        Print help
  -V, --version                     Print version
```

The bit error rate (BER) and block error rate (BLER) versus the signal-to-noise ratio (SNR) are shown below for a block size of 1536 bits and Log-MAP decoding with 8 iterations:

 | Es/N0 (dB) |    BER    |    BLER    |
 |------------|-----------|------------|
 |  -5.00     |  1.09e-1  |   9.65e-1  |
 |  -4.90     |  8.79e-2  |   9.00e-1  |
 |  -4.80     |  6.54e-2  |   7.86e-1  |
 |  -4.70     |  4.27e-2  |   6.12e-1  |
 |  -4.60     |  2.48e-2  |   4.15e-1  |
 |  -4.50     |  1.20e-2  |   2.44e-1  |
 |  -4.40     |  4.97e-3  |   1.15e-1  |
 |  -4.30     |  1.67e-3  |   4.23e-2  |
 |  -4.20     |  5.05e-4  |   1.55e-2  |
 |  -4.10     |  1.31e-4  |   4.39e-3  |
 |  -4.00     |  2.78e-5  |   1.06e-3  |

License: MIT
