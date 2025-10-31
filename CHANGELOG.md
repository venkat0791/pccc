# Changelog

## pccc 0.4.0 (2025-10-31)

[72c6185...HEAD](https://github.com/venkat0791/pccc/compare/72c6185...HEAD)

### Added since 0.3.0

### Changed since 0.3.0

- Removed the lte::summarize_all_sim_results_in_file function (which was in the public interface), since it seemed unsound.

### Fixed since 0.3.0

### Documentation changes since 0.3.0

- Corrected the "Errors" section of the docstring of lte::run_bpsk_awgn_sims.
- Improved the README.

### Internal changes since 0.3.0

- Moved all code out of lib.rs and into some new private modules (common, pccc, and interleaver).
- In main.rs, renamed a helper function and added/improved tests.
- Replaced "x as y" with try_from where possible.
- Reordered all `use` statements consistently, putting std/core ones first, then those from dependencies, and finally those from within the crate.
- Updated all dependencies to their respective latest versions (no code changes were required as a result).

## pccc 0.3.0 (2025-04-13)

[fdd6639...72c6185](https://github.com/venkat0791/pccc/compare/fdd6639...72c6185)

### Added since 0.2.0

### Changed since 0.2.0

- Removed the random number generator argument from several functions (instead, now the thread-local random number generator is called within those functions). This was needed to support parallelization through rayon.
- The `lte::bpsk_awgn_sim` function no longer prints progress messages.
- The `lte::run_bpsk_awgn_sims` function now saves all simulation results to a JSON file only at the end of the entire simulation.

### Fixed since 0.2.0

### Documentation changes since 0.2.0

- Made numerous minor edits to the README.
- Made minor changes to various doctests.
- Improved several error messages.

### Internal changes since 0.2.0

- Updated all dependencies to their respective latest versions (code changes were caused only by the updates to the `rand` and `rand_distr` crates). Now all dependency versions are specified as x.y.z (some were just x.y earlier).
- Added the `rayon` crate as a dependency, and used it to parallelize `lte::bpsk_awgn_sim` and `lte::run_bpsk_awgn_sims`, leading to a very significant speed-up of those two functions (e.g., the simulation described in the README now runs in 15% of the time compared to version 0.2.0).

## pccc 0.2.0 (2024-11-06)

[41389c4...fdd6639](https://github.com/venkat0791/pccc/compare/41389c4...fdd6639)

### Added since 0.1.0

### Changed since 0.1.0

- Renamed a command-line parameter (in the binary crate), and changed the default values of two others.

### Fixed since 0.1.0

### Documentation changes since 0.1.0

- Recreated the README from scratch, and added a section on the binary crate.
- Made several edits for clarity and consistency.

### Internal changes since 0.1.0
