//! Recursive systematic convolutional (RSC) encoder and decoder

use crate::{Bit, DecodingAlgo, Error};

const INF: f64 = 1e100;

/// State of an RSC encoder/decoder
#[derive(Clone, Eq, PartialEq, Debug, Copy)]
struct State(usize);

/// Branch metric for a single state transition in an RSC decoder
#[derive(Clone, PartialEq, Debug, Copy)]
struct BranchMetric {
    /// Component corresponding to prior LLR value for input bit
    prior: f64,
    /// Component corresponding to systematic bit LLR value
    systematic: f64,
    /// Component corresponding to parity bit LLR values
    parity: f64,
}

impl BranchMetric {
    /// Returns branch metric with given components.
    fn new(prior: f64, systematic: f64, parity: f64) -> Self {
        Self {
            prior,
            systematic,
            parity,
        }
    }

    /// Returns branch metric value (sum of all components).
    fn value(&self) -> f64 {
        self.prior + self.systematic + self.parity
    }
}

/// State machine for RSC encoder/decoder
#[derive(Debug)]
pub(crate) struct StateMachine {
    /// Code polynomials
    pub(crate) code_polynomials: Vec<usize>,
    /// Memory length
    pub(crate) memory_len: usize,
    /// Number of states
    pub(crate) num_states: usize,
    /// Number of output bits
    pub(crate) num_output_bits: usize,
    /// Buffer for output bits
    output_bits: Vec<Bit>,
    /// Buffer for branch metric
    branch_metric: BranchMetric,
    /// Current state
    state: State,
}

impl StateMachine {
    /// Returns state machine for RSC encoder/decoder corresponding to given code polynomials.
    ///
    /// # Parameters
    ///
    /// - `code_polynomials`: Integer representations of the generator polynomials for the code.
    ///   Must have length `N` for a code of rate `1/N`. The first element is taken as the feedback
    ///   polynomial (this corresponds to the systematic bit), and all subsequent ones as the
    ///   feedforward polynomials (these correspond to the parity bits). For a code of constraint
    ///   length `L`, the feedback polynomial must be in the range `(2^(L-1), 2^L)`, and each
    ///   feedforward polynomial must be in the range `[1, 2^L)` and different from the feedback
    ///   polynomial.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of code polynomials is less than `2`, if the first code
    /// polynomial is either `0` or a power of `2`, or if any subsequent code polynomial is either
    /// not in the range `[1, 2^L)` or equals the first code polynomial; here, `L` is the positive
    /// integer such that the first code polynomial is in the range `(2^(L-1), 2^L)`.
    ///
    /// # Notes
    ///
    /// Let `b_{n,0}`, `b_{n,1}`, ... , `b_{n,L-1}` be the `L`-bit binary representation of
    /// `code_polynomials[n]`, with `b_{n,0}` being the MSB. Here, `L` is the positive integer such
    /// that `code_polynomials[0]` is in the range `(2^(L-1), 2^L)`, so that `b_{0,0} = 1`.
    ///
    /// The input-output relationship of the state machine can then be described as follows. Let
    /// `x_{k}` be the input bit at time `k`, and let `y_{n,k}` be output bit `n` at time `k`. Then,
    /// `y_{n,k} = sum_{i=0}^{L-1} b_{n,i} * s_{k-i}`, where
    /// `s_{k} = b_{0,0} * x_{k} + sum_{i=1}^{L-1} b_{0,i} * s_{k-i}`. It can be easily seen that
    /// `y_{0,k} = x_{k}`, i.e., output bit `0` is the systematic bit.
    ///
    /// The bits `s_{k-1}`, `s_{k-2}`, ... , `s_{k-L+1}` constitute the binary representation of
    /// the state at time `k`, with `s_{k-1}` being the MSB. The output bits at time `k` are to be
    /// read out in the order `y_{0,k}`, `y_{1,k}`, ... , `y_{N-1,k}`.
    pub(crate) fn new(code_polynomials: &[usize]) -> Result<Self, Error> {
        let constraint_len = constraint_length(code_polynomials)?;
        Ok(Self {
            code_polynomials: code_polynomials.to_vec(),
            memory_len: constraint_len - 1,
            num_states: 1 << (constraint_len - 1),
            num_output_bits: code_polynomials.len(),
            output_bits: vec![Bit::Zero; code_polynomials.len()],
            branch_metric: BranchMetric::new(0.0, 0.0, 0.0),
            state: State(0),
        })
    }

    /// Generates output bits for either given input bit or next tail bit, and updates state.
    fn generate_output_bits(&mut self, input_bit: Option<Bit>) {
        self.output_bits.clear();
        let in_bit = match input_bit {
            Some(bit) => bit,
            None => self.msb_of_next_state(Bit::Zero),
        };
        let aug_state_index = self.augmented_state_index(self.msb_of_next_state(in_bit));
        for &code_poly in &self.code_polynomials {
            self.output_bits.push(bitxor(aug_state_index & code_poly));
        }
        self.state = State(aug_state_index >> 1);
    }

    /// Computes branch metric for given state transition.
    fn compute_branch_metric(
        &mut self,
        state: State,
        input_bit: Bit,
        input_bit_llr_prior: f64,
        output_bits_llr: &[f64],
    ) {
        // State transition
        self.state = state;
        self.generate_output_bits(Some(input_bit));
        // Branch metric computation
        self.branch_metric.prior = bit_metric_from_llr(input_bit, input_bit_llr_prior);
        self.branch_metric.systematic =
            bit_metric_from_llr(self.output_bits[0], output_bits_llr[0]);
        self.branch_metric.parity = 0.0;
        for (&bit, &bit_llr) in self.output_bits.iter().zip(output_bits_llr.iter()).skip(1) {
            self.branch_metric.parity += bit_metric_from_llr(bit, bit_llr);
        }
    }

    /// Returns MSB of next state for given input bit.
    fn msb_of_next_state(&self, input_bit: Bit) -> Bit {
        bitxor(self.augmented_state_index(input_bit) & self.code_polynomials[0])
    }

    /// Returns integer obtained by augmenting current state on the left with given input bit.
    fn augmented_state_index(&self, input_bit: Bit) -> usize {
        match input_bit {
            Bit::Zero => self.state.0,
            Bit::One => self.num_states + self.state.0,
        }
    }
}

/// Calculator for beta values
#[derive(Debug)]
struct BetaCalculator {
    /// Number of encoder states
    num_states: usize,
    /// Decoding algorithm to use
    decoding_algo: DecodingAlgo,
    /// Vector of beta values for all states after all information bits
    all_beta_val: Vec<f64>,
    /// Vector of beta values at previous time instant
    beta_val_prev: Vec<f64>,
}

impl BetaCalculator {
    /// Returns new calculator for beta values.
    fn new(num_states: usize, num_info_bits: usize, decoding_algo: DecodingAlgo) -> Self {
        Self {
            num_states,
            decoding_algo,
            all_beta_val: Vec::with_capacity(num_info_bits * num_states),
            beta_val_prev: Vec::with_capacity(num_states),
        }
    }

    /// Initializes beta values for all states after the last information bit.
    fn init_beta_values_after_last_info_bit(&mut self) {
        self.all_beta_val.clear();
        self.all_beta_val.push(0.0);
        for _ in 1 .. self.num_states {
            self.all_beta_val.push(-INF);
        }
    }

    /// Initializes beta values for all states at previous time instant.
    fn init_previous_beta_values(&mut self) {
        self.beta_val_prev.clear();
        for _ in 0 .. self.num_states {
            self.beta_val_prev.push(-INF);
        }
    }

    /// Updates beta value for a state at previous time instant.
    fn update_previous_beta_value(
        &mut self,
        state: State,
        branch_metric: &BranchMetric,
        next_state: State,
    ) {
        let num_beta_val = self.all_beta_val.len();
        self.beta_val_prev[state.0] = maxstar(
            self.beta_val_prev[state.0],
            branch_metric.value()
                + self.all_beta_val[num_beta_val - self.num_states + next_state.0],
            self.decoding_algo,
        );
    }

    /// Recenters beta values for all states at previous time instant.
    fn recenter_previous_beta_values(&mut self) {
        let beta_val_prev0 = self.beta_val_prev[0];
        self.beta_val_prev
            .iter_mut()
            .for_each(|x| *x -= beta_val_prev0);
    }

    /// Updates beta values for all states after the last information bit.
    fn update_beta_values_after_last_info_bit(&mut self) {
        std::mem::swap(&mut self.beta_val_prev, &mut self.all_beta_val);
    }

    /// Saves beta values for all states after an information bit.
    fn save_beta_values_after_info_bit(&mut self) {
        self.all_beta_val.extend(&self.beta_val_prev);
    }

    /// Deletes beta values for all states after an information bit.
    fn delete_beta_values_after_info_bit(&mut self) {
        for _ in 0 .. self.num_states {
            self.all_beta_val.pop();
        }
    }
}

/// Calculator for alpha values
#[derive(Debug)]
struct AlphaCalculator {
    /// Number of encoder states
    num_states: usize,
    /// Decoding algorithm to use
    decoding_algo: DecodingAlgo,
    /// Vector of alpha values for all states before an information bit
    alpha_val: Vec<f64>,
    /// Vector of alpha values at next time instant
    alpha_val_next: Vec<f64>,
}

impl AlphaCalculator {
    /// Returns new calculator for alpha values.
    fn new(num_states: usize, decoding_algo: DecodingAlgo) -> Self {
        Self {
            num_states,
            decoding_algo,
            alpha_val: Vec::with_capacity(num_states),
            alpha_val_next: Vec::with_capacity(num_states),
        }
    }

    /// Initializes alpha values for all states before an information bit.
    fn init_alpha_values_before_info_bit(&mut self) {
        self.alpha_val.clear();
        self.alpha_val.push(0.0);
        for _ in 1 .. self.num_states {
            self.alpha_val.push(-INF);
        }
    }

    /// Initializes alpha values for all states at next time instant.
    fn init_next_alpha_values(&mut self) {
        self.alpha_val_next.clear();
        for _ in 0 .. self.num_states {
            self.alpha_val_next.push(-INF);
        }
    }

    /// Updates alpha value for a state at next time instant.
    fn update_next_alpha_value(
        &mut self,
        state: State,
        branch_metric: &BranchMetric,
        next_state: State,
    ) {
        self.alpha_val_next[next_state.0] = maxstar(
            self.alpha_val_next[next_state.0],
            self.alpha_val[state.0] + branch_metric.value(),
            self.decoding_algo,
        );
    }

    /// Recenters alpha values for all states at next time instant.
    fn recenter_next_alpha_values(&mut self) {
        let alpha_val_next0 = self.alpha_val_next[0];
        self.alpha_val_next
            .iter_mut()
            .for_each(|x| *x -= alpha_val_next0);
    }

    /// Updates alpha values for all states before the next information bit.
    fn update_alpha_values_before_info_bit(&mut self) {
        std::mem::swap(&mut self.alpha_val, &mut self.alpha_val_next);
    }
}

/// Workspace for decoder
#[derive(Debug)]
pub(crate) struct DecoderWorkspace {
    /// Calculator for beta values of all states after all information bits
    beta_calc: BetaCalculator,
    /// Calculator for alpha values of all states before an information bit
    alpha_calc: AlphaCalculator,
    /// Decoding metric for bit `Zero`
    metric_for_zero: f64,
    /// Decoding metric for bit `One`
    metric_for_one: f64,
    /// Buffer for extrinsic information values for all information bits
    pub(crate) extrinsic_info: Vec<f64>,
    /// Buffer for posterior LLR values for all information bits
    pub(crate) llr_posterior: Vec<f64>,
}

impl DecoderWorkspace {
    /// Returns new workspace for decoder.
    pub(crate) fn new(
        num_states: usize,
        num_info_bits: usize,
        decoding_algo: DecodingAlgo,
    ) -> Self {
        Self {
            beta_calc: BetaCalculator::new(num_states, num_info_bits, decoding_algo),
            alpha_calc: AlphaCalculator::new(num_states, decoding_algo),
            metric_for_zero: -INF,
            metric_for_one: -INF,
            extrinsic_info: Vec::with_capacity(num_info_bits),
            llr_posterior: Vec::with_capacity(num_info_bits),
        }
    }

    /// Initializes metrics for an information bit being `Zero` and `One`.
    fn init_metrics_for_zero_and_one(&mut self) {
        self.metric_for_zero = -INF;
        self.metric_for_one = -INF;
    }

    /// Updates metric for an information bit being `Zero` or `One`.
    fn update_metric_for_zero_or_one(
        &mut self,
        state: State,
        info_bit: Bit,
        branch_metric: &BranchMetric,
        next_state: State,
    ) {
        let num_beta_val = self.beta_calc.all_beta_val.len();
        let beta_next =
            self.beta_calc.all_beta_val[num_beta_val - self.beta_calc.num_states + next_state.0];
        let cand_metric = self.alpha_calc.alpha_val[state.0] + branch_metric.parity + beta_next;
        match info_bit {
            Bit::Zero => {
                self.metric_for_zero = maxstar(
                    self.metric_for_zero,
                    cand_metric,
                    self.alpha_calc.decoding_algo,
                );
            }
            Bit::One => {
                self.metric_for_one = maxstar(
                    self.metric_for_one,
                    cand_metric,
                    self.alpha_calc.decoding_algo,
                );
            }
        };
    }
}

/// Returns constraint length corresponding to given code polynomials.
fn constraint_length(code_polynomials: &[usize]) -> Result<usize, Error> {
    if code_polynomials.len() < 2 {
        return Err(Error::InvalidInput(
            "Expected at least two code polynomials".to_string(),
        ));
    }
    let feedback_poly = code_polynomials[0];
    if feedback_poly == 0 || feedback_poly & (feedback_poly - 1) == 0 {
        return Err(Error::InvalidInput(
            "Feedback polynomial cannot be 0 or a power of 2".to_string(),
        ));
    }
    // OK to cast `u32` to `usize`: Numbers involved will always be small enough.
    let constraint_len = (usize::BITS - feedback_poly.leading_zeros()) as usize;
    let two_pow_constraint_len = 1 << constraint_len;
    if code_polynomials[1 ..]
        .iter()
        .any(|&x| x == 0 || x == feedback_poly || x >= two_pow_constraint_len)
    {
        return Err(Error::InvalidInput(format!(
            "For constraint length of {constraint_len}, each feedforward polynomial \
            must be in the range [1, {two_pow_constraint_len}), and cannot equal the \
            feedback polynomial {feedback_poly}",
        )));
    }
    Ok(constraint_len)
}

/// Returns XOR of bits in the binary representation of given integer.
fn bitxor(num: usize) -> Bit {
    match num.count_ones() % 2 {
        0 => Bit::Zero,
        _ => Bit::One,
    }
}

/// Returns bit corresponding to given index.
fn bit_from_index(bit_index: usize) -> Bit {
    match bit_index {
        0 => Bit::Zero,
        _ => Bit::One,
    }
}

/// Returns metric for given bit corresponding to given LLR value.
fn bit_metric_from_llr(bit: Bit, llr_val: f64) -> f64 {
    match bit {
        Bit::Zero => llr_val / 2.0,
        Bit::One => -llr_val / 2.0,
    }
}

/// Generates code bits from RSC encoder.
///
/// # Parameters
///
/// - `info_bits`: Information bits to be encoded.
///
/// - `state_machine`: State machine for the RSC encoder.
///
/// - `code_bits`: Vector to which code bits from the RSC encoder must be written (any pre-existing
///   elements will be cleared first). The number of code bits generated by the RSC encoder is
///   `(info_bits.len() + state_machine.memory_len) * state_machine.num_output_bits`.
pub(crate) fn encode(
    info_bits: &[Bit],
    state_machine: &mut StateMachine,
    code_bits: &mut Vec<Bit>,
) {
    state_machine.state = State(0);
    code_bits.clear();
    // Code bits corresponding to information bits
    for &info_bit in info_bits {
        state_machine.generate_output_bits(Some(info_bit));
        code_bits.extend(&state_machine.output_bits);
    }
    // Code bits corresponding to tail bits
    for _ in 0 .. state_machine.memory_len {
        state_machine.generate_output_bits(None);
        code_bits.extend(&state_machine.output_bits);
    }
}

/// Generates extrinsic information and posterior LLR values from RSC decoder.
///
/// Decoding is by the BCJR algorithm.
///
/// # Parameters
///
/// - `code_bits_llr`: Log-likelihood-ratio (LLR) values for the code bits.
///
/// - `info_bits_llr_prior`: Prior LLR values for the information bits.
///
/// - `state_machine`: State machine for the RSC decoder.
///
/// - `workspace`: Workspace for RSC decoder, to which extrinsic information and posterior LLR
///   values for the information bits must be written (any pre-existing elements will be cleared
///   first).
///
/// # Errors
///
/// Returns an error if `code_bits_llr.len()` does not equal
/// `(info_bits_llr_prior.len() + state_machine.memory_len) * state_machine.num_output_bits`.
pub(crate) fn decode(
    code_bits_llr: &[f64],
    info_bits_llr_prior: &[f64],
    state_machine: &mut StateMachine,
    workspace: &mut DecoderWorkspace,
) -> Result<(), Error> {
    let expected_code_bits_llr_len =
        (info_bits_llr_prior.len() + state_machine.memory_len) * state_machine.num_output_bits;
    if code_bits_llr.len() != expected_code_bits_llr_len {
        return Err(Error::InvalidInput(format!(
            "Wrong number of code bit LLR values (expected {expected_code_bits_llr_len})",
        )));
    }
    run_bcjr_backward_pass(code_bits_llr, info_bits_llr_prior, state_machine, workspace);
    run_bcjr_forward_pass(code_bits_llr, info_bits_llr_prior, state_machine, workspace);
    Ok(())
}

/// Runs backward pass through the trellis in the BCJR decoding algorithm.
fn run_bcjr_backward_pass(
    code_bits_llr: &[f64],
    info_bits_llr_prior: &[f64],
    state_machine: &mut StateMachine,
    workspace: &mut DecoderWorkspace,
) {
    let index_of_first_tail_code_bit = info_bits_llr_prior.len() * state_machine.num_output_bits;
    // Backward iterations to compute beta values after last information bit
    workspace.beta_calc.init_beta_values_after_last_info_bit();
    for rchunk in
        code_bits_llr[index_of_first_tail_code_bit ..].rchunks_exact(state_machine.num_output_bits)
    {
        compute_previous_beta_values(rchunk, 0.0, state_machine, workspace);
        workspace.beta_calc.update_beta_values_after_last_info_bit();
    }
    // Backward iterations to compute beta values after all other information bits
    for (&llr_prior, rchunk) in info_bits_llr_prior[1 ..].iter().rev().zip(
        code_bits_llr[.. index_of_first_tail_code_bit].rchunks_exact(state_machine.num_output_bits),
    ) {
        compute_previous_beta_values(rchunk, llr_prior, state_machine, workspace);
        workspace.beta_calc.save_beta_values_after_info_bit();
    }
}

/// Runs forward pass through the trellis in the BCJR decoding algorithm.
fn run_bcjr_forward_pass(
    code_bits_llr: &[f64],
    info_bits_llr_prior: &[f64],
    state_machine: &mut StateMachine,
    workspace: &mut DecoderWorkspace,
) {
    let code_bits_llr_chunks = code_bits_llr.chunks_exact(state_machine.num_output_bits);
    // Forward iterations to compute alpha values before each information bit, and thereby to
    // obtain and save extrinsic information and posterior LLR for each information bit
    workspace.alpha_calc.init_alpha_values_before_info_bit();
    workspace.extrinsic_info.clear();
    workspace.llr_posterior.clear();
    for (&llr_prior, chunk) in info_bits_llr_prior.iter().zip(code_bits_llr_chunks) {
        compute_next_alpha_values(chunk, llr_prior, state_machine, workspace);
        let extrinsic_info = workspace.metric_for_zero - workspace.metric_for_one;
        let llr_posterior = chunk[0] + llr_prior + extrinsic_info;
        workspace.extrinsic_info.push(extrinsic_info);
        workspace.llr_posterior.push(llr_posterior);
        workspace.beta_calc.delete_beta_values_after_info_bit();
        workspace.alpha_calc.update_alpha_values_before_info_bit();
    }
}

/// Computes beta values for all states at previous time instant.
fn compute_previous_beta_values(
    code_bits_llr: &[f64],
    input_bit_llr_prior: f64,
    state_machine: &mut StateMachine,
    workspace: &mut DecoderWorkspace,
) {
    // Backward iteration
    workspace.beta_calc.init_previous_beta_values();
    for state_index in 0 .. state_machine.num_states {
        let state = State(state_index);
        for bit_index in 0 .. 2 {
            let input_bit = bit_from_index(bit_index);
            state_machine.compute_branch_metric(
                state,
                input_bit,
                input_bit_llr_prior,
                code_bits_llr,
            );
            workspace.beta_calc.update_previous_beta_value(
                state,
                &state_machine.branch_metric,
                state_machine.state,
            );
        }
    }
    workspace.beta_calc.recenter_previous_beta_values();
}

/// Computes alpha values for all states at next time instant.
fn compute_next_alpha_values(
    code_bits_llr: &[f64],
    input_bit_llr_prior: f64,
    state_machine: &mut StateMachine,
    workspace: &mut DecoderWorkspace,
) {
    // Forward iteration
    workspace.alpha_calc.init_next_alpha_values();
    workspace.init_metrics_for_zero_and_one();
    for state_index in 0 .. state_machine.num_states {
        let state = State(state_index);
        for bit_index in 0 .. 2 {
            let input_bit = bit_from_index(bit_index);
            state_machine.compute_branch_metric(
                state,
                input_bit,
                input_bit_llr_prior,
                code_bits_llr,
            );
            workspace.update_metric_for_zero_or_one(
                state,
                input_bit,
                &state_machine.branch_metric,
                state_machine.state,
            );
            workspace.alpha_calc.update_next_alpha_value(
                state,
                &state_machine.branch_metric,
                state_machine.state,
            );
        }
    }
    workspace.alpha_calc.recenter_next_alpha_values();
}

/// Returns the maxstar of two numbers for given decoding algorithm.
fn maxstar(x: f64, y: f64, decoding_algo: DecodingAlgo) -> f64 {
    x.max(y)
        + match decoding_algo {
            DecodingAlgo::MaxLogMAP(_) => 0.0,
            DecodingAlgo::LinearLogMAP(_) => linear_log_map_correction_term((x - y).abs()),
            DecodingAlgo::LogMAP(_) => log_map_correction_term((x - y).abs()),
        }
}

/// Returns the correction term for Linear-Log-MAP decoding algorithm (Valenti & Sun, 2001).
fn linear_log_map_correction_term(abs_diff: f64) -> f64 {
    let thresh = 2.506_816_400_220_01;
    if abs_diff > thresh {
        0.0
    } else {
        let slope = -0.249_041_818_917_1;
        slope * (abs_diff - thresh)
    }
}

/// Returns the correction term for Log-MAP decoding algorithm.
fn log_map_correction_term(abs_diff: f64) -> f64 {
    (-abs_diff).exp().ln_1p()
}

#[cfg(test)]
mod tests_of_state_machine {
    use super::*;
    use Bit::{One, Zero};

    #[test]
    fn test_new() {
        let state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        assert_eq!(state_machine.code_polynomials, [0o13, 0o15, 0o17]);
        assert_eq!(state_machine.memory_len, 3);
        assert_eq!(state_machine.num_states, 8);
        assert_eq!(state_machine.num_output_bits, 3);
        assert_eq!(state_machine.output_bits, [Zero, Zero, Zero]);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(0.0, 0.0, 0.0)
        );
        assert_eq!(state_machine.state, State(0));
    }

    #[test]
    fn test_generate_output_bits() {
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let correct_output_bits_for_zero = [
            [Zero, Zero, Zero],
            [Zero, Zero, Zero],
            [Zero, One, Zero],
            [Zero, One, Zero],
            [Zero, One, One],
            [Zero, One, One],
            [Zero, Zero, One],
            [Zero, Zero, One],
        ];
        let correct_next_state_for_zero = [0, 4, 5, 1, 2, 6, 7, 3];
        let correct_output_bits_for_one = [
            [One, One, One],
            [One, One, One],
            [One, Zero, One],
            [One, Zero, One],
            [One, Zero, Zero],
            [One, Zero, Zero],
            [One, One, Zero],
            [One, One, Zero],
        ];
        let correct_next_state_for_one = [4, 0, 1, 5, 6, 2, 3, 7];
        for state_index in 0 .. state_machine.num_states {
            // Input bit of `Zero`
            state_machine.state = State(state_index);
            state_machine.generate_output_bits(Some(Zero));
            assert_eq!(
                state_machine.output_bits,
                correct_output_bits_for_zero[state_index]
            );
            assert_eq!(
                state_machine.state,
                State(correct_next_state_for_zero[state_index])
            );
            // Input bit of `One`
            state_machine.state = State(state_index);
            state_machine.generate_output_bits(Some(One));
            assert_eq!(
                state_machine.output_bits,
                correct_output_bits_for_one[state_index]
            );
            assert_eq!(
                state_machine.state,
                State(correct_next_state_for_one[state_index])
            );
            // Tail bits
            state_machine.state = State(state_index);
            for _ in 0 .. state_machine.memory_len {
                state_machine.generate_output_bits(None);
            }
            assert_eq!(state_machine.state, State(0));
        }
    }

    #[test]
    fn test_compute_branch_metric() {
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let input_bit_llr_prior = 2.0;
        let output_bits_llr = [-8.0, 4.0, 16.0];
        state_machine.compute_branch_metric(State(0), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, 10.0)
        );
        assert_eq!(state_machine.state, State(0));
        state_machine.compute_branch_metric(State(0), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, -10.0)
        );
        assert_eq!(state_machine.state, State(4));
        state_machine.compute_branch_metric(State(1), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, 10.0)
        );
        assert_eq!(state_machine.state, State(4));
        state_machine.compute_branch_metric(State(1), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, -10.0)
        );
        assert_eq!(state_machine.state, State(0));
        state_machine.compute_branch_metric(State(2), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, 6.0)
        );
        assert_eq!(state_machine.state, State(5));
        state_machine.compute_branch_metric(State(2), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, -6.0)
        );
        assert_eq!(state_machine.state, State(1));
        state_machine.compute_branch_metric(State(3), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, 6.0)
        );
        assert_eq!(state_machine.state, State(1));
        state_machine.compute_branch_metric(State(3), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, -6.0)
        );
        assert_eq!(state_machine.state, State(5));
        state_machine.compute_branch_metric(State(4), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, -10.0)
        );
        assert_eq!(state_machine.state, State(2));
        state_machine.compute_branch_metric(State(4), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, 10.0)
        );
        assert_eq!(state_machine.state, State(6));
        state_machine.compute_branch_metric(State(5), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, -10.0)
        );
        assert_eq!(state_machine.state, State(6));
        state_machine.compute_branch_metric(State(5), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, 10.0)
        );
        assert_eq!(state_machine.state, State(2));
        state_machine.compute_branch_metric(State(6), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, -6.0)
        );
        assert_eq!(state_machine.state, State(7));
        state_machine.compute_branch_metric(State(6), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, 6.0)
        );
        assert_eq!(state_machine.state, State(3));
        state_machine.compute_branch_metric(State(7), Zero, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(1.0, -4.0, -6.0)
        );
        assert_eq!(state_machine.state, State(3));
        state_machine.compute_branch_metric(State(7), One, input_bit_llr_prior, &output_bits_llr);
        assert_eq!(
            state_machine.branch_metric,
            BranchMetric::new(-1.0, 4.0, 6.0)
        );
        assert_eq!(state_machine.state, State(7));
    }

    #[test]
    fn test_msb_of_next_state() {
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        state_machine.state = State(0);
        assert_eq!(state_machine.msb_of_next_state(Zero), Zero);
        assert_eq!(state_machine.msb_of_next_state(One), One);
        state_machine.state = State(1);
        assert_eq!(state_machine.msb_of_next_state(Zero), One);
        assert_eq!(state_machine.msb_of_next_state(One), Zero);
        state_machine.state = State(2);
        assert_eq!(state_machine.msb_of_next_state(Zero), One);
        assert_eq!(state_machine.msb_of_next_state(One), Zero);
        state_machine.state = State(3);
        assert_eq!(state_machine.msb_of_next_state(Zero), Zero);
        assert_eq!(state_machine.msb_of_next_state(One), One);
        state_machine.state = State(4);
        assert_eq!(state_machine.msb_of_next_state(Zero), Zero);
        assert_eq!(state_machine.msb_of_next_state(One), One);
        state_machine.state = State(5);
        assert_eq!(state_machine.msb_of_next_state(Zero), One);
        assert_eq!(state_machine.msb_of_next_state(One), Zero);
        state_machine.state = State(6);
        assert_eq!(state_machine.msb_of_next_state(Zero), One);
        assert_eq!(state_machine.msb_of_next_state(One), Zero);
        state_machine.state = State(7);
        assert_eq!(state_machine.msb_of_next_state(Zero), Zero);
        assert_eq!(state_machine.msb_of_next_state(One), One);
    }

    #[test]
    fn test_augmented_state_index() {
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        for state_index in 0 .. state_machine.num_states {
            state_machine.state = State(state_index);
            assert_eq!(state_machine.augmented_state_index(Zero), state_index);
            assert_eq!(
                state_machine.augmented_state_index(One),
                state_machine.num_states + state_index
            );
        }
    }
}

#[cfg(test)]
mod tests_of_functions {
    use super::*;
    use float_eq::assert_float_eq;
    use Bit::{One, Zero};

    #[test]
    fn test_constraint_length() {
        assert!(constraint_length(&[]).is_err());
        assert!(constraint_length(&[0o13]).is_err());
        assert!(constraint_length(&[0o0, 0o15]).is_err());
        assert!(constraint_length(&[0o20, 0o15]).is_err());
        assert!(constraint_length(&[0o13, 0o0]).is_err());
        assert!(constraint_length(&[0o13, 0o20]).is_err());
        assert_eq!(constraint_length(&[0o11, 0o15]).unwrap(), 4);
        assert_eq!(constraint_length(&[0o13, 0o15]).unwrap(), 4);
        assert_eq!(constraint_length(&[0o17, 0o15]).unwrap(), 4);
    }

    #[test]
    fn test_bitxor() {
        assert_eq!(bitxor(0x0), Zero);
        assert_eq!(bitxor(0x1), One);
        assert_eq!(bitxor(0x2), One);
        assert_eq!(bitxor(0x3), Zero);
        assert_eq!(bitxor(0x4), One);
        assert_eq!(bitxor(0x5), Zero);
        assert_eq!(bitxor(0x6), Zero);
        assert_eq!(bitxor(0x7), One);
        assert_eq!(bitxor(0x8), One);
        assert_eq!(bitxor(0x9), Zero);
        assert_eq!(bitxor(0xA), Zero);
        assert_eq!(bitxor(0xB), One);
        assert_eq!(bitxor(0xC), Zero);
        assert_eq!(bitxor(0xD), One);
        assert_eq!(bitxor(0xE), One);
        assert_eq!(bitxor(0xF), Zero);
    }

    #[test]
    fn test_bit_from_index() {
        assert_eq!(bit_from_index(0), Zero);
        assert_eq!(bit_from_index(1), One);
    }

    #[test]
    fn test_bit_metric_from_llr() {
        assert_float_eq!(bit_metric_from_llr(Zero, 2.0), 1.0, abs <= 1e-8);
        assert_float_eq!(bit_metric_from_llr(One, 2.0), -1.0, abs <= 1e-8);
    }

    fn correct_branch_metric_values(code_bits_llr: &[f64; 3], in_bit_llr_prior: f64) -> [f64; 4] {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]`
        // Element `k` is for transition from state `2k` or `2k+1` when input bit is `Zero`
        [
            (in_bit_llr_prior + code_bits_llr[0] + code_bits_llr[1] + code_bits_llr[2]) / 2.0,
            (in_bit_llr_prior + code_bits_llr[0] - code_bits_llr[1] + code_bits_llr[2]) / 2.0,
            (in_bit_llr_prior + code_bits_llr[0] - code_bits_llr[1] - code_bits_llr[2]) / 2.0,
            (in_bit_llr_prior + code_bits_llr[0] + code_bits_llr[1] - code_bits_llr[2]) / 2.0,
        ]
    }

    fn correct_beta_val_prev(
        code_bits_llr: &[f64; 3],
        in_bit_llr_prior: f64,
        beta_val: &[f64; 8],
    ) -> [f64; 8] {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]` and Max-Log-MAP decoding
        let [v01, v23, v45, v67] = correct_branch_metric_values(code_bits_llr, in_bit_llr_prior);
        let cv0 = (v01 + beta_val[0]).max(-v01 + beta_val[4]);
        [
            (v01 + beta_val[0]).max(-v01 + beta_val[4]) - cv0,
            (v01 + beta_val[4]).max(-v01 + beta_val[0]) - cv0,
            (v23 + beta_val[5]).max(-v23 + beta_val[1]) - cv0,
            (v23 + beta_val[1]).max(-v23 + beta_val[5]) - cv0,
            (v45 + beta_val[2]).max(-v45 + beta_val[6]) - cv0,
            (v45 + beta_val[6]).max(-v45 + beta_val[2]) - cv0,
            (v67 + beta_val[7]).max(-v67 + beta_val[3]) - cv0,
            (v67 + beta_val[3]).max(-v67 + beta_val[7]) - cv0,
        ]
    }

    fn correct_all_beta_val(
        code_bits_llr: &[f64; 21],
        info_bits_llr_prior: &[f64; 4],
    ) -> [f64; 32] {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]` and Max-Log-MAP decoding
        let bv6 = [0.0, -INF, -INF, -INF, -INF, -INF, -INF, -INF];
        let bv5 = correct_beta_val_prev(
            &[code_bits_llr[18], code_bits_llr[19], code_bits_llr[20]],
            0.0,
            &bv6,
        );
        let bv4 = correct_beta_val_prev(
            &[code_bits_llr[15], code_bits_llr[16], code_bits_llr[17]],
            0.0,
            &bv5,
        );
        let bv3 = correct_beta_val_prev(
            &[code_bits_llr[12], code_bits_llr[13], code_bits_llr[14]],
            0.0,
            &bv4,
        );
        let bv2 = correct_beta_val_prev(
            &[code_bits_llr[9], code_bits_llr[10], code_bits_llr[11]],
            info_bits_llr_prior[3],
            &bv3,
        );
        let bv1 = correct_beta_val_prev(
            &[code_bits_llr[6], code_bits_llr[7], code_bits_llr[8]],
            info_bits_llr_prior[2],
            &bv2,
        );
        let bv0 = correct_beta_val_prev(
            &[code_bits_llr[3], code_bits_llr[4], code_bits_llr[5]],
            info_bits_llr_prior[1],
            &bv1,
        );
        [
            bv3[0], bv3[1], bv3[2], bv3[3], bv3[4], bv3[5], bv3[6], bv3[7], bv2[0], bv2[1], bv2[2],
            bv2[3], bv2[4], bv2[5], bv2[6], bv2[7], bv1[0], bv1[1], bv1[2], bv1[3], bv1[4], bv1[5],
            bv1[6], bv1[7], bv0[0], bv0[1], bv0[2], bv0[3], bv0[4], bv0[5], bv0[6], bv0[7],
        ]
    }

    fn correct_alpha_val_next(
        alpha_val: &[f64; 8],
        code_bits_llr: &[f64; 3],
        in_bit_llr_prior: f64,
    ) -> [f64; 8] {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]` and Max-Log-MAP decoding
        let [v01, v23, v45, v67] = correct_branch_metric_values(code_bits_llr, in_bit_llr_prior);
        let cv0 = (alpha_val[0] + v01).max(alpha_val[1] - v01);
        [
            (alpha_val[0] + v01).max(alpha_val[1] - v01) - cv0,
            (alpha_val[3] + v23).max(alpha_val[2] - v23) - cv0,
            (alpha_val[4] + v45).max(alpha_val[5] - v45) - cv0,
            (alpha_val[7] + v67).max(alpha_val[6] - v67) - cv0,
            (alpha_val[1] + v01).max(alpha_val[0] - v01) - cv0,
            (alpha_val[2] + v23).max(alpha_val[3] - v23) - cv0,
            (alpha_val[5] + v45).max(alpha_val[4] - v45) - cv0,
            (alpha_val[6] + v67).max(alpha_val[7] - v67) - cv0,
        ]
    }

    fn correct_metrics_for_zero_and_one(
        alpha_val: &[f64; 8],
        code_bits_llr: &[f64; 3],
        beta_val: &[f64; 8],
    ) -> (f64, f64) {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]` and Max-Log-MAP decoding
        let p01 = (code_bits_llr[1] + code_bits_llr[2]) / 2.0;
        let p23 = (-code_bits_llr[1] + code_bits_llr[2]) / 2.0;
        let p45 = (-code_bits_llr[1] - code_bits_llr[2]) / 2.0;
        let p67 = (code_bits_llr[1] - code_bits_llr[2]) / 2.0;
        let correct_metric_for_zero = (-INF)
            .max(alpha_val[0] + p01 + beta_val[0])
            .max(alpha_val[1] + p01 + beta_val[4])
            .max(alpha_val[2] + p23 + beta_val[5])
            .max(alpha_val[3] + p23 + beta_val[1])
            .max(alpha_val[4] + p45 + beta_val[2])
            .max(alpha_val[5] + p45 + beta_val[6])
            .max(alpha_val[6] + p67 + beta_val[7])
            .max(alpha_val[7] + p67 + beta_val[3]);
        let correct_metric_for_one = (-INF)
            .max(alpha_val[0] - p01 + beta_val[4])
            .max(alpha_val[1] - p01 + beta_val[0])
            .max(alpha_val[2] - p23 + beta_val[1])
            .max(alpha_val[3] - p23 + beta_val[5])
            .max(alpha_val[4] - p45 + beta_val[6])
            .max(alpha_val[5] - p45 + beta_val[2])
            .max(alpha_val[6] - p67 + beta_val[3])
            .max(alpha_val[7] - p67 + beta_val[7]);
        (correct_metric_for_zero, correct_metric_for_one)
    }

    fn correct_extrinsic_info(
        code_bits_llr: &[f64; 21],
        info_bits_llr_prior: &[f64; 4],
    ) -> [f64; 4] {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]` and Max-Log-MAP decoding
        let abv = correct_all_beta_val(code_bits_llr, info_bits_llr_prior);
        // Bit 0
        let av0 = [0.0, -INF, -INF, -INF, -INF, -INF, -INF, -INF];
        let (m00, m01) = correct_metrics_for_zero_and_one(
            &av0,
            &[code_bits_llr[0], code_bits_llr[1], code_bits_llr[2]],
            &[
                abv[24], abv[25], abv[26], abv[27], abv[28], abv[29], abv[30], abv[31],
            ],
        );
        // Bit 1
        let av1 = correct_alpha_val_next(
            &av0,
            &[code_bits_llr[0], code_bits_llr[1], code_bits_llr[2]],
            info_bits_llr_prior[0],
        );
        let (m10, m11) = correct_metrics_for_zero_and_one(
            &av1,
            &[code_bits_llr[3], code_bits_llr[4], code_bits_llr[5]],
            &[
                abv[16], abv[17], abv[18], abv[19], abv[20], abv[21], abv[22], abv[23],
            ],
        );
        // Bit 2
        let av2 = correct_alpha_val_next(
            &av1,
            &[code_bits_llr[3], code_bits_llr[4], code_bits_llr[5]],
            info_bits_llr_prior[1],
        );
        let (m20, m21) = correct_metrics_for_zero_and_one(
            &av2,
            &[code_bits_llr[6], code_bits_llr[7], code_bits_llr[8]],
            &[
                abv[8], abv[9], abv[10], abv[11], abv[12], abv[13], abv[14], abv[15],
            ],
        );
        // Bit 3
        let av3 = correct_alpha_val_next(
            &av2,
            &[code_bits_llr[6], code_bits_llr[7], code_bits_llr[8]],
            info_bits_llr_prior[2],
        );
        let (m30, m31) = correct_metrics_for_zero_and_one(
            &av3,
            &[code_bits_llr[9], code_bits_llr[10], code_bits_llr[11]],
            &[
                abv[0], abv[1], abv[2], abv[3], abv[4], abv[5], abv[6], abv[7],
            ],
        );
        [m00 - m01, m10 - m11, m20 - m21, m30 - m31]
    }

    fn correct_llr_posterior(
        code_bits_llr: &[f64; 21],
        info_bits_llr_prior: &[f64; 4],
    ) -> [f64; 4] {
        // NOTE: Assuming code polynomials are `[0o13, 0o15, 0o17]` and Max-Log-MAP decoding
        let [e0, e1, e2, e3] = correct_extrinsic_info(code_bits_llr, info_bits_llr_prior);
        [
            info_bits_llr_prior[0] + code_bits_llr[0] + e0,
            info_bits_llr_prior[1] + code_bits_llr[3] + e1,
            info_bits_llr_prior[2] + code_bits_llr[6] + e2,
            info_bits_llr_prior[3] + code_bits_llr[9] + e3,
        ]
    }

    #[test]
    fn test_encode() {
        let info_bits = [Zero, One, One, Zero];
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let mut code_bits = Vec::with_capacity(
            (info_bits.len() + state_machine.memory_len) * state_machine.num_output_bits,
        );
        encode(&info_bits, &mut state_machine, &mut code_bits);
        let correct_code_bits = [
            Zero, Zero, Zero, One, One, One, One, Zero, Zero, Zero, Zero, One, Zero, Zero, One,
            Zero, One, Zero, One, One, One,
        ];
        assert_eq!(code_bits, correct_code_bits);
    }

    #[test]
    fn test_decode() {
        let code_bits_llr = [
            1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
            -1.0, 1.0, -1.0, -1.0, -1.0,
        ];
        let mut info_bits_llr_prior = [-1.0, -5.0, 1.0, 3.0];
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let mut workspace = DecoderWorkspace::new(
            state_machine.num_states,
            info_bits_llr_prior.len(),
            DecodingAlgo::MaxLogMAP(0),
        );
        for idx in 0 .. 8 {
            if idx > 0 {
                info_bits_llr_prior.copy_from_slice(&workspace.extrinsic_info);
            }
            decode(
                &code_bits_llr,
                &info_bits_llr_prior,
                &mut state_machine,
                &mut workspace,
            )
            .unwrap();
            assert_float_eq!(
                workspace.extrinsic_info,
                correct_extrinsic_info(&code_bits_llr, &info_bits_llr_prior).to_vec(),
                abs_all <= 1e-8
            );
            assert_float_eq!(
                workspace.llr_posterior,
                correct_llr_posterior(&code_bits_llr, &info_bits_llr_prior).to_vec(),
                abs_all <= 1e-8
            );
        }
    }

    #[test]
    fn test_run_bcjr_backward_pass() {
        let code_bits_llr = [
            10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0, -10.0, 10.0,
            10.0, -10.0, 10.0, -10.0, 10.0, -10.0, -10.0, -10.0,
        ];
        let info_bits_llr_prior = [-1.0, -5.0, 1.0, 3.0];
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let mut workspace = DecoderWorkspace::new(
            state_machine.num_states,
            info_bits_llr_prior.len(),
            DecodingAlgo::MaxLogMAP(0),
        );
        run_bcjr_backward_pass(
            &code_bits_llr,
            &info_bits_llr_prior,
            &mut state_machine,
            &mut workspace,
        );
        assert_float_eq!(
            workspace.beta_calc.all_beta_val,
            correct_all_beta_val(&code_bits_llr, &info_bits_llr_prior).to_vec(),
            abs_all <= 1e-8
        );
    }

    #[test]
    fn test_run_bcjr_forward_pass() {
        let code_bits_llr = [
            10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0, -10.0, 10.0,
            10.0, -10.0, 10.0, -10.0, 10.0, -10.0, -10.0, -10.0,
        ];
        let info_bits_llr_prior = [-1.0, -5.0, 1.0, 3.0];
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let mut workspace = DecoderWorkspace::new(
            state_machine.num_states,
            info_bits_llr_prior.len(),
            DecodingAlgo::MaxLogMAP(0),
        );
        workspace
            .beta_calc
            .all_beta_val
            .extend_from_slice(&correct_all_beta_val(&code_bits_llr, &info_bits_llr_prior));
        run_bcjr_forward_pass(
            &code_bits_llr,
            &info_bits_llr_prior,
            &mut state_machine,
            &mut workspace,
        );
        assert_float_eq!(
            workspace.extrinsic_info,
            correct_extrinsic_info(&code_bits_llr, &info_bits_llr_prior).to_vec(),
            abs_all <= 1e-8
        );
        assert_float_eq!(
            workspace.llr_posterior,
            correct_llr_posterior(&code_bits_llr, &info_bits_llr_prior).to_vec(),
            abs_all <= 1e-8
        );
    }

    #[test]
    fn test_compute_previous_beta_values() {
        let code_bits_llr = [-8.0, 4.0, 16.0];
        let in_bit_llr_prior = -1.0;
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let mut workspace =
            DecoderWorkspace::new(state_machine.num_states, 1, DecodingAlgo::MaxLogMAP(0));
        let beta_val = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        workspace.beta_calc.all_beta_val.extend(&beta_val);
        compute_previous_beta_values(
            &code_bits_llr,
            in_bit_llr_prior,
            &mut state_machine,
            &mut workspace,
        );
        assert_float_eq!(
            workspace.beta_calc.beta_val_prev,
            correct_beta_val_prev(&code_bits_llr, in_bit_llr_prior, &beta_val).to_vec(),
            abs_all <= 1e-8
        );
    }

    #[test]
    fn test_compute_next_alpha_values() {
        let code_bits_llr = [-8.0, 4.0, 16.0];
        let in_bit_llr_prior = -1.0;
        let mut state_machine = StateMachine::new(&[0o13, 0o15, 0o17]).unwrap();
        let mut workspace =
            DecoderWorkspace::new(state_machine.num_states, 1, DecodingAlgo::MaxLogMAP(0));
        let beta_val = [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0];
        workspace.beta_calc.all_beta_val.extend(&beta_val);
        let alpha_val = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        workspace.alpha_calc.alpha_val.extend(&alpha_val);
        compute_next_alpha_values(
            &code_bits_llr,
            in_bit_llr_prior,
            &mut state_machine,
            &mut workspace,
        );
        assert_float_eq!(
            workspace.alpha_calc.alpha_val_next,
            correct_alpha_val_next(&alpha_val, &code_bits_llr, in_bit_llr_prior).to_vec(),
            abs_all <= 1e-8
        );
        let (correct_metric_for_zero, correct_metric_for_one) =
            correct_metrics_for_zero_and_one(&alpha_val, &code_bits_llr, &beta_val);
        assert_float_eq!(
            workspace.metric_for_zero,
            correct_metric_for_zero,
            abs <= 1e-8
        );
        assert_float_eq!(
            workspace.metric_for_one,
            correct_metric_for_one,
            abs <= 1e-8
        );
    }

    #[test]
    fn test_maxstar() {
        assert_float_eq!(
            maxstar(1.2, 1.3, DecodingAlgo::MaxLogMAP(0)),
            1.3,
            abs <= 1e-8
        );
        assert_float_eq!(
            maxstar(-1.2, -1.3, DecodingAlgo::MaxLogMAP(0)),
            -1.2,
            abs <= 1e-8
        );
        assert_float_eq!(
            maxstar(1.3, -1.3, DecodingAlgo::LinearLogMAP(0)),
            1.3,
            abs <= 1e-8
        );
        assert_float_eq!(
            maxstar(-1.2, 1.2, DecodingAlgo::LinearLogMAP(0)),
            1.226_601_750_600_968_3,
            abs <= 1e-8
        );
        assert_float_eq!(
            maxstar(1.2, 1.3, DecodingAlgo::LogMAP(0)),
            1.944_396_660_073_571,
            abs <= 1e-8
        );
        assert_float_eq!(
            maxstar(-1.2, -1.3, DecodingAlgo::LogMAP(0)),
            -0.555_603_339_926_429_1,
            abs <= 1e-8
        );
    }

    #[test]
    fn test_linear_log_map_correction_term() {
        assert_float_eq!(linear_log_map_correction_term(2.6), 0.0, abs <= 1e-8);
        assert_float_eq!(
            linear_log_map_correction_term(2.4),
            0.026_601_750_600_968_28,
            abs <= 1e-8
        );
    }

    #[test]
    fn test_log_map_correction_term() {
        assert_float_eq!(
            log_map_correction_term(2.6),
            0.071_644_691_967_669_72,
            abs <= 1e-8
        );
        assert_float_eq!(
            log_map_correction_term(2.4),
            0.086_836_152_153_949_63,
            abs <= 1e-8
        );
    }
}
