/// Hyper-parameters of Ising's message passing algorithms
#[derive(Debug, Clone, Copy)]
pub struct IsingFactorHyperParameters {
    /// Inverse temperature
    pub beta: f64,

    /// Exponential moving average coefficient
    pub gamma: f64,
}

/// Returns a scheduler for messages update rule of an Ising factor
/// with exponentially changing inverse temperature
///
/// # Arguments
///
/// * `beta_start` - Initial inverse temperature
/// * `beta_end` - Inverse temperature after one epoch of iterations
/// * `iterations_number` - Number of iterations passed from `beta_start` to `beta_end`
/// * `gamma` - Exponential moving average coefficient
pub fn get_exponential_factor_scheduler(
    beta_start: f64,
    beta_end: f64,
    iterations_number: usize,
    gamma: f64,
) -> impl Fn(usize) -> IsingFactorHyperParameters {
    let coeff = (beta_end / beta_start).powf(1f64 / iterations_number as f64);
    move |iter| {
        let beta = coeff.powi(iter as i32) * beta_start;
        IsingFactorHyperParameters { beta, gamma }
    }
}

/// Returns a scheduler for messages update rule of an Ising factor
/// with inverse temperature = 1
///
/// # Arguments
///
/// * `gamma` - Exponential moving average coefficient
pub fn get_standard_factor_scheduler(gamma: f64) -> impl Fn(usize) -> IsingFactorHyperParameters {
    move |_| IsingFactorHyperParameters { beta: 1f64, gamma }
}

/// Returns a scheduler for messages update rule of and Ising variable
///
/// # Arguments
///
/// * `gamma` - exponential moving average coefficient
pub fn get_standard_variable_scheduler(gamma: f64) -> impl Fn(usize) -> f64 {
    move |_| gamma
}
