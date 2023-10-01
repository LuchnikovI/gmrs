use clap::Parser;
use gmrs::{
    core::FGError,
    ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct},
};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// This part is for parsing the parameters from command line,
// can be safely skipped, does not affect the understanding
// of the example
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Inverse temperature of the SK system
    #[arg(short, long)]
    beta: f64,

    /// Number of spins in the SK system
    #[arg(short, long, default_value = "1000")]
    spins_number: usize,

    /// Maximal number of iterations
    #[arg(short, long, default_value = "1000")]
    max_iter: usize,

    /// Threshold of the convergence criterion
    #[arg(short, long, default_value = "1e-6")]
    threshold: f64,

    /// Sum-product decay hyper parameter
    #[arg(short, long, default_value = "0.5")]
    decay: f64,
}

// This part if for serialization of the output result into
// a yaml config. Can be safely skipped. Does not affect the
// understanding of the example
#[derive(Serialize, Deserialize)]
struct ExampleResult {
    is_converged: bool,
    iterations_number: usize,
    discrepancy: f64,
    bethe_free_entropy: f64,
    replica_symmetric_free_entropy: f64,
}

/// Returns a replica symmetric free entropy
/// (valid for high temperature, beta < 1)
fn rs_sk_free_entropy(beta: f64) -> f64 {
    0.25 * beta.powi(2) + f64::ln(2f64)
}

fn main() {
    let cli = Cli::parse();

    // parameters --------------------------------------------------------------------------

    // number of spins in SK model
    let spins_number = cli.spins_number;
    // inverse temperature
    let beta = cli.beta;
    // threshold of the convergence criterion
    let error = cli.threshold;
    // sum-product decay hyper parameter
    let decay = cli.decay;
    // maximal number of message passing iterations
    let max_iter = cli.max_iter;
    // mean value of coupling constants
    let mu = 0f64;
    // std of coupling constants
    let std = beta / (spins_number as f64).sqrt();
    // distribution of coupling constants
    let distr = Normal::new(mu, std).unwrap();
    // this generator is used to sample coupling constants
    let mut rng_couplings = thread_rng();
    // messages initializer
    let mut initializer = random_message_initializer(thread_rng());

    // -------------------------------------------------------------------------------------

    let mut fgb =
        new_ising_builder::<SumProduct>(spins_number, (spins_number - 1) * spins_number / 2);
    for i in 0..spins_number {
        for j in (i + 1)..spins_number {
            fgb.add_factor(
                IsingFactor::new(distr.sample(&mut rng_couplings), 0f64, 0f64),
                &[i, j],
                &mut initializer,
            )
            .unwrap();
        }
    }
    let mut fg = fgb.build();
    let info = fg.run_message_passing_parallel(max_iter, error, &decay);
    let variable_marginals = fg.variable_marginals();
    let factors = fg.factors();
    let factor_marginals = fg.factor_marginals();
    let mut bethe_free_entropy = 0f64;
    for (fm, f) in factor_marginals.iter().zip(&factors) {
        bethe_free_entropy -= (fm * (fm / f).mapv(f64::ln)).sum();
    }
    for vm in &variable_marginals {
        bethe_free_entropy += ((spins_number - 2) as f64) * (vm * vm.mapv(f64::ln)).sum();
    }
    bethe_free_entropy /= spins_number as f64;
    let replica_symmetric_free_entropy = rs_sk_free_entropy(beta);
    let (is_converged, iterations_number, discrepancy) = match info {
        Ok(info) => (true, info.iterations_number, info.final_discrepancy),
        Err(err) => {
            if let FGError::MessagePassingError {
                iterations_number,
                final_discrepancy,
                ..
            } = err
            {
                (false, iterations_number, final_discrepancy)
            } else {
                unreachable!()
            }
        }
    };
    let example_result = ExampleResult {
        is_converged,
        iterations_number,
        discrepancy,
        bethe_free_entropy,
        replica_symmetric_free_entropy,
    };
    println!("{}", serde_yaml::to_string(&example_result).unwrap());
}
