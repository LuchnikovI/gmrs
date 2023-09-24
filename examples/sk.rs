use gmrs::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

fn main() {
    // parameters --------------------------------------------------------------------------
    // number of spins in SK model
    let spins_number = 2000;
    // magnetic field per spin
    let magnetic_field = 1.789;
    // error threshold specifying message passing stopping criterion
    let error = 1e-10f64;
    // decay parameter (a hyper parameter of a sum-product algorithm)
    let decay = 0.5;
    // mean value of coupling constants
    let mu = 0f64;
    // std of coupling constants
    let std = 1f64 / (f64::sqrt(2f64) * 2f64 * spins_number as f64);
    // distribution of coupling constants
    let distr = Normal::new(mu, std).unwrap();
    // this generator is used to sample coupling constants
    let mut rng_couplings = thread_rng();
    // messages initializer
    let mut initializer = random_message_initializer(thread_rng());
    // maximal number of message passing iterations
    let max_iter = 10000;
    // -------------------------------------------------------------------------------------
    let mut fgb =
        new_ising_builder::<SumProduct>(spins_number, (spins_number - 1) * spins_number / 2);
    for i in 0..spins_number {
        for j in (i + 1)..spins_number {
            fgb.add_factor(
                IsingFactor::new(
                    distr.sample(&mut rng_couplings),
                    magnetic_field / ((spins_number - 1) as f64),
                    magnetic_field / ((spins_number - 1) as f64),
                ),
                &[i, j],
                &mut initializer,
            )
            .unwrap();
        }
    }
    let mut fg = fgb.build();
    let info = fg
        .run_message_passing_parallel(max_iter, error, decay)
        .unwrap();
    println!("{}", info);
    // All marginal probabilities of a spin being in up position
    let marginals = fg.eval_marginals();
    // Averaged over all spins probability to be in the up position
    let p_up_approx = marginals.iter().copied().sum::<f64>() / spins_number as f64;
    // Exact up probability
    let p_up_exact = f64::tanh(magnetic_field);
    println!("Approximate average spin: {}", 2f64 * p_up_approx - 1f64,);
    println!("Exact average spin: {}", 2f64 * p_up_exact - 1f64,);
}
