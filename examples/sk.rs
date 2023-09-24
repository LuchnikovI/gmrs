use gmrs::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

fn main() {
    // parameters --------------------------------------------------------------------------
    // number of spins in SK model
    let spins_number = 2000;
    // magnetic field per spin
    let magnetic_field = 0.89;
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
    // Marginal distributions for each spin
    let marginals = fg.eval_marginals();
    // Averaged single spin distribution
    let mut approx_distr = [0f64; 2];
    marginals.iter().for_each(|x| {
        approx_distr[0] += x[0];
        approx_distr[1] += x[1];
    });
    approx_distr[0] /= spins_number as f64;
    approx_distr[1] /= spins_number as f64;
    // Exact single spin distribution
    let exact_distr = [
        f64::exp(2f64 * magnetic_field) / (1f64 + f64::exp(2f64 * magnetic_field)),
        1f64 / (1f64 + f64::exp(2f64 * magnetic_field)),
    ];
    println!("Approximate single spin distribution: {:?}", approx_distr);
    println!("Exact single spin distribution: {:?}", exact_distr);
}
