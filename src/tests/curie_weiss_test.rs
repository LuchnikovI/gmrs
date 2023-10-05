use super::ising_utils::{exact_curie_weiss_free_entropy, exact_curie_weiss_up_probability};
use crate::ising::schedulers::{get_standard_factor_scheduler, get_standard_variable_scheduler};
use crate::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;

#[test]
fn curie_weiss_test() {
    let spins_number = 100;
    let coupling = 1.1234;
    let magnetic_field = 0.7654;
    let error = 1e-10f64;
    let factor_scheduler = get_standard_factor_scheduler(0.5);
    let variable_scheduler = get_standard_variable_scheduler(0.5);
    let mut initializer = random_message_initializer(thread_rng(), -0.5, 0.5);
    let mut fgb =
        new_ising_builder::<SumProduct>(spins_number, (spins_number - 1) * spins_number / 2);
    for i in 0..spins_number {
        for j in (i + 1)..spins_number {
            fgb.add_factor(
                IsingFactor::new(
                    coupling / (spins_number as f64),
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
    let _ = fg
        .run_message_passing_parallel(10000, 0, error, &factor_scheduler, &variable_scheduler)
        .unwrap();
    let variable_marginals = fg.variable_marginals();
    let exact_up_prob = exact_curie_weiss_up_probability(coupling, magnetic_field, error);
    assert!((variable_marginals[spins_number / 2][0] - exact_up_prob).abs() < 1e-2);
    let factors = fg.factors();
    let factor_marginals = fg.factor_marginals();
    let mut bethe_free_entropy = 0f64;
    for (fm, f) in factor_marginals.iter().zip(&factors) {
        bethe_free_entropy -= (fm * (fm / f).mapv(f64::ln)).sum();
    }
    for vm in &variable_marginals {
        bethe_free_entropy += ((spins_number - 2) as f64) * (vm * vm.mapv(f64::ln)).sum();
    }
    assert!(
        (bethe_free_entropy / spins_number as f64
            - exact_curie_weiss_free_entropy(coupling, magnetic_field, error))
        .abs()
            < 1e-2
    );
}
