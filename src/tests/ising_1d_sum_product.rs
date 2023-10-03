use super::ising_utils::{
    exact_infinite_1d_ising_free_entropy, exact_infinite_1d_ising_up_probability,
};
use crate::ising::schedulers::{get_standard_factor_scheduler, get_standard_variable_scheduler};
use crate::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;

#[test]
fn ising_1d_test() {
    let spins_number = 101;
    let coupling = 1.1f64;
    let magnetic_field = 0.3f64;
    let error = 1e-10f64;
    let mut fgb = new_ising_builder::<SumProduct>(spins_number, spins_number - 1);
    let mut initializer = random_message_initializer(thread_rng());
    let factor_scheduler = get_standard_factor_scheduler(0.5);
    let variable_scheduler = get_standard_variable_scheduler(0.5);
    fgb.add_factor(
        IsingFactor::new(0f64, 0f64, magnetic_field),
        &[spins_number - 2, spins_number - 1],
        &mut initializer,
    )
    .unwrap();
    for i in 0..(spins_number - 1) {
        fgb.add_factor(
            IsingFactor::new(coupling, magnetic_field, 0f64),
            &[i, i + 1],
            &mut initializer,
        )
        .unwrap();
    }
    let mut fg = fgb.build();
    let _ = fg
        .run_message_passing_parallel(1000, 0, error, &factor_scheduler, &variable_scheduler)
        .unwrap();
    let variable_marginals = fg.variable_marginals();
    let (exact_mid_spin_prob_up, exact_bound_spin_prob_up) =
        exact_infinite_1d_ising_up_probability(coupling, magnetic_field, error);
    let calculated_mid_spin_prob_up = variable_marginals[spins_number / 2 + 1][0];
    let calculated_bound_spin_prob_up = variable_marginals[0][0];
    assert!(
        (exact_mid_spin_prob_up - calculated_mid_spin_prob_up).abs() < error * 10f64,
        "Error amplitude: {}",
        (exact_mid_spin_prob_up - calculated_mid_spin_prob_up).abs()
    );
    assert!(
        (exact_bound_spin_prob_up - calculated_bound_spin_prob_up).abs() < error * 10f64,
        "Error amplitude: {}",
        (exact_bound_spin_prob_up - calculated_bound_spin_prob_up).abs()
    );
    let factors = fg.factors();
    let factor_marginals = fg.factor_marginals();
    let mut bethe_free_entropy = 0f64;
    let fm = &factor_marginals[spins_number / 2];
    let f = &factors[spins_number / 2];
    let vm = &variable_marginals[spins_number / 2];
    bethe_free_entropy -= (fm * (fm / f).mapv(f64::ln)).sum();
    bethe_free_entropy += (vm * vm.mapv(f64::ln)).sum();
    assert!(
        (bethe_free_entropy - exact_infinite_1d_ising_free_entropy(coupling, magnetic_field)).abs()
            < error * 10f64
    );
}
