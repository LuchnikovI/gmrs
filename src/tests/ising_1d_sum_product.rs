use super::utils::field2prob;
use crate::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;

fn exact_infinite_ising_1d_magnetization(
    coupling: f64,
    magnetic_field: f64,
    error: f64,
) -> (f64, f64) {
    let f = |x| (1f64 / coupling) * f64::atanh(f64::tanh(coupling) * f64::tanh(coupling * x));
    let mut old_u = f64::MAX;
    let mut new_u = f64::MIN;
    while (old_u - new_u).abs() > error {
        old_u = new_u;
        new_u = f(old_u + magnetic_field / coupling);
    }
    (
        field2prob(2f64 * coupling * new_u + magnetic_field),
        field2prob(coupling * new_u + magnetic_field),
    )
}

fn exact_free_entropy(coupling: f64, magnetic_field: f64) -> f64 {
    f64::ln(
        f64::exp(coupling) * f64::cosh(magnetic_field)
            + f64::sqrt(
                f64::exp(2f64 * coupling) * f64::sinh(magnetic_field).powf(2f64)
                    + f64::exp(-2f64 * coupling),
            ),
    )
}

#[test]
fn ising_1d_test() {
    let spins_number = 101;
    let coupling = 1.1f64;
    let magnetic_field = 0.3f64;
    let error = 1e-10f64;
    let decay = 0f64;
    let mut fgb = new_ising_builder::<SumProduct>(spins_number, spins_number - 1);
    let mut initializer = random_message_initializer(thread_rng());
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
    let _ = fg.run_message_passing_parallel(1000, error, decay).unwrap();
    let variable_marginals = fg.variable_marginals();
    let (exact_mid_spin_prob_up, exact_bound_spin_prob_up) =
        exact_infinite_ising_1d_magnetization(coupling, magnetic_field, error);
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
        (bethe_free_entropy - exact_free_entropy(coupling, magnetic_field)).abs() < error * 10f64
    );
}
