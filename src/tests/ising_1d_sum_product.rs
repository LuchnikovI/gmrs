use rand::thread_rng;

use crate::ising::{SumProduct, IsingFactor, new_ising_builder, random_message_initializer};

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
        f64::tanh(2f64 * coupling * new_u + magnetic_field),
        f64::tanh(coupling * new_u + magnetic_field),
    )
}

#[test]
fn ising_1d_test() {
    let spins_number = 101;
    let coupling = 1f64;
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
    let marginals = fg.eval_marginals();
    let (exact_mid_spin, exact_bound_spin) =
        exact_infinite_ising_1d_magnetization(coupling, magnetic_field, error);
    let calculated_mid_spin = f64::tanh(marginals[spins_number / 2 + 1]);
    let calculated_bound_spin = f64::tanh(marginals[0]);
    assert!((exact_mid_spin - calculated_mid_spin).abs() < error * 10f64, "Error amplitude: {}", (exact_mid_spin - calculated_mid_spin).abs());
    assert!((exact_bound_spin - calculated_bound_spin).abs() < error * 10f64, "Error amplitude: {}", (exact_bound_spin - calculated_bound_spin).abs());
}
