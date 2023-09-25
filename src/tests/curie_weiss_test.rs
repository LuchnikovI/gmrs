use super::utils::field2prob;
use crate::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;

fn exact_curie_weiss_up_prob(coupling: f64, magnetic_field: f64, error: f64) -> f64 {
    let f = |x| f64::tanh(coupling * x + magnetic_field);
    let mut old_u = f64::MAX;
    let mut new_u = f64::MIN;
    while (old_u - new_u).abs() > error {
        old_u = new_u;
        new_u = f(old_u);
    }
    field2prob(f64::atanh(new_u))
}

fn entropy(p: f64) -> f64 {
    -p * f64::ln(p) - (1f64 - p) * f64::ln(1f64 - p)
}

fn exact_free_entropy(coupling: f64, magnetic_field: f64, error: f64) -> f64 {
    let m = 2f64 * exact_curie_weiss_up_prob(coupling, magnetic_field, error) - 1f64;
    0.5f64 * coupling * m * m + magnetic_field * m + entropy((1f64 + m) / 2f64)
}

#[test]
fn curie_weiss_test() {
    let spins_number = 100;
    let coupling = 1.1234;
    let magnetic_field = 0.7654;
    let error = 1e-10f64;
    let decay = 0.5;
    let mut initializer = random_message_initializer(thread_rng());
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
        .run_message_passing_parallel(10000, error, decay)
        .unwrap();
    let variable_marginals = fg.variable_marginals();
    let exact_up_prob = exact_curie_weiss_up_prob(coupling, magnetic_field, error);
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
            - exact_free_entropy(coupling, magnetic_field, error))
        .abs()
            < 1e-2
    );
}
