use crate::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;
use super::utils::field2prob;

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

#[test]
fn curie_weiss_test() {
    let size = 100;
    let coupling = 1.234;
    let magnetic_field = 0.987;
    let error = 1e-10f64;
    let decay = 0.5;
    let mut initializer = random_message_initializer(thread_rng());
    let mut fgb = new_ising_builder::<SumProduct>(size, (size - 1) * size / 2);
    for i in 0..size {
        for j in (i + 1)..size {
            fgb.add_factor(
                IsingFactor::new(
                    coupling / (size as f64),
                    magnetic_field / ((size - 1) as f64),
                    magnetic_field / ((size - 1) as f64),
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
    let marginals = fg.eval_marginals();
    let exact_up_prob = exact_curie_weiss_up_prob(coupling, magnetic_field, error);
    assert!((marginals[size / 2][0] - exact_up_prob).abs() < 1e-3);
}
