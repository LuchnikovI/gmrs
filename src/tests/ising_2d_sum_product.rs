use crate::ising::{SumProduct, IsingFactor, new_ising_builder, random_message_initializer};
use rand::{rngs::StdRng, SeedableRng};
fn exact_magnetization(coupling: f64) -> f64 {
    f64::powf(1f64 - f64::powf(f64::sinh(2f64 * coupling), -4f64), 1f64 / 8f64)
}

#[test]
fn ising_2d_test() {
    let size = 20;
    let coupling = -1.7;
    let magnetic_fields = 0.;
    let error = 1e-10f64;
    let decay = 0.5;
    let rng = StdRng::seed_from_u64(42);
    let mut initializer = random_message_initializer(rng);
    let mut fgb = new_ising_builder::<SumProduct>(
        size * size,
        2 * size * size,
    );
    for i in 0..size {
        for j in 0..size {
            fgb.add_factor(
                IsingFactor::new(coupling, magnetic_fields / 4., magnetic_fields / 4.),
                &[j + i * size, (j + 1) % size + i * size],
                &mut initializer,
            ).unwrap();
            fgb.add_factor(
                IsingFactor::new(coupling, magnetic_fields / 4., magnetic_fields / 4.),
                &[j + i * size, j + ((i + 1) % size) * size],
                &mut initializer,
            ).unwrap();
        }
    }
    let mut fg = fgb.build();
    let _ = fg.run_message_passing_parallel(10000, error, decay).unwrap();
    let marginals = fg.eval_marginals();
    let calculated_m = marginals[0];
    let m = exact_magnetization(coupling);
    for i in 0..size {
        for j in 0..size {
            assert!((marginals[size * i + j] + marginals[size * i + (j + 1) % size]).abs() < 1e-5);
        }
    }
    assert!((calculated_m.abs() - f64::atanh(m)).abs() < 1e-4);
}
