use super::utils::field2prob;
use crate::ising::schedulers::{get_standard_factor_scheduler, get_standard_variable_scheduler};
use crate::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::{rngs::StdRng, SeedableRng};

fn exact_up_probability(coupling: f64) -> f64 {
    let spin = f64::powf(
        1f64 - f64::powf(f64::sinh(2f64 * coupling), -4f64),
        1f64 / 8f64,
    );
    field2prob(f64::atanh(spin))
}

#[test]
fn ising_2d_test() {
    let size = 20;
    let coupling = -0.7123;
    let magnetic_fields = 0.;
    let error = 1e-10f64;
    let factor_scheduler = get_standard_factor_scheduler(0.5);
    let variable_scheduler = get_standard_variable_scheduler(0.5);
    let rng = StdRng::seed_from_u64(42);
    let mut initializer = random_message_initializer(rng);
    let mut fgb = new_ising_builder::<SumProduct>(size * size, 2 * size * size);
    for i in 0..size {
        for j in 0..size {
            fgb.add_factor(
                IsingFactor::new(coupling, magnetic_fields / 4., magnetic_fields / 4.),
                &[j + i * size, (j + 1) % size + i * size],
                &mut initializer,
            )
            .unwrap();
            fgb.add_factor(
                IsingFactor::new(coupling, magnetic_fields / 4., magnetic_fields / 4.),
                &[j + i * size, j + ((i + 1) % size) * size],
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
    let calculated_up_prob = if variable_marginals[0][0] > variable_marginals[0][1] {
        variable_marginals[0][0]
    } else {
        variable_marginals[0][1]
    };
    let exact_up_prob = exact_up_probability(coupling);
    for i in 0..size {
        for j in 0..size {
            assert!(
                (variable_marginals[size * i + j][0]
                    + variable_marginals[size * i + (j + 1) % size][0]
                    - 1f64)
                    .abs()
                    < 1e-5
            );
        }
    }
    assert!(
        (calculated_up_prob - exact_up_prob).abs() < 1e-4,
        "{calculated_up_prob}, {exact_up_prob}"
    );
}
