#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    if x > 0f64 {
        1f64 / (1f64 + f64::exp(-x))
    } else {
        f64::exp(x) / (1f64 + f64::exp(x))
    }
}

#[inline(always)]
fn entropy(p: f64) -> f64 {
    -p * f64::ln(p) - (1f64 - p) * f64::ln(1f64 - p)
}

#[inline(always)]
pub(super) fn exact_infinite_1d_ising_up_probability(
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
    let effective_field_mid_spin = 2f64 * coupling * new_u + magnetic_field;
    let effective_field_boundary_spin = coupling * new_u + magnetic_field;
    (
        sigmoid(2f64 * effective_field_mid_spin),
        sigmoid(2f64 * effective_field_boundary_spin),
    )
}

#[inline(always)]
pub(super) fn exact_infinite_1d_ising_free_entropy(coupling: f64, magnetic_field: f64) -> f64 {
    f64::ln(
        f64::exp(coupling) * f64::cosh(magnetic_field)
            + f64::sqrt(
                f64::exp(2f64 * coupling) * f64::sinh(magnetic_field).powf(2f64)
                    + f64::exp(-2f64 * coupling),
            ),
    )
}

#[inline(always)]
pub(super) fn exact_infinite_2d_ising_up_probability(coupling: f64) -> f64 {
    let spin = f64::powf(
        1f64 - f64::powf(f64::sinh(2f64 * coupling), -4f64),
        1f64 / 8f64,
    );
    (spin + 1f64) / 2f64
}

#[inline(always)]
pub(super) fn exact_curie_weiss_up_probability(
    coupling: f64,
    magnetic_field: f64,
    error: f64,
) -> f64 {
    let f = |x| f64::tanh(coupling * x + magnetic_field);
    let mut old_u = f64::MAX;
    let mut new_u = f64::MIN;
    while (old_u - new_u).abs() > error {
        old_u = new_u;
        new_u = f(old_u);
    }
    (new_u + 1f64) / 2f64
}

#[inline(always)]
pub(super) fn exact_curie_weiss_free_entropy(
    coupling: f64,
    magnetic_field: f64,
    error: f64,
) -> f64 {
    let m = 2f64 * exact_curie_weiss_up_probability(coupling, magnetic_field, error) - 1f64;
    0.5f64 * coupling * m * m + magnetic_field * m + entropy((1f64 + m) / 2f64)
}
