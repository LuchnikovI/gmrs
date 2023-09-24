pub(super) fn field2prob(field: f64) -> f64 {
    f64::exp(2f64 * field) / (1f64 + f64::exp(2f64 * field))
}
