use super::{
    common::{IsingMessage, IsingMessagePassingType},
    IsingFactor,
};
use ndarray::{Array1, Array2};

/// Sum product type of message passing
#[derive(Debug, Clone, Copy)]
pub struct SumProduct;

impl IsingMessagePassingType for SumProduct {
    type Parameters = f64;
    #[inline(always)]
    fn factor_message_update(
        message: super::common::IsingMessage,
        prev_message: super::common::IsingMessage,
        coupling: f64,
        input_spin_magnetic_field: f64,
        output_spin_magnetic_field: f64,
        parameters: &f64,
    ) -> super::common::IsingMessage {
        IsingMessage(
            (1f64 - parameters)
                * (output_spin_magnetic_field
                    + 0.5f64
                        * f64::ln(
                            f64::cosh(coupling + message.0 + input_spin_magnetic_field)
                                / f64::cosh(-coupling + message.0 + input_spin_magnetic_field),
                        ))
                + parameters * prev_message.0,
        )
    }

    #[inline(always)]
    fn variable_message_update(src: &[IsingMessage], dst: &mut [IsingMessage]) {
        let sum_all: f64 = src.iter().map(|x| x.0).sum();
        for index in 0..src.len() {
            unsafe {
                dst.get_unchecked_mut(index).0 = sum_all - src.get_unchecked(index).0;
            }
        }
    }

    #[inline(always)]
    fn variable_marginal(messages: &[IsingMessage]) -> Array1<f64> {
        let sum_all: f64 = messages.iter().map(|x| x.0).sum();
        let p_down = 1f64 / (f64::exp(2f64 * sum_all) + 1f64);
        Array1::from_vec(vec![1f64 - p_down, p_down])
    }

    #[inline(always)]
    fn factor_marginal(factor: &IsingFactor<SumProduct>, messages: &[IsingMessage]) -> Array2<f64> {
        let mut marginal = Vec::with_capacity(4);
        let ptr = marginal.as_mut_ptr();
        unsafe {
            let m1 = messages.get_unchecked(0).0;
            let m2 = messages.get_unchecked(1).0;
            let nu_up_1 = f64::exp(2f64 * m1) / (f64::exp(2f64 * m1) + 1f64);
            let nu_up_2 = f64::exp(2f64 * m2) / (f64::exp(2f64 * m2) + 1f64);
            let nu_down_1 = 1f64 - nu_up_1;
            let nu_down_2 = 1f64 - nu_up_2;
            *ptr = nu_up_1
                * nu_up_2
                * f64::exp(factor.coupling + factor.first_spin_b + factor.second_spin_b);
            *ptr.add(1) = nu_up_1
                * nu_down_2
                * f64::exp(-factor.coupling + factor.first_spin_b - factor.second_spin_b);
            *ptr.add(2) = nu_down_1
                * nu_up_2
                * f64::exp(-factor.coupling - factor.first_spin_b + factor.second_spin_b);
            *ptr.add(3) = nu_down_1
                * nu_down_2
                * f64::exp(factor.coupling - factor.first_spin_b - factor.second_spin_b);
            marginal.set_len(4);
        }
        let mut marginal = Array2::from_shape_vec([2, 2], marginal).unwrap();
        marginal /= marginal.sum();
        marginal
    }
}
