use super::common::{IsingMessage, IsingMessagePassingType};

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
    fn marginal(messages: &[IsingMessage]) -> f64 {
        let sum_all: f64 = messages.iter().map(|x| x.0).sum();
        sum_all
    }
}
