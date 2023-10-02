use super::common::{
    log_sigmoid, log_sum_exponents, sigmoid, IsingMessage, IsingMessagePassingType,
};
use crate::ising::IsingFactorHyperParameters;
use rand_distr::Uniform;

/// Sum product type of message passing
#[derive(Debug, Clone, Copy)]
pub struct SumProduct;

impl IsingMessagePassingType for SumProduct {
    #[inline(always)]
    fn factor_message_update(
        message: IsingMessage,
        prev_message: IsingMessage,
        log_p_ou_iu: f64,
        log_p_ou_id: f64,
        log_p_od_iu: f64,
        log_p_od_id: f64,
        parameters: &IsingFactorHyperParameters,
    ) -> IsingMessage {
        let nu_up = log_sigmoid(message.0);
        let nu_down = log_sigmoid(-message.0);
        let gamma = parameters.gamma;
        let beta = parameters.beta;
        IsingMessage(
            (1f64 - gamma)
                * (log_sum_exponents(beta * log_p_ou_iu + nu_up, beta * log_p_ou_id + nu_down)
                    - log_sum_exponents(beta * log_p_od_iu + nu_up, beta * log_p_od_id + nu_down))
                + gamma * prev_message.0,
        )
    }

    #[inline(always)]
    fn sample(messages: &[IsingMessage], rng: &mut impl rand::Rng) -> i8 {
        let sum_all = messages.iter().map(|x| x.0).sum();
        if rng.sample(Uniform::new(0f64, 1f64)) < sigmoid(sum_all) {
            1
        } else {
            -1
        }
    }
}
