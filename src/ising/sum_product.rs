use super::common::{log_sigmoid, log_sum_exponents, logit, IsingMessage, IsingMessagePassingType};
use rand_distr::Uniform;

/// Sum product type of message passing
#[derive(Debug, Clone, Copy)]
pub struct SumProduct;

impl IsingMessagePassingType for SumProduct {
    type Parameters = f64;
    #[inline(always)]
    fn factor_message_update(
        message: IsingMessage,
        prev_message: IsingMessage,
        log_p_ou_iu: f64,
        log_p_ou_id: f64,
        log_p_od_iu: f64,
        log_p_od_id: f64,
        parameters: &Self::Parameters,
    ) -> IsingMessage {
        let nu_up = log_sigmoid(message.0);
        let nu_down = log_sigmoid(-message.0);
        IsingMessage(
            (1f64 - parameters)
                * (log_sum_exponents(log_p_ou_iu + nu_up, log_p_ou_id + nu_down)
                    - log_sum_exponents(log_p_od_iu + nu_up, log_p_od_id + nu_down))
                + parameters * prev_message.0,
        )
    }

    #[inline(always)]
    fn sample(messages: &[IsingMessage], rng: &mut impl rand::Rng) -> i8 {
        let sum_all = messages.iter().map(|x| x.0).sum();
        if logit(rng.sample(Uniform::new(0f64, 1f64))) < sum_all {
            1
        } else {
            -1
        }
    }
}
