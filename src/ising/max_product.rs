use super::{
    common::{log_sigmoid, IsingMessage, IsingMessagePassingType},
    IsingFactorHyperParameters,
};

/// Sum product type of message passing
#[derive(Debug, Clone, Copy)]
pub struct MaxProduct;

impl IsingMessagePassingType for MaxProduct {
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
        IsingMessage(
            (1f64 - gamma)
                * ((log_p_ou_iu + nu_up).max(log_p_ou_id + nu_down)
                    - (log_p_od_iu + nu_up).max(log_p_od_id + nu_down))
                + gamma * prev_message.0,
        )
    }

    #[inline(always)]
    fn sample(messages: &[IsingMessage], _: &mut impl rand::Rng) -> i8 {
        let sum_all: f64 = messages.iter().map(|x| x.0).sum();
        if sum_all > 0f64 {
            1
        } else {
            -1
        }
    }
}
