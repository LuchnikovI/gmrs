use crate::core::{Factor, FactorGraphBuilder, Message, Variable};
use ndarray::{Array1, ArrayD, IxDyn};
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use std::{fmt::Debug, marker::PhantomData};

// ------------------------------------------------------------------------------------------

/// Ising message
#[derive(Debug, Clone, Copy)]
pub struct IsingMessage(pub f64);

impl Message for IsingMessage {
    #[inline(always)]
    fn discrepancy(&self, other: &Self) -> f64 {
        (self.0 - other.0).abs()
    }
}

// ------------------------------------------------------------------------------------------

#[inline(always)]
pub(super) fn sigmoid(x: f64) -> f64 {
    if x > 0f64 {
        1f64 / (1f64 + f64::exp(-x))
    } else {
        f64::exp(x) / (1f64 + f64::exp(x))
    }
}

#[inline(always)]
pub(super) fn log_sigmoid(x: f64) -> f64 {
    if x > 0f64 {
        -f64::ln(1f64 + f64::exp(-x))
    } else {
        x - f64::ln(1f64 + f64::exp(x))
    }
}

#[inline(always)]
pub(super) fn log_sum_exponents(x: f64, y: f64) -> f64 {
    if x > y {
        x + f64::ln(1f64 + f64::exp(y - x))
    } else {
        y + f64::ln(1f64 + f64::exp(x - y))
    }
}

// ------------------------------------------------------------------------------------------

pub trait IsingMessagePassingType {
    type Parameters: Sync;
    fn factor_message_update(
        message: IsingMessage,
        prev_message: IsingMessage,
        log_p_ou_iu: f64,
        log_p_ou_id: f64,
        log_p_od_iu: f64,
        log_p_od_id: f64,
        parameters: &Self::Parameters,
    ) -> IsingMessage;

    fn sample(messages: &[IsingMessage], rng: &mut impl Rng) -> i8;
}

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
/// Coupling Ising factor of the form
/// exp ( coupling * s1 * s2 + first_spin_b * s1 + second_spin_b * s2 )
/// or a unit degree factor containing a value determining an output message
pub enum IsingFactor<T: IsingMessagePassingType + ?Sized> {
    Coupling {
        marker: PhantomData<T>,
        log_puu: f64,
        log_pud: f64,
        log_pdu: f64,
        log_pdd: f64,
    },
    UnitFactor(f64),
}

impl<T> IsingFactor<T>
where
    T: IsingMessagePassingType + Debug + Send,
{
    /// Crates a new Ising coupling factor.
    ///
    /// # Arguments
    ///
    /// * `coupling` - A coupling between spins
    /// * `first_spin_b` - A magnetic field acting on the first spin
    /// * `second_spin_b` - A magnetic field acting on the second spin
    #[inline]
    pub fn new(coupling: f64, first_spin_b: f64, second_spin_b: f64) -> Self {
        IsingFactor::Coupling {
            marker: PhantomData,
            log_puu: coupling + first_spin_b + second_spin_b,
            log_pud: -coupling + first_spin_b - second_spin_b,
            log_pdu: -coupling - first_spin_b + second_spin_b,
            log_pdd: coupling - first_spin_b - second_spin_b,
        }
    }
}

impl<T> Factor for IsingFactor<T>
where
    T: IsingMessagePassingType + Clone + Debug + Send,
{
    type Message = IsingMessage;
    type Parameters = T::Parameters;
    type Marginal = ArrayD<f64>;

    #[inline(always)]
    fn from_message(message: &Self::Message) -> Self {
        IsingFactor::UnitFactor(message.0)
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        match self {
            IsingFactor::Coupling { .. } => 2,
            IsingFactor::UnitFactor(_) => 1,
        }
    }

    #[inline(always)]
    fn send_messages(
        &self,
        src: &[Self::Message],
        dst: &mut [Self::Message],
        parameters: &T::Parameters,
    ) {
        match self {
            IsingFactor::Coupling {
                marker: _,
                log_puu,
                log_pud,
                log_pdu,
                log_pdd,
            } => unsafe {
                let prev_message = *dst.get_unchecked(1);
                *dst.get_unchecked_mut(1) = T::factor_message_update(
                    *src.get_unchecked(0),
                    prev_message,
                    *log_puu,
                    *log_pdu,
                    *log_pud,
                    *log_pdd,
                    parameters,
                );
                let prev_message = *dst.get_unchecked(0);
                *dst.get_unchecked_mut(0) = T::factor_message_update(
                    *src.get_unchecked(1),
                    prev_message,
                    *log_puu,
                    *log_pud,
                    *log_pdu,
                    *log_pdd,
                    parameters,
                );
            },
            IsingFactor::UnitFactor(m) => unsafe {
                *dst.get_unchecked_mut(0) = IsingMessage(*m);
            },
        }
    }

    #[inline(always)]
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal {
        match self {
            IsingFactor::Coupling {
                marker: _,
                log_puu,
                log_pud,
                log_pdu,
                log_pdd,
            } => {
                let nu_up_1 = log_sigmoid(unsafe { messages.get_unchecked(0).0 });
                let nu_up_2 = log_sigmoid(unsafe { messages.get_unchecked(1).0 });
                let nu_down_1 = log_sigmoid(unsafe { -messages.get_unchecked(0).0 });
                let nu_down_2 = log_sigmoid(unsafe { -messages.get_unchecked(1).0 });
                let marginal = vec![
                    (log_puu + nu_up_1 + nu_up_2).exp(),
                    (log_pud + nu_up_1 + nu_down_2).exp(),
                    (log_pdu + nu_down_1 + nu_up_2).exp(),
                    (log_pdd + nu_down_1 + nu_down_2).exp(),
                ];
                let mut marginal = ArrayD::from_shape_vec(IxDyn(&[2, 2]), marginal).unwrap();
                marginal /= marginal.sum();
                marginal
            }
            IsingFactor::UnitFactor(m) => {
                let log_pu = log_sigmoid(*m);
                let log_pd = log_sigmoid(-*m);
                let nu_up = log_sigmoid(unsafe { messages.get_unchecked(0).0 });
                let nu_down = log_sigmoid(unsafe { messages.get_unchecked(1).0 });
                let marginal = vec![(log_pu + nu_up).exp(), (log_pd + nu_down).exp()];
                let mut marginal = ArrayD::from_shape_vec(IxDyn(&[2]), marginal).unwrap();
                marginal /= marginal.sum();
                marginal
            }
        }
    }

    #[inline(always)]
    fn factor(&self) -> Self::Marginal {
        match self {
            IsingFactor::Coupling {
                marker: _,
                log_puu,
                log_pud,
                log_pdu,
                log_pdd,
            } => {
                let factor = vec![log_puu.exp(), log_pud.exp(), log_pdu.exp(), log_pdd.exp()];
                ArrayD::from_shape_vec(IxDyn(&[2, 2]), factor).unwrap()
            }
            IsingFactor::UnitFactor(m) => {
                let factor = vec![log_sigmoid(*m), log_sigmoid(-*m)];
                let mut factor = ArrayD::from_shape_vec(IxDyn(&[2]), factor).unwrap();
                factor /= factor.sum();
                factor
            }
        }
    }
}

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct IsingVariable<T: IsingMessagePassingType>(PhantomData<T>);

impl<T> Variable for IsingVariable<T>
where
    T: IsingMessagePassingType + Clone + Debug + Send,
{
    type Message = IsingMessage;
    type Marginal = Array1<f64>;
    type Sample = i8;

    #[inline(always)]
    fn new() -> Self {
        IsingVariable(PhantomData)
    }

    #[inline(always)]
    fn send_messages(&self, src: &[Self::Message], dst: &mut [Self::Message]) {
        let sum_all: f64 = src.iter().map(|x| x.0).sum();
        for (d, s) in dst.iter_mut().zip(src) {
            d.0 = sum_all - s.0;
        }
    }

    #[inline(always)]
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal {
        let all_sum = messages.iter().map(|x| x.0).sum();
        let p_up = sigmoid(all_sum);
        Array1::from_vec(vec![p_up, 1f64 - p_up])
    }

    #[inline(always)]
    fn sample(&self, messages: &[Self::Message], rng: &mut impl Rng) -> Self::Sample {
        T::sample(messages, rng)
    }

    #[inline(always)]
    fn sample_to_message(sample: &Self::Sample) -> Self::Message {
        match sample {
            1 => IsingMessage(1e30f64),
            -1 => IsingMessage(-1e30f64),
            other => panic!("Unsupported sample value {other}, must be ether 1 or -1. It is a bug, please open an issue"),
        }
    }
}

// ------------------------------------------------------------------------------------------

/// Crates a new Ising factor graph builder.
///
/// # Arguments
///
/// * `variables_number` - A number of variables
/// * `factors_capacity` - A number of factors we need to preallocate memory for
pub fn new_ising_builder<T>(
    variables_number: usize,
    factors_capacity: usize,
) -> FactorGraphBuilder<IsingFactor<T>, IsingVariable<T>>
where
    T: IsingMessagePassingType + Clone + Debug + Send,
{
    FactorGraphBuilder::new_with_variables(variables_number, factors_capacity)
}

/// Crates a new random ising message initializer.
/// A created generator samples messages at random from
/// a uniform distribution over [-1, 1].
///
/// # Arguments
///
/// * `rng` - A generator of random numbers
pub fn random_message_initializer(mut rng: impl Rng) -> impl FnMut() -> IsingMessage {
    let distr = Uniform::new(-1f64, 1f64);
    move || IsingMessage(distr.sample(&mut rng))
}
