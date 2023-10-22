use crate::core::{Factor, FactorGraphBuilder, Message, Variable};
use ndarray::{Array1, ArrayD, IxDyn};
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use std::{fmt::Debug, marker::PhantomData};

use super::IsingFactorHyperParameters;

// ------------------------------------------------------------------------------------------

/// An Ising factor graph's message type
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

/// A trait containing message passing type specific methods
pub trait IsingMessagePassingType {
    fn factor_message_update(
        message: IsingMessage,
        prev_message: IsingMessage,
        log_p_ou_iu: f64,
        log_p_ou_id: f64,
        log_p_od_iu: f64,
        log_p_od_id: f64,
        parameters: &IsingFactorHyperParameters,
    ) -> IsingMessage;

    fn sample(messages: &[IsingMessage], rng: &mut impl Rng) -> i8;
}

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
/// An Ising factor type. It is either a coupling factor of the form
/// `exp ( coupling * s1 * s2 + first_spin_b * s1 + second_spin_b * s2 )`
/// containing logarithms of factor's elements
/// or a unit degree factor of the form `exp ( s * b )` containing a
/// corresponding message `2 * b`
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
    /// * `coupling` - A coupling magnitude
    /// * `first_spin_b` - A magnetic field acting on the first spin
    /// * `second_spin_b` - A magnetic field acting on the second spin
    ///
    /// # Notes
    ///
    /// A resulting factor has form `exp ( coupling * s1 * s2 + first_spin_b * s1 + second_spin_b * s2 )`
    ///
    /// # Example
    ///
    /// ```
    /// use gmrs::ising::{IsingFactor, MaxProduct};
    ///
    /// let ising_factor = IsingFactor::<MaxProduct>::new(0.5f64, 0.5f64, -0.5f64);
    /// ```
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
    type Marginal = ArrayD<f64>;
    type Parameters = IsingFactorHyperParameters;

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
        parameters: &IsingFactorHyperParameters,
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

/// An Ising variable type
#[derive(Debug, Clone, Copy)]
pub struct IsingVariable<T: IsingMessagePassingType>(PhantomData<T>);

impl<T: IsingMessagePassingType> IsingVariable<T> {
    /// Creates a new variable.
    ///
    /// # Example
    /// ```
    /// use gmrs::ising::{IsingVariable, SumProduct};
    ///
    /// let var = IsingVariable::<SumProduct>::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        IsingVariable(PhantomData)
    }
}

impl<T: IsingMessagePassingType> Default for IsingVariable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Variable for IsingVariable<T>
where
    T: IsingMessagePassingType + Clone + Debug + Send,
{
    type Message = IsingMessage;
    type Marginal = Array1<f64>;
    type Parameters = f64;
    type Sample = i8;

    #[inline(always)]
    fn send_messages(&self, src: &[Self::Message], dst: &mut [Self::Message], parameters: &f64) {
        let sum_all: f64 = src.iter().map(|x| x.0).sum();
        for (d, s) in dst.iter_mut().zip(src) {
            let prev_message = d.0;
            d.0 = (1f64 - parameters) * (sum_all - s.0) + parameters * prev_message;
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
/// * `factors_capacity` - A number of factors used to preallocate memory
///
/// # Example
/// ```
/// use rand::thread_rng;
/// use gmrs::ising::new_ising_builder;
/// use gmrs::ising::SumProduct;
///
/// let fgb = new_ising_builder::<SumProduct>(10, 5);
/// ```
pub fn new_ising_builder<T>(
    variables_number: usize,
    factors_capacity: usize,
) -> FactorGraphBuilder<IsingFactor<T>, IsingVariable<T>>
where
    T: IsingMessagePassingType + Clone + Debug + Send,
{
    let mut fgb = FactorGraphBuilder::new_with_capacity(variables_number, factors_capacity);
    fgb.fill(IsingVariable::new());
    fgb
}

/// Crates a new random Ising message initializer.
/// A created initializer samples messages at random from
/// a uniform distribution over the segment [lower, upper].
///
/// # Arguments
///
/// * `rng` - A generator of random numbers
/// * `lower` - A lower bound
/// * `upper` - An upper bound
///
/// # Example
///
/// ```
/// use rand::thread_rng;
/// use gmrs::ising::random_message_initializer;
///
/// // Messages initializer
/// let rng = thread_rng();
/// let initializer = random_message_initializer(rng, -0.5, 0.5);
/// ```
pub fn random_message_initializer(
    mut rng: impl Rng,
    lower: f64,
    upper: f64,
) -> impl FnMut() -> IsingMessage {
    let distr = Uniform::new(lower, upper);
    move || IsingMessage(distr.sample(&mut rng))
}
