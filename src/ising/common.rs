use crate::core::{Factor, FactorGraphBuilder, Message, Variable};
use ndarray::{Array1, Array2};
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

pub trait IsingMessagePassingType {
    type Parameters: Sync;
    fn factor_message_update(
        message: IsingMessage,
        prev_message: IsingMessage,
        coupling: f64,
        input_spin_magnetic_field: f64,
        output_spin_magnetic_field: f64,
        parameters: &Self::Parameters,
    ) -> IsingMessage;

    fn variable_message_update(src: &[IsingMessage], dst: &mut [IsingMessage]);

    fn variable_marginal(messages: &[IsingMessage]) -> Array1<f64>;

    fn factor_marginal(factor: &IsingFactor<Self>, messages: &[IsingMessage]) -> Array2<f64>;
}

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
/// Factor node for Ising model of the form
/// exp ( coupling * s1 * s2 + first_spin_b * s1 + second_spin_b * s2 )
pub struct IsingFactor<T: IsingMessagePassingType + ?Sized> {
    marker: PhantomData<T>,

    /// Coupling between neighboring spins
    pub coupling: f64,

    /// Magnetic field acting on the first spin
    pub first_spin_b: f64,

    /// Magnetic field acting on the second spin
    pub second_spin_b: f64,
}

impl<T> IsingFactor<T>
where
    T: IsingMessagePassingType + Debug + Send,
{
    /// Crates a new Ising factor.
    ///
    /// # Arguments
    ///
    /// * `coupling` - A coupling between spins
    /// * `first_spin_b` - A magnetic field acting on the first spin
    /// * `second_spin_b` - A magnetic field acting on the second spin
    ///
    /// # Notes
    ///
    /// Ising factor always has degree (number of adjoint variables) equal to 2
    #[inline]
    pub fn new(coupling: f64, first_spin_b: f64, second_spin_b: f64) -> Self {
        IsingFactor {
            marker: PhantomData,
            coupling,
            first_spin_b,
            second_spin_b,
        }
    }

    /// Returns a coupling amplitude
    #[inline(always)]
    pub fn get_coupling(&self) -> f64 {
        self.coupling
    }

    /// Returns magnetic field amplitude acting on the first spin
    #[inline(always)]
    pub fn get_first_spin_field(&self) -> f64 {
        self.first_spin_b
    }

    /// Returns magnetic field amplitude acting on the second spin
    #[inline(always)]
    pub fn get_second_spin_field(&self) -> f64 {
        self.second_spin_b
    }
}

impl<T> Factor for IsingFactor<T>
where
    T: IsingMessagePassingType + Clone + Debug + Send,
{
    type Message = IsingMessage;
    type Parameters = T::Parameters;
    type Marginal = Array2<f64>;

    #[inline(always)]
    fn from_message(_: &Self::Message) -> Self {
        todo!("Needs to be implemented")
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    #[inline(always)]
    fn send_messages(
        &self,
        src: &[Self::Message],
        dst: &mut [Self::Message],
        parameters: &T::Parameters,
    ) {
        unsafe {
            let prev_message = *dst.get_unchecked(1);
            *dst.get_unchecked_mut(1) = T::factor_message_update(
                *src.get_unchecked(0),
                prev_message,
                self.coupling,
                self.first_spin_b,
                self.second_spin_b,
                parameters,
            );
            let prev_message = *dst.get_unchecked(0);
            *dst.get_unchecked_mut(0) = T::factor_message_update(
                *src.get_unchecked(1),
                prev_message,
                self.coupling,
                self.second_spin_b,
                self.first_spin_b,
                parameters,
            );
        }
    }

    #[inline(always)]
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal {
        T::factor_marginal(self, messages)
    }

    #[inline(always)]
    fn factor(&self) -> Self::Marginal {
        let mut factor = Vec::with_capacity(4);
        let ptr = factor.as_mut_ptr();
        unsafe {
            *ptr = f64::exp(
                self.get_coupling() + self.get_first_spin_field() + self.get_second_spin_field(),
            );
            *ptr.add(1) = f64::exp(
                -self.get_coupling() + self.get_first_spin_field() - self.get_second_spin_field(),
            );
            *ptr.add(2) = f64::exp(
                -self.get_coupling() - self.get_first_spin_field() + self.get_second_spin_field(),
            );
            *ptr.add(3) = f64::exp(
                self.get_coupling() - self.get_first_spin_field() - self.get_second_spin_field(),
            );
            factor.set_len(4);
        }
        Array2::from_shape_vec([2, 2], factor).unwrap()
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
        T::variable_message_update(src, dst);
    }

    #[inline(always)]
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal {
        T::variable_marginal(messages)
    }

    #[inline(always)]
    fn sample(&self, _: &[Self::Message], _: &mut impl Rng) -> Self::Sample {
        todo!("Needs to be implemented")
    }

    #[inline(always)]
    fn sample_to_message(_: &Self::Sample) -> Self::Message {
        todo!("Needs to be implemented")
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
/// * `rng` - A thread-local generator of random numbers
pub fn random_message_initializer(mut rng: impl Rng) -> impl FnMut() -> IsingMessage {
    let distr = Uniform::new(-1f64, 1f64);
    move || IsingMessage(distr.sample(&mut rng))
}
