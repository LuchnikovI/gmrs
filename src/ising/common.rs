use crate::core::{Factor, Message, Variable};
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
    fn factor_message_update(
        message: IsingMessage,
        coupling: f64,
        input_spin_magnetic_field: f64,
        output_spin_magnetic_field: f64,
    ) -> IsingMessage;

    fn variable_message_update(src: &[IsingMessage], dst: &mut [IsingMessage]);

    fn marginal(messages: &[IsingMessage]) -> f64;

    fn message_init() -> IsingMessage;
}

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
/// Factor node for Ising model of the form
/// exp ( coupling * s1 * s2 + first_spin_b * s1 + second_spin_b * s2 )
pub struct IsingFactor<T: IsingMessagePassingType> {
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
}

impl<T> Factor for IsingFactor<T>
where
    T: IsingMessagePassingType + Debug + Send,
{
    type Message = IsingMessage;

    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    #[inline(always)]
    fn send_messages(&self, src: &[Self::Message], dst: &mut [Self::Message]) {
        unsafe {
            *dst.get_unchecked_mut(1) = T::factor_message_update(
                *src.get_unchecked(0),
                self.coupling,
                self.first_spin_b,
                self.second_spin_b,
            );
            *dst.get_unchecked_mut(0) = T::factor_message_update(
                *src.get_unchecked(1),
                self.coupling,
                self.second_spin_b,
                self.first_spin_b,
            );
        }
    }
}

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct IsingVariable<T: IsingMessagePassingType>(PhantomData<T>);

impl<T> Variable for IsingVariable<T>
where
    T: IsingMessagePassingType + Debug + Send,
{
    type Message = IsingMessage;
    type Marginal = f64;

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
        T::marginal(messages)
    }
}
