use rand::Rng;

use crate::core::message::Message;
use std::fmt::Debug;

pub trait Variable: Clone + Debug + Send {
    /// Message type
    type Message: Message;
    /// Type representing a marginal distribution
    type Marginal;
    /// Type representing a sample
    type Sample;

    /// Create a new variable instance
    fn new() -> Self;

    /// Sends messages to adjoint factors
    ///
    /// # Arguments
    ///
    /// * `src` - Messages received from adjoint factors previously
    /// * `dst` - Destinations where to send messages
    ///
    /// # Notes
    ///
    /// src[0] corresponds to the message received from the first factor,
    /// dst[0] corresponds to the message receiver of the first factor,
    /// src[1] corresponds to the message received from the second factor,
    /// dst[1] corresponds to the message receiver of the second factor,
    /// etc
    fn send_messages(&self, src: &[Self::Message], dst: &mut [Self::Message]);

    /// Computes a marginal of a variable
    ///
    /// # Arguments
    ///
    /// * `messages` - Messages received from adjoint factors previously
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal;

    /// Computes a sample from a variable
    ///
    /// # Arguments
    ///
    /// * `messages` - Messages received from adjoint factors previously
    /// * `rng` - A random numbers generator
    fn sample(&self, messages: &[Self::Message], rng: &mut impl Rng) -> Self::Sample;

    /// Returns a message that sets a variable to the state corresponding to
    /// a given sample
    ///
    /// # Arguments
    ///
    /// * `sample` - A sample to convert to a message
    ///
    /// # Notes
    ///
    /// This method is useful to fix a value of a variable after sampling.
    /// One can make a unit degree factor node sending this message that
    /// sets a variable to the sampled value
    fn sample_to_message(sample: &Self::Sample) -> Self::Message;
}
