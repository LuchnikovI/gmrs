use rand::Rng;

use crate::core::message::Message;
use std::fmt::Debug;

pub trait Variable: Clone + Debug + Send {
    /// Type of a message
    type Message: Message;
    /// Message passing hyper parameters
    type Parameters: Sync;
    /// Type representing a marginal distribution
    type Marginal;
    /// Type representing a variable sample
    type Sample: Copy;

    /// Create a new variable instance
    fn new() -> Self;

    /// Sends messages to adjoint factors
    ///
    /// # Arguments
    ///
    ///
    /// * `src` - Messages received from adjoint factors previously
    /// * `dst` - Destinations where to send messages
    /// * `parameters` - Hyper parameters of message passing rules
    ///
    /// # Notes
    ///
    /// src[0] corresponds to the message received from the first factor,
    /// dst[0] corresponds to the message receiver of the first factor,
    /// src[1] corresponds to the message received from the second factor,
    /// dst[1] corresponds to the message receiver of the second factor,
    /// etc.
    /// This method defines the logic of how a variable transforms received messages
    /// to messages that it sends to adjoint variables
    fn send_messages(
        &self,
        src: &[Self::Message],
        dst: &mut [Self::Message],
        parameters: &Self::Parameters,
    );

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
    /// This method typically is used together with `from_message`
    /// method of the Factor trait in order to create a factor that
    /// fixes a variable value by the sampled one. It is done in
    /// two steps: (1) one creates a message that fixes a variable by
    /// calling the given method, (2) one creates the factor that produces
    /// a created message by calling a `from_message` method
    fn sample_to_message(sample: &Self::Sample) -> Self::Message;
}
