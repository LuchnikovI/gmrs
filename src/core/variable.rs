use crate::core::message::Message;
use std::fmt::Debug;

pub trait Variable: Debug + Send {
    /// Message type
    type Message: Message;
    /// Type representing a marginal distribution
    type Marginal;

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
}
