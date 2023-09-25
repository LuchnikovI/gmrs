use crate::core::message::Message;
use std::fmt::Debug;

pub trait Factor: Debug + Send {
    /// Type of a message
    type Message: Message;
    /// Message passing hyper parameters
    type Parameters: Sync;
    /// Type representing a marginal distribution
    type Marginal;

    /// Returns a degree of a factor (number of adjoint variables)
    fn degree(&self) -> usize;

    /// Sends messages to adjoint variables
    ///
    /// # Arguments
    ///
    /// * `src` - Messages received from adjoint variables previously
    /// * `dst` - Destinations where to send messages
    /// * `parameters` - Hyper parameters of message passing
    ///
    /// # Notes
    ///
    /// src[0] corresponds to the message received from the first variable,
    /// dst[0] corresponds to the message receiver of the first variable,
    /// src[1] corresponds to the message received from the second variable,
    /// dst[1] corresponds to the message receiver of the second variable,
    /// etc
    fn send_messages(
        &self,
        src: &[Self::Message],
        dst: &mut [Self::Message],
        parameters: &Self::Parameters,
    );

    /// Computes a marginal of a factor
    ///
    /// # Arguments
    ///
    /// * `messages` - Messages received from adjoint variables previously
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal;

    /// Returns a factor as a standalone object
    ///
    /// # Notes
    ///
    /// The most natural data structure representing a standalone factor
    /// is that used to represent a marginal
    fn factor(&self) -> Self::Marginal;
}
