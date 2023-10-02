use crate::core::message::Message;
use std::fmt::Debug;

pub trait Factor: Clone + Debug + Send {
    /// Type of a message
    type Message: Message;
    /// Message passing hyper parameters
    type Parameters: Sync;
    /// Factor marginal distribution type
    type Marginal;

    /// Creates a unit degree factor that produces a given message
    ///
    /// # Arguments
    ///
    /// * `message` - A message
    ///
    /// # Note
    ///
    /// This method primarily use case is creating a factor that
    /// fixes a variable value by constantly sending the same
    /// 'strong' message to a variable during message passing
    fn from_message(message: &Self::Message) -> Self;

    /// Returns a degree of a factor (number of adjoint variables)
    fn degree(&self) -> usize;

    /// Sends messages to adjoint variables
    ///
    /// # Arguments
    ///
    /// * `src` - Messages received from adjoint variables previously
    /// * `dst` - Destinations where to send messages
    /// * `parameters` - Hyper parameters of message passing rules
    ///
    /// # Notes
    ///
    /// src[0] corresponds to the message received from the first variable,
    /// dst[0] corresponds to the message receiver of the first variable,
    /// src[1] corresponds to the message received from the second variable,
    /// dst[1] corresponds to the message receiver of the second variable,
    /// etc.
    /// This method defines the logic of how a factor transforms received
    /// messages to messages that it sends to adjoint variables
    fn send_messages(
        &self,
        src: &[Self::Message],
        dst: &mut [Self::Message],
        parameters: &Self::Parameters,
    );

    /// Computes a joint marginal of adjoint factor variables
    ///
    /// # Arguments
    ///
    /// * `messages` - Messages received from adjoint variables previously
    fn marginal(&self, messages: &[Self::Message]) -> Self::Marginal;

    /// Returns a factor as a standalone object, e.g. a tensor psi(x_1, ..., x_n)
    ///
    /// # Notes
    ///
    /// Do not be confused by the returned type,
    /// the most natural data structure representing a standalone factor
    /// is the same used to represent a marginal
    fn factor(&self) -> Self::Marginal;
}
