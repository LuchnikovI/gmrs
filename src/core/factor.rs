use crate::core::message::Message;
use std::fmt::Debug;

pub trait Factor: Debug + Send {
    type Message: Message;

    /// Returns a degree of a factor (number of adjoint variables)
    fn degree(&self) -> usize;

    /// Sends messages to adjoint variables
    ///
    /// # Arguments
    ///
    /// * `src` - Messages received from adjoint variables previously
    /// * `dst` - Destinations where to send messages
    ///
    /// # Notes
    ///
    /// src[0] corresponds to the message received from the first variable,
    /// dst[0] corresponds to the message receiver of the first variable,
    /// src[1] corresponds to the message received from the second variable,
    /// dst[1] corresponds to the message receiver of the second variable,
    /// etc
    fn send_messages(&self, src: &[Self::Message], dst: &mut [Self::Message]);
}
