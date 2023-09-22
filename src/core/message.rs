use std::fmt::Debug;

pub trait Message: Debug + Copy + 'static {
    /// Evaluates a distance between messages
    ///
    /// # Arguments
    ///
    /// * `other` - A second message
    ///
    /// # Notes
    ///
    /// This method is used in message passing in order to
    /// define a stopping criterion: when discrepancy is smaller
    /// than some threshold, message passing stops
    fn discrepancy(&self, other: &Self) -> f64;
}
