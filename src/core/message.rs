use std::fmt::Debug;

/// A trait providing message's methods
pub trait Message: Debug + Clone + 'static {
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

    /// Copy message to dst
    ///
    /// # Arguments
    ///
    /// * `dst` - A destination where to copy a message
    ///
    /// # Notes
    ///
    /// It might be useful to reimplement this method in order to avoid
    /// reallocation of memory
    #[inline(always)]
    fn memcpy(&self, dst: &mut Self) {
        *dst = self.clone();
    }
}
