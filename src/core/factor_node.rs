use crate::{
    core::factor::Factor, core::message::Message, core::variable::Variable,
    core::variable_node::VariableNode,
};

#[derive(Debug)]
pub struct FactorNode<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    factor: F,
    pub(crate) var_node_indices: Vec<usize>,
    pub(crate) var_node_receiver_indices: Vec<usize>,
    pub(crate) messages: Vec<V::Message>,
    pub(crate) senders: Vec<*mut V::Message>,
    pub(crate) receivers: Vec<F::Message>,
}

unsafe impl<F, V> Send for FactorNode<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
}

impl<F, V> FactorNode<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    #[inline(always)]
    pub(super) fn new_disconnected(factor: F) -> Self {
        FactorNode {
            factor,
            var_node_indices: Vec::new(),
            var_node_receiver_indices: Vec::new(),
            messages: Vec::new(),
            senders: Vec::new(),
            receivers: Vec::new(),
        }
    }

    #[inline(always)]
    pub(super) fn init_senders(&mut self, variables: &mut [VariableNode<V, F>]) {
        let indices_iter = self
            .var_node_indices
            .iter()
            .zip(&self.var_node_receiver_indices);
        let senders: Vec<_> = indices_iter
            .map(|(var_index, var_receiver_index)| unsafe {
                let var = variables.get_unchecked_mut(*var_index);
                let mut_ptr: *mut _ = var.receivers.get_unchecked_mut(*var_receiver_index);
                mut_ptr
            })
            .collect();
        self.senders = senders;
    }

    #[inline(always)]
    pub(super) fn eval_messages(&mut self) {
        self.factor
            .send_messages(&self.receivers, &mut self.messages)
    }

    #[inline(always)]
    pub(super) fn eval_discrepancy(&self) -> f64 {
        let mut max_discrepancy = 0f64;
        for (new_msg, old_msg_ptr) in self.messages.iter().zip(&self.senders) {
            let discrepancy = new_msg.discrepancy(unsafe { &**old_msg_ptr });
            if max_discrepancy < discrepancy {
                max_discrepancy = discrepancy;
            }
        }
        max_discrepancy
    }

    #[inline(always)]
    pub(super) fn send_messages(&mut self) {
        for (msg, dst_ptr) in self.messages.iter().zip(&mut self.senders) {
            unsafe { **dst_ptr = *msg }
        }
    }
}
