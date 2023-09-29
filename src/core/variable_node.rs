use crate::{
    core::factor::Factor, core::factor_node::FactorNode, core::message::Message,
    core::variable::Variable,
};

#[derive(Debug, Clone)]
pub struct VariableNode<V, F>
where
    V: Variable,
    F: Factor<Message = V::Message>,
{
    variable: V,
    pub(crate) fac_node_indices: Vec<usize>,
    pub(crate) fac_node_receiver_indices: Vec<usize>,
    pub(crate) messages: Vec<F::Message>,
    pub(crate) senders: Vec<*mut F::Message>,
    pub(crate) receivers: Vec<V::Message>,
}

unsafe impl<V, F> Send for VariableNode<V, F>
where
    V: Variable,
    F: Factor<Message = V::Message>,
{
}

impl<V, F> VariableNode<V, F>
where
    V: Variable,
    F: Factor<Message = V::Message>,
{
    #[inline(always)]
    pub(super) fn new_disconnected() -> Self {
        let variable = V::new();
        VariableNode {
            variable,
            fac_node_indices: Vec::new(),
            messages: Vec::new(),
            fac_node_receiver_indices: Vec::new(),
            senders: Vec::new(),
            receivers: Vec::new(),
        }
    }

    #[inline(always)]
    pub(super) fn degree(&self) -> usize {
        self.receivers.len()
    }

    #[inline(always)]
    pub(super) fn init_senders(&mut self, factors: &mut [FactorNode<F, V>]) {
        let indices_iter = self
            .fac_node_indices
            .iter()
            .zip(&self.fac_node_receiver_indices);
        let senders: Vec<_> = indices_iter
            .map(|(fac_index, fac_receiver_index)| unsafe {
                let var = factors.get_unchecked_mut(*fac_index);
                let mut_ptr: *mut _ = var.receivers.get_unchecked_mut(*fac_receiver_index);
                mut_ptr
            })
            .collect();
        self.senders = senders;
    }

    #[inline(always)]
    pub(super) fn eval_messages(&mut self) {
        self.variable
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

    #[inline(always)]
    pub(super) fn marginal(&self) -> V::Marginal {
        self.variable.marginal(&self.receivers)
    }
}
