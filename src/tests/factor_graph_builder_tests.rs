use rand::{distributions::Uniform, thread_rng, Rng};

use crate::core::{Factor, FactorGraphBuilder, Message, Variable};

// The simples fake implementation of the message passing traits.
// Note, that it is nonsense for all the applications apart
// usage for validating builder. ------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FakeMessage(usize);

#[derive(Debug)]
struct FakeFactor(usize);

#[derive(Debug)]
struct FakeVariable;

impl Message for FakeMessage {
    #[inline(always)]
    fn discrepancy(&self, _: &Self) -> f64 {
        unimplemented!()
    }
}

impl Factor for FakeFactor {
    type Message = FakeMessage;
    type Parameters = ();

    #[inline(always)]
    fn degree(&self) -> usize {
        self.0
    }

    #[inline(always)]
    fn send_messages(&self, _: &[Self::Message], _: &mut [Self::Message], _: &()) {
        unimplemented!()
    }
}

impl Variable for FakeVariable {
    type Message = FakeMessage;
    type Marginal = ();

    #[inline(always)]
    fn new() -> Self {
        FakeVariable
    }

    #[inline(always)]
    fn send_messages(&self, _: &[Self::Message], _: &mut [Self::Message]) {
        unimplemented!()
    }

    #[inline(always)]
    fn marginal(&self, _: &[Self::Message]) -> Self::Marginal {
        unimplemented!()
    }
}

// ------------------------------------------------------------------------------------------

#[test]
fn small_factor_graph_builder_logic() {
    let mut rng = thread_rng();
    let mut mesage_initializer = || FakeMessage(rng.sample(Uniform::new(usize::MIN, usize::MAX)));
    let mut fgb = FactorGraphBuilder::<FakeFactor, FakeVariable>::new_with_variables(4, 3);
    fgb.add_factor(FakeFactor(3), &[0, 1, 3], &mut mesage_initializer)
        .unwrap();
    fgb.add_factor(FakeFactor(2), &[1, 2], &mut mesage_initializer)
        .unwrap();
    fgb.add_factor(FakeFactor(2), &[3, 1], &mut mesage_initializer)
        .unwrap();
    let fg = fgb.build();
    let fac0 = &fg.factors[0];
    let fac1 = &fg.factors[1];
    let fac2 = &fg.factors[2];
    assert_eq!(3, fg.factors.len());
    let var0 = &fg.variables[0];
    let var1 = &fg.variables[1];
    let var2 = &fg.variables[2];
    let var3 = &fg.variables[3];
    assert_eq!(4, fg.variables.len());
    // --------------------------------------------------------------------------------------
    assert_eq!(fac0.receivers.len(), 3);
    assert_eq!(fac1.receivers.len(), 2);
    assert_eq!(fac2.receivers.len(), 2);
    assert_eq!(fac0.messages.len(), 3);
    assert_eq!(fac1.messages.len(), 2);
    assert_eq!(fac2.messages.len(), 2);
    // --------------------------------------------------------------------------------------
    assert_eq!(var0.receivers.len(), 1);
    assert_eq!(var1.receivers.len(), 3);
    assert_eq!(var2.receivers.len(), 1);
    assert_eq!(var3.receivers.len(), 2);
    assert_eq!(var0.messages.len(), 1);
    assert_eq!(var1.messages.len(), 3);
    assert_eq!(var2.messages.len(), 1);
    assert_eq!(var3.messages.len(), 2);
    // --------------------------------------------------------------------------------------
    assert_eq!(fac0.var_node_indices, [0, 1, 3]);
    assert_eq!(fac1.var_node_indices, [1, 2]);
    assert_eq!(fac2.var_node_indices, [3, 1]);
    // --------------------------------------------------------------------------------------
    assert_eq!(var0.fac_node_indices, [0]);
    assert_eq!(var1.fac_node_indices, [0, 1, 2]);
    assert_eq!(var2.fac_node_indices, [1]);
    assert_eq!(var3.fac_node_indices, [0, 2]);
    // --------------------------------------------------------------------------------------
    assert_eq!(fac0.var_node_receiver_indices, [0, 0, 0]);
    assert_eq!(fac1.var_node_receiver_indices, [1, 0]);
    assert_eq!(fac2.var_node_receiver_indices, [1, 2]);
    // --------------------------------------------------------------------------------------
    assert_eq!(var0.fac_node_receiver_indices, [0]);
    assert_eq!(var1.fac_node_receiver_indices, [1, 0, 1]);
    assert_eq!(var2.fac_node_receiver_indices, [1]);
    assert_eq!(var3.fac_node_receiver_indices, [2, 0]);
    // --------------------------------------------------------------------------------------
    assert_eq!(unsafe { (*fac0.senders[0]).0 }, var0.receivers[0].0);
    assert_eq!(unsafe { (*fac0.senders[1]).0 }, var1.receivers[0].0);
    assert_eq!(unsafe { (*fac0.senders[2]).0 }, var3.receivers[0].0);
    assert_eq!(unsafe { (*fac1.senders[0]).0 }, var1.receivers[1].0);
    assert_eq!(unsafe { (*fac1.senders[1]).0 }, var2.receivers[0].0);
    assert_eq!(unsafe { (*fac2.senders[0]).0 }, var3.receivers[1].0);
    assert_eq!(unsafe { (*fac2.senders[1]).0 }, var1.receivers[2].0);
    // --------------------------------------------------------------------------------------
    assert_eq!(unsafe { (*var0.senders[0]).0 }, fac0.receivers[0].0);
    assert_eq!(unsafe { (*var1.senders[0]).0 }, fac0.receivers[1].0);
    assert_eq!(unsafe { (*var1.senders[1]).0 }, fac1.receivers[0].0);
    assert_eq!(unsafe { (*var1.senders[2]).0 }, fac2.receivers[1].0);
    assert_eq!(unsafe { (*var2.senders[0]).0 }, fac1.receivers[1].0);
    assert_eq!(unsafe { (*var3.senders[0]).0 }, fac0.receivers[2].0);
    assert_eq!(unsafe { (*var3.senders[1]).0 }, fac2.receivers[0].0);
}
