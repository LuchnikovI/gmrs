use rand::{distributions::Uniform, thread_rng, Rng};

use crate::core::{Factor, FactorGraphBuilder, Message, Variable};

// The simples fake implementation of the message passing traits.
// Note, that it is nonsense for all the applications apart
// usage for validating builder. ------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FakeMessage(usize);

#[derive(Debug, Clone)]
struct FakeFactor(usize);

#[derive(Debug, Clone)]
struct FakeVariable;

impl Message for FakeMessage {
    #[inline(always)]
    fn discrepancy(&self, _: &Self) -> f64 {
        unimplemented!()
    }
}

impl Factor for FakeFactor {
    type Message = FakeMessage;
    type Marginal = ();
    type Parameters = ();

    #[inline(always)]
    fn from_message(_: &Self::Message) -> Self {
        FakeFactor(1)
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        self.0
    }

    #[inline(always)]
    fn marginal(&self, _: &[Self::Message]) -> Self::Marginal {
        unimplemented!()
    }

    #[inline(always)]
    fn send_messages(&self, _: &[Self::Message], _: &mut [Self::Message], _: &()) {
        unimplemented!()
    }

    #[inline(always)]
    fn factor(&self) -> Self::Marginal {
        unimplemented!()
    }
}

impl Variable for FakeVariable {
    type Message = FakeMessage;
    type Marginal = ();
    type Sample = usize;
    type Parameters = ();

    #[inline(always)]
    fn send_messages(&self, _: &[Self::Message], _: &mut [Self::Message], _: &()) {
        unimplemented!()
    }

    #[inline(always)]
    fn marginal(&self, _: &[Self::Message]) -> Self::Marginal {
        unimplemented!()
    }

    #[inline(always)]
    fn sample(&self, _: &[Self::Message], _: &mut impl Rng) -> Self::Sample {
        unimplemented!()
    }

    #[inline(always)]
    fn sample_to_message(sample: &Self::Sample) -> Self::Message {
        FakeMessage(*sample)
    }
}

// ------------------------------------------------------------------------------------------

#[test]
fn small_factor_graph_builder_logic() {
    let mut rng = thread_rng();
    let mut mesage_initializer = || FakeMessage(rng.sample(Uniform::new(usize::MIN, usize::MAX)));
    let mut fgb = FactorGraphBuilder::<FakeFactor, FakeVariable>::new_with_capacity(4, 3);
    fgb.fill(FakeVariable);
    fgb.add_factor(FakeFactor(3), &[0, 1, 3], &mut mesage_initializer)
        .unwrap();
    fgb.add_factor(FakeFactor(2), &[1, 2], &mut mesage_initializer)
        .unwrap();
    fgb.add_factor(FakeFactor(2), &[3, 1], &mut mesage_initializer)
        .unwrap();
    let mut fg1 = fgb.build();
    fg1.freeze_variable(&0, 0).unwrap();
    fg1.freeze_variable(&1, 1).unwrap();
    fg1.freeze_variable(&2, 2).unwrap();
    fg1.freeze_variable(&3, 3).unwrap();
    fgb = FactorGraphBuilder::<FakeFactor, FakeVariable>::new();
    fgb.add_variable(FakeVariable);
    fgb.add_variable(FakeVariable);
    fgb.add_variable(FakeVariable);
    fgb.add_variable(FakeVariable);
    fgb.add_factor(FakeFactor(3), &[0, 1, 3], &mut mesage_initializer)
        .unwrap();
    fgb.add_factor(FakeFactor(2), &[1, 2], &mut mesage_initializer)
        .unwrap();
    fgb.add_factor(FakeFactor(2), &[3, 1], &mut mesage_initializer)
        .unwrap();
    let mut fg2 = fgb.build();
    fg2.freeze_variable(&0, 0).unwrap();
    fg2.freeze_variable(&1, 1).unwrap();
    fg2.freeze_variable(&2, 2).unwrap();
    fg2.freeze_variable(&3, 3).unwrap();
    let fg3 = fg2.clone();
    for fg in [fg1, fg2, fg3] {
        let fac0 = &fg.factors[0];
        let fac1 = &fg.factors[1];
        let fac2 = &fg.factors[2];
        let freeze1 = &fg.factors[3];
        let freeze2 = &fg.factors[4];
        let freeze3 = &fg.factors[5];
        let freeze4 = &fg.factors[6];
        assert_eq!(7, fg.factors.len());
        let var0 = &fg.variables[0];
        let var1 = &fg.variables[1];
        let var2 = &fg.variables[2];
        let var3 = &fg.variables[3];
        assert_eq!(4, fg.variables.len());
        // --------------------------------------------------------------------------------------
        assert_eq!(fac0.receivers.len(), 3);
        assert_eq!(fac1.receivers.len(), 2);
        assert_eq!(fac2.receivers.len(), 2);
        assert_eq!(freeze1.receivers.len(), 1);
        assert_eq!(freeze2.receivers.len(), 1);
        assert_eq!(freeze3.receivers.len(), 1);
        assert_eq!(freeze4.receivers.len(), 1);
        assert_eq!(fac0.messages.len(), 3);
        assert_eq!(fac1.messages.len(), 2);
        assert_eq!(fac2.messages.len(), 2);
        assert_eq!(freeze1.messages.len(), 1);
        assert_eq!(freeze2.messages.len(), 1);
        assert_eq!(freeze3.messages.len(), 1);
        assert_eq!(freeze4.messages.len(), 1);
        // --------------------------------------------------------------------------------------
        assert_eq!(var0.receivers.len(), 2);
        assert_eq!(var1.receivers.len(), 4);
        assert_eq!(var2.receivers.len(), 2);
        assert_eq!(var3.receivers.len(), 3);
        assert_eq!(var0.messages.len(), 2);
        assert_eq!(var1.messages.len(), 4);
        assert_eq!(var2.messages.len(), 2);
        assert_eq!(var3.messages.len(), 3);
        // --------------------------------------------------------------------------------------
        assert_eq!(fac0.var_node_indices, [0, 1, 3]);
        assert_eq!(fac1.var_node_indices, [1, 2]);
        assert_eq!(fac2.var_node_indices, [3, 1]);
        assert_eq!(freeze1.var_node_indices, [0]);
        assert_eq!(freeze2.var_node_indices, [1]);
        assert_eq!(freeze3.var_node_indices, [2]);
        assert_eq!(freeze4.var_node_indices, [3]);
        // --------------------------------------------------------------------------------------
        assert_eq!(var0.fac_node_indices, [0, 3]);
        assert_eq!(var1.fac_node_indices, [0, 1, 2, 4]);
        assert_eq!(var2.fac_node_indices, [1, 5]);
        assert_eq!(var3.fac_node_indices, [0, 2, 6]);
        // --------------------------------------------------------------------------------------
        assert_eq!(fac0.var_node_receiver_indices, [0, 0, 0]);
        assert_eq!(fac1.var_node_receiver_indices, [1, 0]);
        assert_eq!(fac2.var_node_receiver_indices, [1, 2]);
        assert_eq!(freeze1.var_node_receiver_indices, [1]);
        assert_eq!(freeze2.var_node_receiver_indices, [3]);
        assert_eq!(freeze3.var_node_receiver_indices, [1]);
        assert_eq!(freeze4.var_node_receiver_indices, [2]);
        // --------------------------------------------------------------------------------------
        assert_eq!(var0.fac_node_receiver_indices, [0, 0]);
        assert_eq!(var1.fac_node_receiver_indices, [1, 0, 1, 0]);
        assert_eq!(var2.fac_node_receiver_indices, [1, 0]);
        assert_eq!(var3.fac_node_receiver_indices, [2, 0, 0]);
        // --------------------------------------------------------------------------------------
        assert_eq!(unsafe { (*fac0.senders[0]).0 }, var0.receivers[0].0);
        assert_eq!(unsafe { (*fac0.senders[1]).0 }, var1.receivers[0].0);
        assert_eq!(unsafe { (*fac0.senders[2]).0 }, var3.receivers[0].0);
        assert_eq!(unsafe { (*fac1.senders[0]).0 }, var1.receivers[1].0);
        assert_eq!(unsafe { (*fac1.senders[1]).0 }, var2.receivers[0].0);
        assert_eq!(unsafe { (*fac2.senders[0]).0 }, var3.receivers[1].0);
        assert_eq!(unsafe { (*fac2.senders[1]).0 }, var1.receivers[2].0);
        assert_eq!(
            unsafe { (*freeze1.senders[0]).0 },
            var0.receivers.last().unwrap().0
        );
        assert_eq!(
            unsafe { (*freeze2.senders[0]).0 },
            var1.receivers.last().unwrap().0
        );
        assert_eq!(
            unsafe { (*freeze3.senders[0]).0 },
            var2.receivers.last().unwrap().0
        );
        assert_eq!(
            unsafe { (*freeze4.senders[0]).0 },
            var3.receivers.last().unwrap().0
        );
        // --------------------------------------------------------------------------------------
        assert_eq!(var0.receivers.last().unwrap().0, 0);
        assert_eq!(var1.receivers.last().unwrap().0, 1);
        assert_eq!(var2.receivers.last().unwrap().0, 2);
        assert_eq!(var3.receivers.last().unwrap().0, 3);
        // --------------------------------------------------------------------------------------
        assert_eq!(unsafe { (*var0.senders[0]).0 }, fac0.receivers[0].0);
        assert_eq!(unsafe { (*var0.senders[1]).0 }, freeze1.receivers[0].0);
        assert_eq!(unsafe { (*var1.senders[0]).0 }, fac0.receivers[1].0);
        assert_eq!(unsafe { (*var1.senders[1]).0 }, fac1.receivers[0].0);
        assert_eq!(unsafe { (*var1.senders[2]).0 }, fac2.receivers[1].0);
        assert_eq!(unsafe { (*var1.senders[3]).0 }, freeze2.receivers[0].0);
        assert_eq!(unsafe { (*var2.senders[0]).0 }, fac1.receivers[1].0);
        assert_eq!(unsafe { (*var2.senders[1]).0 }, freeze3.receivers[0].0);
        assert_eq!(unsafe { (*var3.senders[0]).0 }, fac0.receivers[2].0);
        assert_eq!(unsafe { (*var3.senders[1]).0 }, fac2.receivers[0].0);
        assert_eq!(unsafe { (*var3.senders[2]).0 }, freeze4.receivers[0].0);
        drop(fg);
    }
}
