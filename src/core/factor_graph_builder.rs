use std::{error::Error, fmt::Display, iter::from_fn, ptr::null_mut};

use crate::{
    core::factor::Factor, core::factor_graph::FactorGraph, core::factor_node::FactorNode,
    core::variable::Variable, core::variable_node::VariableNode,
};

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors that could appear in factor graph builder's methods
pub enum FGBuilderError {
    /// Degree of a factor does not match a number of adjacent variables
    DegreeError(usize, Vec<usize>),

    /// Index of a variable is out of range
    OutOfRangeVariable(usize, usize),
}

impl Display for FGBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FGBuilderError::DegreeError(deg, vars) => write!(
                f,
                "Degree of a factor does not match the number of variables. The factor's degree: {}, the variables list {:?}",
                deg,
                vars,
            ),
            FGBuilderError::OutOfRangeVariable(size, pos) => write!(
                f,
                "ID (index) of a variable {} is out of range of [0..{}] variables",
                pos,
                size,
            ),
        }
    }
}

impl Error for FGBuilderError {}

/// Factor graph builder's methods result type
pub type FGBuilderResult<T> = Result<T, FGBuilderError>;

// public methods ---------------------------------------------------------------------------

#[derive(Debug)]
/// A factor graph builder
pub struct FactorGraphBuilder<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    factors: Vec<FactorNode<F, V>>,
    variables: Vec<VariableNode<V, F>>,
}

impl<F, V> Default for FactorGraphBuilder<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, V> FactorGraphBuilder<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    /// Creates an empty factor graph
    ///
    /// # Example
    ///
    /// ```
    /// use gmrs::core::FactorGraphBuilder;
    /// use gmrs::ising::{IsingFactor, IsingVariable, SumProduct};
    ///
    /// // Aliases to shorten types
    /// type Factor = IsingFactor<SumProduct>;
    /// type Variable = IsingVariable<SumProduct>;
    ///
    /// let fgb = FactorGraphBuilder::<Factor, Variable>::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        FactorGraphBuilder {
            factors: Vec::new(),
            variables: Vec::new(),
        }
    }

    /// Creates a factor graph with predefined set of variables and
    /// preallocated memory for factors
    ///
    /// # Arguments
    ///
    /// * `variables_number` - A number of variables
    /// * `factors_capacity` - A number of factors we need to preallocate memory for
    ///
    /// # Example
    ///
    /// ```
    /// use gmrs::core::FactorGraphBuilder;
    /// use gmrs::ising::{IsingFactor, IsingVariable, SumProduct};
    ///
    /// // Aliases to shorten types
    /// type Factor = IsingFactor<SumProduct>;
    /// type Variable = IsingVariable<SumProduct>;
    ///
    /// let fgb = FactorGraphBuilder::<Factor, Variable>::new_with_variables(2, 2);
    /// ```
    #[inline]
    pub fn new_with_variables(variables_number: usize, factors_capacity: usize) -> Self {
        let variables: Vec<_> = from_fn(|| Some(VariableNode::new_disconnected()))
            .take(variables_number)
            .collect();
        let factors = Vec::with_capacity(factors_capacity);
        FactorGraphBuilder { factors, variables }
    }

    /// Adds a variable to a factor graph
    ///
    /// # Example
    ///
    /// ```
    /// use gmrs::core::FactorGraphBuilder;
    /// use gmrs::ising::{IsingFactor, IsingVariable, SumProduct};
    ///
    /// // Aliases to shorten types
    /// type Factor = IsingFactor<SumProduct>;
    /// type Variable = IsingVariable<SumProduct>;
    ///
    /// let mut fgb = FactorGraphBuilder::<Factor, Variable>::new_with_variables(0, 1);
    /// fgb.add_variable();
    /// ```
    #[inline]
    pub fn add_variable(&mut self) {
        self.variables.push(VariableNode::new_disconnected())
    }

    /// Adds a factor to a factor graph
    ///
    /// # Arguments
    ///
    /// * `factor` - A new factor
    /// * `var_indices` - Indices of adjoint variables
    /// * `message_initializer` - An object that initializes messages
    ///
    /// # Notes
    ///
    /// If number of `var_indices` does not match a factor degree, the method
    /// returns an error. If an index from `var_indices` is out of range of
    /// the variables list, the method returns an error
    ///
    /// # Example
    ///
    /// ```
    /// use gmrs::core::FactorGraphBuilder;
    /// use gmrs::ising::{IsingFactor, IsingVariable, SumProduct, random_message_initializer};
    /// use rand::thread_rng;
    ///
    /// // Aliases to shorten types
    /// type Factor = IsingFactor<SumProduct>;
    /// type Variable = IsingVariable<SumProduct>;
    ///
    /// // Builder creation
    /// let mut fgb = FactorGraphBuilder::<Factor, Variable>::new_with_variables(10, 1);
    ///
    /// // Messages initializer
    /// let rng = thread_rng();
    /// let mut initializer = random_message_initializer(rng, -0.5, 0.5);
    ///
    /// fgb.add_factor(
    ///     IsingFactor::new(0.5f64, -0.5f64, 0.5f64),
    ///    &[3, 8],
    ///    &mut initializer,
    /// );
    /// ```
    #[inline]
    pub fn add_factor(
        &mut self,
        factor: F,
        var_indices: &[usize],
        message_initializer: &mut impl FnMut() -> F::Message,
    ) -> FGBuilderResult<()> {
        let factor_deg = var_indices.len();
        if factor.degree() != factor_deg {
            return Err(FGBuilderError::DegreeError(
                factor.degree(),
                var_indices.to_vec(),
            ));
        }
        let factor_node = FactorNode::new_disconnected(factor);
        self.factors.push(factor_node);
        let factors_number = self.factors.len();
        let MutFactorsAndVariables { factors, variables } = self.factors_and_variables();
        let last_factor = factors.last_mut().unwrap();
        for index in var_indices {
            let variable = if let Some(v) = variables.get_mut(*index) {
                v
            } else {
                return Err(FGBuilderError::OutOfRangeVariable(variables.len(), *index));
            };
            let factor_message = message_initializer();
            let variable_message = message_initializer();
            last_factor.senders.push(null_mut());
            last_factor.receivers.push(factor_message);
            last_factor.messages.push(variable_message);
            last_factor.var_node_indices.push(*index);
            variable.senders.push(null_mut());
            variable.messages.push(factor_message);
            variable.receivers.push(variable_message);
            variable.fac_node_indices.push(factors_number - 1);
            let variable_recivers_number = variable.receivers.len();
            let factor_recivers_number = last_factor.receivers.len();
            last_factor
                .var_node_receiver_indices
                .push(variable_recivers_number - 1);
            variable
                .fac_node_receiver_indices
                .push(factor_recivers_number - 1);
        }
        Ok(())
    }

    /// Returns a factor graph
    ///
    /// # Example
    ///
    /// ```
    /// use gmrs::core::FactorGraphBuilder;
    /// use gmrs::ising::{IsingFactor, IsingVariable, SumProduct, random_message_initializer};
    /// use rand::thread_rng;
    ///
    /// // Aliases to shorten types
    /// type Factor = IsingFactor<SumProduct>;
    /// type Variable = IsingVariable<SumProduct>;
    ///
    /// // Builder creation
    /// let mut fgb = FactorGraphBuilder::<Factor, Variable>::new_with_variables(10, 9);
    ///
    /// // Messages initializer
    /// let rng = thread_rng();
    /// let mut initializer = random_message_initializer(rng, -0.5, 0.5);
    ///
    /// for i in 0..9 {
    ///    fgb.add_factor(
    ///        IsingFactor::new(0.5f64, -0.5f64, 0.5f64),
    ///        &[i, i + 1],
    ///        &mut initializer,
    ///    );
    /// }
    ///
    /// // Building a factor graph
    /// let fg = fgb.build();
    /// ```
    #[inline]
    pub fn build(mut self) -> FactorGraph<F, V> {
        for factor in &mut self.factors {
            factor.init_senders(&mut self.variables);
        }
        for variable in &mut self.variables {
            variable.init_senders(&mut self.factors);
        }
        FactorGraph {
            factors: self.factors,
            variables: self.variables,
        }
    }
}

// private methods --------------------------------------------------------------------------

struct MutFactorsAndVariables<'a, F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    factors: &'a mut Vec<FactorNode<F, V>>,
    variables: &'a mut Vec<VariableNode<V, F>>,
}

impl<F, V> FactorGraphBuilder<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    #[inline(always)]
    fn factors_and_variables(&mut self) -> MutFactorsAndVariables<F, V> {
        let mut_ptr: *mut Self = self;
        unsafe {
            MutFactorsAndVariables {
                factors: &mut (*mut_ptr).factors,
                variables: &mut (*mut_ptr).variables,
            }
        }
    }
}
