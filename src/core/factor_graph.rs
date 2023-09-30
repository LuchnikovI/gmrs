use std::{error::Error, fmt::Display};

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    core::factor::Factor, core::factor_node::FactorNode, core::variable::Variable,
    core::variable_node::VariableNode,
};

use serde::{Deserialize, Serialize};

use rand::Rng;

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// Possible factor graph errors
pub enum FGError {
    /// Message passing error when method does not converge
    MessagePassingError(MessagePassingError),

    /// ID (index) of a variable is out of range of variables list
    OutOfRangeVariable(usize, usize),
}

impl Display for FGError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FGError::MessagePassingError(err) => write!(f, "{err}"),
            FGError::OutOfRangeVariable(size, pos) => write!(
                f,
                "ID (index) of a variable {} is out of range of [0..{}] variables",
                pos, size,
            ),
        }
    }
}

impl From<MessagePassingError> for FGError {
    fn from(value: MessagePassingError) -> Self {
        FGError::MessagePassingError(value)
    }
}

/// Error info that appears if message passing does not converge
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MessagePassingError {
    /// Number of iterations past
    pub iterations_number: usize,

    /// Final discrepancy between last and previous iteration messages maximal across variables and factors
    pub final_discrepancy: f64,

    /// Convergence threshold: discrepancy must be smaller than the threshold for convergence
    pub threshold: f64,
}

impl Display for MessagePassingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Messaged passing has not converged after {} iterations, discrepancy threshold: {}, last iteration discrepancy: {}",
            self.iterations_number,
            self.threshold,
            self.final_discrepancy,
        )
    }
}

impl Error for MessagePassingError {}

/// Info that appears if message passing converges
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MessagePassingInfo {
    /// Number of iterations past
    pub iterations_number: usize,

    /// Final discrepancy between last and previous iteration messages maximal across variables and factors
    pub final_discrepancy: f64,

    /// Convergence threshold: discrepancy must be smaller than the threshold for convergence
    pub threshold: f64,
}

impl Display for MessagePassingInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Messaged passing has converged after {} iterations, discrepancy threshold: {}, last iteration discrepancy: {}",
            self.iterations_number,
            self.threshold,
            self.final_discrepancy,
        )
    }
}

pub type FGResult<T> = Result<T, FGError>;

// ------------------------------------------------------------------------------------------

/// A factor graph
#[derive(Debug)]
pub struct FactorGraph<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    pub(crate) factors: Vec<FactorNode<F, V>>,
    pub(crate) variables: Vec<VariableNode<V, F>>,
}

impl<F, V> Clone for FactorGraph<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    fn clone(&self) -> Self {
        let mut factors = self.factors.clone();
        let mut variables = self.variables.clone();
        for factor in &mut factors {
            factor.init_senders(&mut variables);
        }
        for variable in &mut variables {
            variable.init_senders(&mut factors);
        }
        FactorGraph { factors, variables }
    }
}

impl<F, V> FactorGraph<F, V>
where
    F: Factor,
    V: Variable<Message = F::Message>,
{
    /// Returns degree (number of adjoint factors) of each variable
    #[inline]
    pub fn get_variable_degrees(&self) -> Vec<usize> {
        self.variables.iter().map(|x| x.degree()).collect()
    }

    /// Returns degree (number of adjoint factors) of each factor
    /// in order they were added to a factor graph
    #[inline]
    pub fn get_factors_degrees(&self) -> Vec<usize> {
        self.factors.iter().map(|x| x.degree()).collect()
    }

    /// Runs a message passing algorithm in parallel
    ///
    /// # Arguments
    ///
    /// * `max_iterations_number` - A maximal number of iterations
    /// * `threshold` - A threshold specifying the convergence criterion: if
    ///     the discrepancy between last and previous iteration messages maximal
    ///     across variables and factors is smaller than this threshold,
    ///     message passing is converged
    /// * `parameters` - Hyper parameters of message passing
    #[inline]
    pub fn run_message_passing_parallel(
        &mut self,
        max_iterations_number: usize,
        threshold: f64,
        parameters: &F::Parameters,
    ) -> FGResult<MessagePassingInfo> {
        let mut last_discrepancy = f64::MAX;
        for i in 0..max_iterations_number {
            let factors_discrepancy = self
                .factors
                .par_iter_mut()
                .map(|factor| {
                    factor.eval_messages(parameters);
                    let max_discrepancy = factor.eval_discrepancy();
                    factor.send_messages();
                    max_discrepancy
                })
                .reduce(|| 0f64, |x, y| x.max(y));
            let variables_discrepancy = self
                .variables
                .par_iter_mut()
                .map(|variable| {
                    variable.eval_messages();
                    let max_discrepancy = variable.eval_discrepancy();
                    variable.send_messages();
                    max_discrepancy
                })
                .reduce(|| 0f64, |x, y| x.max(y));
            let max_discrepancy = factors_discrepancy.max(variables_discrepancy);
            if max_discrepancy < threshold {
                return Ok(MessagePassingInfo {
                    iterations_number: i,
                    threshold,
                    final_discrepancy: max_discrepancy,
                });
            }
            last_discrepancy = max_discrepancy;
        }
        Err(MessagePassingError {
            iterations_number: max_iterations_number,
            threshold,
            final_discrepancy: last_discrepancy,
        }
        .into())
    }

    /// Computes marginals for all variables
    #[inline]
    pub fn variable_marginals(&self) -> Vec<V::Marginal> {
        self.variables.iter().map(|x| x.marginal()).collect()
    }

    /// Computes marginals for all factors
    #[inline]
    pub fn factor_marginals(&self) -> Vec<F::Marginal> {
        self.factors.iter().map(|x| x.marginal()).collect()
    }

    /// Return factors as standalone objects
    ///
    /// # Notes
    ///
    /// The most natural data structure representing a standalone factor
    /// is that used to represent a marginal
    #[inline]
    pub fn factors(&self) -> Vec<F::Marginal> {
        self.factors.iter().map(|x| x.factor()).collect()
    }

    /// Adds a unit degree factor fixing a variable value
    ///
    /// # Arguments
    ///
    /// * `value` - A value of a fixed variable
    /// * `var_index` - The index of a fixed variable
    ///
    /// # Notes
    ///
    /// One should not freeze one variable twice or more times. This
    /// could lead to nonsense result
    #[inline]
    pub fn freeze_variable(&mut self, value: &V::Sample, var_index: usize) -> FGResult<()> {
        let message = V::sample_to_message(value);
        let factor = F::from_message(&message);
        let degree = factor.degree();
        if degree != 1 {
            panic!(
                "Factor degree is {degree} but it must be 1. This is a bug, please make an issue."
            )
        }
        let factor_node = FactorNode::<F, V>::new_disconnected(factor);
        self.factors.push(factor_node);
        let factor_node = self.factors.last_mut().unwrap();
        let variable_node = if let Some(var) = self.variables.get_mut(var_index) {
            var
        } else {
            return Err(FGError::OutOfRangeVariable(self.variables.len(), var_index));
        };
        let receivers_len = variable_node.receivers.len();
        let receivers_cap = variable_node.receivers.capacity();
        factor_node.receivers.push(message);
        factor_node.messages.push(message);
        factor_node.var_node_indices.push(var_index);
        factor_node.var_node_receiver_indices.push(receivers_len);
        variable_node.receivers.push(message);
        factor_node
            .senders
            .push(variable_node.receivers.last_mut().unwrap());
        variable_node
            .senders
            .push(factor_node.receivers.last_mut().unwrap());
        variable_node.messages.push(message);
        variable_node.fac_node_indices.push(self.factors.len() - 1);
        variable_node.fac_node_receiver_indices.push(0);
        // if reallocation happens, one needs to reinitialize pointers
        if receivers_cap == receivers_len {
            let receivers_iter = variable_node.receivers.iter_mut();
            let fac_node_index_iter = variable_node.fac_node_indices.iter();
            let fac_node_receiver_index_iter = variable_node.fac_node_receiver_indices.iter();
            let iter = receivers_iter.zip(fac_node_index_iter.zip(fac_node_receiver_index_iter));
            for (r, (i, ri)) in iter {
                unsafe {
                    *self
                        .factors
                        .get_unchecked_mut(*i)
                        .senders
                        .get_unchecked_mut(*ri) = r;
                }
            }
        }
        Ok(())
    }

    /// Samples from a factor graph
    ///
    /// # Arguments
    ///
    /// * `max_iterations_number` - A maximal number of iterations in a message passing algorithm
    /// * `threshold` - A threshold specifying the convergence criterion: if
    ///     the discrepancy between last and previous iteration messages maximal
    ///     across variables and factors is smaller than this threshold,
    ///     message passing is converged
    /// * `parameters` - Hyper parameters of message passing
    /// * `rng` - A random numbers generator
    ///
    /// # Notes
    ///
    /// Sampling requires to run message passing after sampling and fixing of each
    /// variable to get stationary messages for the updated factor graph.
    /// This is why one has some arguments similar to those of 'run_message_passing_parallel'
    /// method. Note also, that this method fixes all variables of a factor graph making
    /// them further unusable. To keep the initial graph simply clone it before running
    /// sampling
    pub fn sample(
        &mut self,
        max_iterations_number: usize,
        threshold: f64,
        parameters: &F::Parameters,
        rng: &mut impl Rng,
    ) -> FGResult<Vec<V::Sample>> {
        let variables_number = self.variables.len();
        let mut samples = Vec::with_capacity(variables_number);
        for i in 0..variables_number {
            let sample = self.variables.get_mut(i).unwrap().sample(rng);
            samples.push(sample);
            self.freeze_variable(&sample, i).unwrap();
            self.run_message_passing_parallel(max_iterations_number, threshold, parameters)?;
        }
        Ok(samples)
    }
}
