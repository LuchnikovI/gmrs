use std::{error::Error, fmt::Display};

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    core::factor::Factor, core::factor_node::FactorNode, core::variable::Variable,
    core::variable_node::VariableNode,
};

use serde::{Deserialize, Serialize};

// ------------------------------------------------------------------------------------------

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

pub type MessagePassingResult = Result<MessagePassingInfo, MessagePassingError>;

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
    /// * `max_iterations_number` - A maximal number of iteration
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
        parameters: F::Parameters,
    ) -> MessagePassingResult {
        let mut last_discrepancy = f64::MAX;
        for i in 0..max_iterations_number {
            let factors_discrepancy = self
                .factors
                .par_iter_mut()
                .map(|factor| {
                    factor.eval_messages(&parameters);
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
        })
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
}
