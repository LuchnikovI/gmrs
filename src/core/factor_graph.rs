use std::{error::Error, fmt::Display};

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    core::factor::Factor, core::factor_node::FactorNode, core::variable::Variable,
    core::variable_node::VariableNode,
};

use serde::{Deserialize, Serialize};

use rand::Rng;

// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Errors that could appear in factor graph methods
pub enum FGError {
    /// Message passing error appears when a message passing does not converge
    MessagePassingError {
        /// Number of iterations past before failure
        iterations_number: usize,

        /// Final discrepancy between last and previous iteration's messages maximized across variables and factors
        last_discrepancy: f64,

        /// Dynamics of discrepancy before failure
        discrepancy_dynamics: Vec<f64>,
    },

    SamplingError {
        /// Number of successfully sampled variables
        variables_number: usize,

        ///  Total number of message passing iterations passed before failure
        total_iterations_number: usize,
    },

    /// Index of a variable is out of range
    OutOfRangeVariable(usize, usize),
}

impl Display for FGError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FGError::MessagePassingError {
                iterations_number,
                last_discrepancy,
                discrepancy_dynamics: _,
            } => write!(
                f,
                "Messaged passing has not converged after {} iterations, last iteration discrepancy: {}",
                iterations_number,
                last_discrepancy,
            ),
            FGError::OutOfRangeVariable(size, pos) => write!(
                f,
                "Index of a variable {} is out of range of [0..{}] variables",
                pos, size,
            ),
            FGError::SamplingError { variables_number, total_iterations_number } => {
                write!(
                    f,
                    "While sampling {}-th variable procedure has failed. Total number of message passing iteration passed before the failure: {}",
                    variables_number + 1,
                    total_iterations_number,
                )
            }
        }
    }
}

impl Error for FGError {}

/// Message passing algorithm information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePassingInfo {
    /// Number of iterations past before convergence
    pub iterations_number: usize,

    /// Final discrepancy between last and previous iteration's messages maximized across variables and factors
    pub last_discrepancy: f64,

    /// Dynamics of discrepancy before failure
    pub discrepancy_dynamics: Vec<f64>,
}

impl Display for MessagePassingInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Messaged passing has converged after {} iterations, last iteration discrepancy: {}",
            self.iterations_number, self.last_discrepancy,
        )
    }
}

pub type FGResult<T> = Result<T, FGError>;

/// Info that appears if sampling procedure succeed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingInfo<S> {
    /// Generated samples
    pub samples: Vec<S>,

    /// Number of passed message passing iterations per variable
    pub iterations_per_variable: Vec<usize>,

    /// Total number of message passing iterations passed
    pub total_iterations_number: usize,
}

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

    /// Returns degree (number of adjoint variables) of each factor
    /// in order they were added to a factor graph
    #[inline]
    pub fn get_factors_degrees(&self) -> Vec<usize> {
        self.factors.iter().map(|x| x.degree()).collect()
    }

    /// Runs a message passing algorithm in parallel. Typically, it is
    /// a fixed point iteration method targeted on achieving an equilibrium
    /// configuration of messages. This method mutates a factor graph
    /// since it updates messages that are stored internally.
    ///
    /// # Arguments
    ///
    /// * `max_iterations_number` - A maximal number of iterations, if a process
    ///     does not converge before reaching this number of iterations, it fails
    /// * `min_iterations_number` - A minimal number of iterations that is performed
    ///     disregards reaching the convergence criterion. It could be useful
    ///     for example when one performs belief propagation with temperature annealing
    /// * `threshold` - A threshold specifying the convergence criterion. A process
    ///     is considered as successful if the discrepancy between two subsequent
    ///     messages configurations is less than the threshold
    /// * `factor_scheduler` - A scheduler of a factor's messages update rule hyper-parameters.
    ///     It takes an iteration number (starts from 0) and return hyper-parameters.
    /// * `variable_scheduler` - A scheduler of a variable's messages update rule hyper-parameters.
    ///     It takes an iteration number (starts from 0) and return hyper-parameters.
    #[inline]
    pub fn run_message_passing_parallel(
        &mut self,
        max_iterations_number: usize,
        min_iterations_number: usize,
        threshold: f64,
        factor_scheduler: &impl Fn(usize) -> F::Parameters,
        variable_scheduler: &impl Fn(usize) -> V::Parameters,
    ) -> FGResult<MessagePassingInfo> {
        let mut last_discrepancy = f64::MAX;
        let mut discrepancy_dynamics = Vec::with_capacity(max_iterations_number);
        for i in 0..max_iterations_number {
            let factor_parameters = factor_scheduler(i);
            let variable_parameters = variable_scheduler(i);
            let factors_discrepancy = self
                .factors
                .par_iter_mut()
                .map(|factor| {
                    factor.eval_messages(&factor_parameters);
                    let max_discrepancy = factor.eval_discrepancy();
                    factor.send_messages();
                    max_discrepancy
                })
                .reduce(|| 0f64, |x, y| x.max(y));
            let variables_discrepancy = self
                .variables
                .par_iter_mut()
                .map(|variable| {
                    variable.eval_messages(&variable_parameters);
                    let max_discrepancy = variable.eval_discrepancy();
                    variable.send_messages();
                    max_discrepancy
                })
                .reduce(|| 0f64, |x, y| x.max(y));
            let max_discrepancy = factors_discrepancy.max(variables_discrepancy);
            discrepancy_dynamics.push(max_discrepancy);
            last_discrepancy = max_discrepancy;
            if (max_discrepancy < threshold) && (i + 1 >= min_iterations_number) {
                return Ok(MessagePassingInfo {
                    iterations_number: i,
                    discrepancy_dynamics,
                    last_discrepancy,
                });
            }
        }
        Err(FGError::MessagePassingError {
            iterations_number: max_iterations_number,
            discrepancy_dynamics,
            last_discrepancy,
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
    /// * `min_iterations_number` - A minimal number of iterations that is performed
    ///     disregards reaching the convergence criterion. It could be useful
    ///     for example when one performs belief propagation with temperature annealing
    /// * `threshold` - A threshold specifying the convergence criterion. A process
    ///     is considered as successful if the discrepancy between two subsequent
    ///     messages configurations is less than the threshold
    /// * `rng` - A random numbers generator
    /// * `factor_scheduler` - A scheduler of a factor's messages update rule hyper-parameters.
    ///     It takes an iteration number (starts from 0) and return hyper-parameters.
    /// * `variable_scheduler` - A scheduler of a variable's messages update rule hyper-parameters.
    ///     It takes an iteration number (starts from 0) and return hyper-parameters.
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
        min_iterations_number: usize,
        threshold: f64,
        rng: &mut impl Rng,
        factor_scheduler: &impl Fn(usize) -> F::Parameters,
        variable_scheduler: &impl Fn(usize) -> V::Parameters,
    ) -> FGResult<SamplingInfo<V::Sample>> {
        let variables_number = self.variables.len();
        let mut samples = Vec::with_capacity(variables_number);
        let mut total_iterations_number = 0;
        let mut iterations_per_variable = Vec::with_capacity(self.variables.len());
        for i in 0..variables_number {
            let sample = self.variables.get_mut(i).unwrap().sample(rng);
            samples.push(sample);
            self.freeze_variable(&sample, i).unwrap();
            match self.run_message_passing_parallel(
                max_iterations_number,
                min_iterations_number,
                threshold,
                factor_scheduler,
                variable_scheduler,
            ) {
                Ok(info) => {
                    total_iterations_number += info.iterations_number;
                    iterations_per_variable.push(info.iterations_number);
                }
                Err(info) => {
                    if let FGError::MessagePassingError {
                        iterations_number, ..
                    } = info
                    {
                        return Err(FGError::SamplingError {
                            variables_number: i,
                            total_iterations_number: total_iterations_number + iterations_number,
                        });
                    } else {
                        unreachable!()
                    }
                }
            }
        }
        Ok(SamplingInfo {
            samples,
            iterations_per_variable,
            total_iterations_number,
        })
    }
}
