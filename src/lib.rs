/// A module containing general logic of factor graphs
pub mod core;
/// A module containing message passing algorithms implementation specific for Ising like models on an arbitrary graph
pub mod ising;

#[cfg(test)]
mod tests;
