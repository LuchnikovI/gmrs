use crate::ising::{
    new_ising_builder, random_message_initializer,
    schedulers::{get_standard_factor_scheduler, get_standard_variable_scheduler},
    IsingFactor, MaxProduct,
};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rand_distr::Uniform;
use std::collections::VecDeque;

struct RandomTreeData {
    edges: Vec<[usize; 2]>,
    weights: Vec<f64>,
    argmax: Vec<i8>,
}

#[inline]
fn eval_energy(config: &[i8], edges: &[[usize; 2]], weights: &[f64]) -> f64 {
    let mut energy = 0f64;
    for ([n1, n2], w) in edges.iter().zip(weights) {
        let s1 = config[*n1] as f64;
        let s2 = config[*n2] as f64;
        energy += s1 * s2 * w;
    }
    energy
}

#[inline]
fn gen_random_ising_tree(
    rng: &mut impl Rng,
    nodes_number: usize,
    max_node_degree: usize,
) -> RandomTreeData {
    let distr = Uniform::new(1, max_node_degree);
    let mut edges = Vec::with_capacity(nodes_number - 1);
    let mut weights = Vec::with_capacity(nodes_number - 1);
    let mut argmax = Vec::with_capacity(nodes_number);
    argmax.push(-1i8);
    let mut nodes_queue = VecDeque::with_capacity(nodes_number);
    nodes_queue.push_front(0);
    let mut max_node_number = 0;
    while let Some(current_node) = nodes_queue.pop_back() {
        if nodes_number > max_node_number + 1 {
            let children_number =
                std::cmp::min(rng.sample(distr), nodes_number - max_node_number - 1);
            let current_state = argmax[current_node];
            for i in 0..children_number {
                let new_node = max_node_number + i + 1;
                let weight = 2f64 * rng.gen::<f64>() - 1f64;
                let mut new_edge = [current_node, new_node];
                new_edge.shuffle(rng);
                edges.push(new_edge);
                weights.push(weight);
                nodes_queue.push_front(new_node);
                if weight > 0f64 {
                    argmax.push(current_state);
                } else {
                    argmax.push(-current_state);
                }
            }
            max_node_number += children_number;
        } else {
            break;
        }
    }
    let mut indices: Vec<_> = (0..edges.len()).collect();
    indices.shuffle(rng);
    let edges = indices.iter().map(|i| edges[*i]).collect();
    let weights = indices.iter().map(|i| weights[*i]).collect();
    RandomTreeData {
        edges,
        weights,
        argmax,
    }
}

#[test]
fn maxcut_random_tree_test() {
    let mut rng = thread_rng();
    let nodes_number = 100;
    let max_node_degree = 6;
    let max_iterations_number = 1000;
    let min_iterations_number = 0;
    let error = 1e-10;
    let factor_scheduler = get_standard_factor_scheduler(0.2);
    let variable_scheduler = get_standard_variable_scheduler(0.2);
    let RandomTreeData {
        edges,
        weights,
        argmax,
    } = gen_random_ising_tree(&mut rng, nodes_number, max_node_degree);
    let mut fgb = new_ising_builder::<MaxProduct>(nodes_number, nodes_number - 1);
    let mut initializer = random_message_initializer(rng, -1., 1.);
    for (edge, weight) in edges.iter().zip(&weights) {
        fgb.add_factor(
            IsingFactor::new(*weight, 0f64, 0f64),
            edge,
            &mut initializer,
        )
        .unwrap();
    }
    let mut fg = fgb.build();
    let _ = fg
        .run_message_passing_parallel(
            max_iterations_number,
            min_iterations_number,
            error,
            &factor_scheduler,
            &variable_scheduler,
        )
        .unwrap();
    let mut energy = 0.;
    for (fm, f) in fg.factor_marginals().into_iter().zip(fg.factors()) {
        let (e, _) = f
            .iter()
            .zip(fm.iter())
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
            .unwrap();
        energy += e.ln();
    }
    let exact_energy = eval_energy(&argmax, &edges, &weights);
    assert!((energy - exact_energy).abs() < 1e-10);
    let mut rng = thread_rng();
    let sampling_info = fg
        .sample(
            max_iterations_number,
            min_iterations_number,
            error,
            &mut rng,
            &factor_scheduler,
            &variable_scheduler,
        )
        .unwrap();
    let decimation_energy = eval_energy(&sampling_info.samples, &edges, &weights);
    assert!((decimation_energy - exact_energy).abs() < 1e-10);
}
