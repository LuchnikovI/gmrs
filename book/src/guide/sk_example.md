# Sum-product message passing: Sherrington Kirkpatrick model

_Complete code of the example is in_ `./examples/sk.rs`

In this example we consider a Sherrington Kirkpatrick model of a spin glass. In particular, we are going to calculate a various thermodynamics quantities of the model and compare them with the corresponding exact quantities in the thermodynamic limit.

First, let us introduce a Sherrington Kirkpatrick (SK) model. Let \\( \mathbf{x} \in \\{+1,-1\\}^n\\) be a configuration of \\( n \\) spins. We define a probability measure over all those configurations as follows \\[ p_J[\mathbf{x}] =  \frac{1}{Z_{J}}\exp\left\\{\sum_{i, j=1, i>j}^n J_{ij}x_ix_j + B\sum_{i=1}^nx_i\right\\},\\] where \\( Z_J \\) is the partition function (normalization constant) and \\( J_{ij} \\) is a matrix of coupling constants. Coupling constants are i.i.d. random variables, each of them is sampled from the normal distribution with zero mean and \\( \frac{1}{2\sqrt{2}n} \\) standard deviation. As one can see, each particular realization of \\( p_J \\) is a statistical mechanics model. We would be interested in 'averaged' over \\( J \\) thermodynamics properties of all realizations that are the essence of the SK model. SK model is exactly solvable in the thermodynamic limit (\\( n\to\infty\\)), and therefore all the thermodynamic quantity could be calculated analytically in this limit.

We are going to solve SK model numerically, using sum-product message passing algorithm. One useful property of SK model is a _self-averaging_ which states that for \\( n\to\infty \\) thermodynamic properties of all \\( p_J[\mathbf{x}] \\) realizations are the same. This allows us to study only a single particular realization of \\( p_J[\mathbf{x}] \\) for sufficiently large \\( n \\).

Now let us turn to the code, `gmrs` library contains an implementation of a factor graph for Ising type models. Each factor in this graph is an exponent of the following kind \\[ \psi_{J, b_1, b_2}(x_1, x_2) = \exp(Jx_1x_2 + b_1 x_1 + b_2 x_2),\\] where \\( J \\) is a coupling constant, \\( b_1 \\) is a magnetic field acting on the first spin and \\( b_2 \\) is a magnetic field acting on the second field. One can build a realization of the SK model probability function from these factors as follows: \\[ \prod_{i, j=1, i>j}^n \psi_{J_{ij}, \frac{B}{n-1},\frac{B}{n-1}}(x_i, x_j),\\] where \\( J_{ij} \\) is sampled using a random numbers generator. Let us implement this factor graph in code. First, let us bring to the scope all the necessary objects:
```rust
use gmrs::ising::{new_ising_builder, random_message_initializer, IsingFactor, SumProduct};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
```
Here `new_ising_builder` is a function returning builder object for an Ising factor graph, `random_message_initializer` is a function that creates an 'initializer' for initial values of messages, `IsingFactor` is an Ising factor data type and `SumProduct` is a message passing type. All other object are necessary for generating couplings and initial values of messages at random.

Next, we define parameters of a problem:
```rust
// Number of spins in SK model
let spins_number = 2000;
// magnetic field per spin
let magnetic_field = 1.789;
// Error threshold specifying message passing stopping criterion
let error = 1e-10f64;
// Decay parameter (a hyper parameter of a sum-product algorithm)
let decay = 0.5;
// mean value of coupling constants
let mu = 0f64;
// std of coupling constants
let std = 1f64 / (f64::sqrt(2f64) * 2f64 * spins_number as f64);
// distribution of coupling constants
let distr = Normal::new(mu, std).unwrap();
// this generator is used to sample coupling constants
let mut rng_couplings = thread_rng();
// messages initializer
let mut initializer = random_message_initializer(thread_rng());
// maximal number of message passing iterations
    let max_iter = 10000;
```
All the parameters are self-explanatory, except the `decay` parameter that we explain later in the text and `initializer` that is an object used to initialize values of messages.

Now we are ready to build a factor graph:
```rust
let mut fgb =
    new_ising_builder::<SumProduct>(spins_number, (spins_number - 1) * spins_number / 2);
for i in 0..spins_number {
    for j in (i + 1)..spins_number {
        fgb.add_factor(
            IsingFactor::new(
                distr.sample(&mut rng_couplings),
                magnetic_field / ((spins_number - 1) as f64),
                magnetic_field / ((spins_number - 1) as f64),
            ),
            &[i, j],
            &mut initializer,
        )
        .unwrap();
    }
}
let mut fg = fgb.build();
```
Here first we create a factor graph with `spins_number` variables and memory allocated for `(spins_number - 1) * spins_number / 2` factors. Then, in two `for` loops we connect each two variables by an Ising factor with random coupling constant, one can check that this loop is in one-to-one correspondence with the formula for probability \\( \prod_{i, j=1, i>j}^n \psi_{J_{ij}, \frac{B}{n-1},\frac{B}{n-1}}(x_i, x_j) \\). Finally, we create (`.build()`) a factor graph from a builder.

Now we are ready to run the message passing algorithm and print its convergence info:
```rust
let info = fg
    .run_message_passing_parallel(max_iter, error, decay)
    .unwrap();
println!("{}", info);
```
Before we go further, let us discuss what happens under the hood. `.run_message_passing_parallel` method iterates a standard sum-product update rules:
\\[ \nu_{j\to a}^{t+1}(x_j) \propto \prod_{b\in \partial j / a}\hat{\nu}_{b\to j}^t(x_j)\\]

\\[ \hat{\nu}_{a\to j}^{t+1}(x_j) \propto \left[\sum_{x_k} \psi_{J_{jk}, \frac{B}{n-1},\frac{B}{n-1}}(x_j, x_k) \nu^t_{k\to a}(x_k)\right]^{1-\gamma}\left[\hat{\nu}_{a\to j}^{t}(x_j)\right]^\gamma, \\]


where \\( \gamma \\) is the dumping parameter introduced in the code above, it stabilizes algorithm by smoothing the update. This trick with damping sometimes also called _exponential moving average_.

After the convergence of the message passing algorithm, we can start extracting information about the system in the thermodynamic equilibrium. The simplest quantity that we can compute is the average spin over all the system's spins. The exact value of the average spin also averaged over all realizations is simple \\( \langle s \rangle = 1\cdot p_\uparrow + (-1)\cdot(1 - p_\uparrow) = 2p_\uparrow - 1 \\), where \\( p_\uparrow = {\rm tanh}(B) \\) is the average probability to have a spin in the up position. Converged messages allow us to calculate all the marginal distributions of all spins. In particular, a method `.eval_marginals` of a factor graph computes marginal probability of a spin being in the up state for all spins. Averaging this probability across all the spins and bearing in mind the self-averaging argument, we get an estimate of \\( p_\uparrow \\) and an estimate of \\( \langle s \rangle \\). The corresponding code is given below:
```rust
// All marginal probabilities of a spin being in up position
let marginals = fg.eval_marginals(); 
// Averaged over all spins probability to be in the up position
let p_up_approx = marginals.iter().map(|x| *x).sum::<f64>() / spins_number as f64;
// Exact up probability
let p_up_exact = f64::tanh(magnetic_field);
println!(
    "Approximate average spin: {}",
    2f64 * p_up_approx - 1f64,
);
println!(
    "Exact average spin: {}",
    2f64 * p_up_exact - 1f64,
);
```
This code prints the exact and estimated values of \\( \langle s \rangle \\), one can see, that those values are very close.

!!!To be continued!!!