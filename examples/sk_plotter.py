#!/usr/bin/env python3

import os
import sys
import subprocess
from dataclasses import dataclass
import yaml
import matplotlib.pyplot as plt  # type: ignore
import numpy as np


@dataclass
class ExampleResult:
    is_converged: bool
    iterations_number: int
    discrepancy: float
    bethe_free_entropy: float
    replica_symmetric_free_entropy: float

"""Runs sk.rs file with its command line parameters passed
as function arguments"""
def run_sk(
        beta: float,
        spins_number: int = 1000,
        max_iter: int = 1000,
        threshold: float = 1e-6,
        decay: float = 0.5,
) -> ExampleResult:
    proc = subprocess.run(
        f"cargo run --release --example sk -- \
         --beta {beta} \
         --spins-number {spins_number} \
         --max-iter {max_iter} \
         --threshold {threshold} \
         --decay {decay}"
        ,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=True,
        check=True,
    )
    result_dict = yaml.safe_load(proc.stdout)
    result = ExampleResult(
        result_dict["is_converged"],
        result_dict["iterations_number"],
        result_dict["discrepancy"],
        result_dict["bethe_free_entropy"],
        result_dict["replica_symmetric_free_entropy"],
    )
    return result

# -------------------------------------------------------------------------------------------

def main():
    betas = np.arange(0., 1.2, 0.05)
    def result_parser(result: ExampleResult):
        print(f"{result}")
        return result.iterations_number, \
            result.discrepancy, \
            result.bethe_free_entropy, \
            result.replica_symmetric_free_entropy,
    beta, iterations_number, discrepancy, bethe_free_entropy, replica_symmetric_free_entropy = \
        zip(*[(beta, *result_parser(run_sk(beta))) for beta in betas])
    fig, axs = plt.subplots(3, 1)
    fig.tight_layout(pad=1.0)
    axs[0].plot(beta, iterations_number, 'b')
    axs[0].axvline(1., color='k')
    axs[0].set_ylabel("iter. number")
    axs[1].plot(beta, discrepancy, 'b')
    axs[1].axvline(1., color='k')
    axs[1].set_ylabel("discrepancy")
    axs[2].plot(beta, bethe_free_entropy, 'ro', label="Bethe free energy (sum-product)")
    axs[2].plot(beta, replica_symmetric_free_entropy, 'b-', label="RS free energy")
    axs[2].axvline(1., color='k')
    axs[2].set_xlabel("inverse temperature")
    axs[2].legend()
    plt.savefig(f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/plots.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()