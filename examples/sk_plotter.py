#!/usr/bin/env python3

import os
import sys
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from utils import run_sk, ExampleResult

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
    axs[2].plot(beta, bethe_free_entropy, 'ro', label="Bethe free entropy (sum-product)")
    axs[2].plot(beta, replica_symmetric_free_entropy, 'b-', label="RS free entropy")
    axs[2].axvline(1., color='k')
    axs[2].set_xlabel("inverse temperature")
    axs[2].legend()
    plt.savefig(f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/sk_plots.svg", bbox_inches="tight")

if __name__ == "__main__":
    main()