import subprocess
from dataclasses import dataclass
import yaml

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