#!/usr/bin/env python3

import numpy as np
from utils import run_sk

def main():
    print("Testing SK example...")
    result = run_sk(0.8)
    print(f"Bethe free energy: {result.bethe_free_entropy}, RS free energy: {result.replica_symmetric_free_entropy}")
    assert np.abs(result.bethe_free_entropy - result.replica_symmetric_free_entropy) < 3e-3
    assert result.is_converged
    print("OK")

if __name__ == "__main__":
    main()