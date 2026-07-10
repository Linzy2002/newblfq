# BLFQ Hamiltonian workflow

This directory provides a single-calculation interface, separate from the
renormalisation scan in `main.py`.  It uses the repository's existing basis
enumeration and Hamiltonian kernels, so its matrix is identical to the legacy
construction for the same parameters.

From the repository root, run:

```bash
python3 -m blfq_agent --nmax 4 --kmax 4 --coupling 1.0 --b 1.0
```

For use in a script:

```python
from blfq_agent import BLFQParameters, solve

result = solve(BLFQParameters(nmax=4, kmax=4), eigenvalues=3)
print(result.eigenvalues)
```

`p_plus` defaults to `kmax`; pass it explicitly when a different total
longitudinal momentum convention is required.  The solver checks Hermiticity
before diagonalising and uses a dense solver for small cutoffs and `eigsh` for
larger ones.
