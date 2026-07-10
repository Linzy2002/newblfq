"""Command-line entry point for one BLFQ Hamiltonian calculation."""

from __future__ import annotations

import argparse

from .solver import BLFQParameters, solve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and diagonalise the truncated BLFQ Hamiltonian."
    )
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--mass-g", type=float, default=0.0)
    parser.add_argument("--mass-gg", type=float, default=0.0)
    parser.add_argument("--p-plus", type=float, default=None)
    parser.add_argument("--eigenvalues", type=int, default=3)
    args = parser.parse_args()

    result = solve(
        BLFQParameters(
            nmax=args.nmax,
            kmax=args.kmax,
            coupling=args.coupling,
            b=args.b,
            mass_g=args.mass_g,
            mass_gg=args.mass_gg,
            p_plus=args.p_plus,
        ),
        eigenvalues=args.eigenvalues,
    )
    print(f"dimension={result.dimension}, nnz={result.nnz}")
    print(f"hermiticity_error={result.hermiticity_error:.3e}")
    for index, value in enumerate(result.eigenvalues):
        print(f"E[{index}] = {value:.12e}")


if __name__ == "__main__":
    main()
