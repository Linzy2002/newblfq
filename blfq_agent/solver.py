"""Build and diagonalise the truncated one-gluon/two-gluon BLFQ Hamiltonian.

The established basis and matrix-element kernels live at the repository root.
This module deliberately calls those kernels unchanged, while exposing the
construction and eigensolver as a small, reusable workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from Ctool import to_sparse_matrix
from basis import build_global_basis, generate_1p_basis, generate_gg_basis
from hami import build_sparse_hamiltonian


@dataclass(frozen=True)
class BLFQParameters:
    """Input parameters for a single truncated Hamiltonian calculation."""

    nmax: int
    kmax: int
    coupling: float = 1.0
    b: float = 1.0
    mass_g: float = 0.0
    mass_gg: float = 0.0
    p_plus: float | None = None
    mj: int = 2
    color_states: int = 2

    def __post_init__(self) -> None:
        if self.nmax < 2:
            raise ValueError("nmax must be at least 2 for the gg sector")
        if self.kmax < 2:
            raise ValueError("kmax must be at least 2 for the gg sector")
        if self.color_states < 1:
            raise ValueError("color_states must be positive")
        if self.p_plus is not None and self.p_plus <= 0:
            raise ValueError("p_plus must be positive")
        if self.p_plus is None:
            # hami.py accesses ``params.p_plus`` directly.
            object.__setattr__(self, "p_plus", float(self.kmax))

    @property
    def resolved_p_plus(self) -> float:
        """Use K as P+ unless an explicit total longitudinal momentum is given."""
        return self.p_plus

    # The legacy kernels consume this attribute-only parameter object.
    @property
    def couplings(self) -> float:
        return self.coupling


@dataclass(frozen=True)
class DiagonalizationResult:
    """Hamiltonian spectrum and metadata for one BLFQ truncation."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    dimension: int
    nnz: int
    hermiticity_error: float


def build_hamiltonian(params: BLFQParameters) -> tuple[csr_matrix, list[tuple[str, object]]]:
    """Construct the global-sector sparse Hamiltonian for ``params``."""
    g_basis = generate_1p_basis(params.nmax, params.mj, params.kmax)
    gg_basis = generate_gg_basis(
        params.nmax, params.mj, params.kmax, params.color_states
    )
    global_basis, _ = build_global_basis({"g": g_basis, "gg": gg_basis})
    entries, dimension = build_sparse_hamiltonian(global_basis, params)
    hamiltonian = to_sparse_matrix(entries, dimension).tocsr()
    return hamiltonian, global_basis


def solve(
    params: BLFQParameters,
    eigenvalues: int = 3,
    dense_threshold: int = 64,
) -> DiagonalizationResult:
    """Build the Hamiltonian and return its lowest algebraic eigenpairs.

    Small matrices use a dense Hermitian solver; larger matrices use ARPACK.
    This avoids the ``k < dimension`` restriction of ``eigsh`` at low cutoffs.
    """
    hamiltonian, _ = build_hamiltonian(params)
    dimension = hamiltonian.shape[0]
    if not 1 <= eigenvalues <= dimension:
        raise ValueError(f"eigenvalues must be in [1, {dimension}]")

    hermiticity_error = float(
        np.max(np.abs((hamiltonian - hamiltonian.getH()).data), initial=0.0)
    )
    if hermiticity_error > 1e-10:
        raise ValueError(
            f"Hamiltonian is not Hermitian (max difference {hermiticity_error:.3e})"
        )

    if dimension <= dense_threshold or eigenvalues == dimension:
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        values, vectors = values[:eigenvalues], vectors[:, :eigenvalues]
    else:
        values, vectors = eigsh(hamiltonian, k=eigenvalues, which="SA")
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]

    return DiagonalizationResult(
        eigenvalues=values,
        eigenvectors=vectors,
        dimension=dimension,
        nnz=hamiltonian.nnz,
        hermiticity_error=hermiticity_error,
    )
