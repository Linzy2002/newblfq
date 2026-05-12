from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix



@dataclass(frozen=True)
class SingleParticleState:

    k: int
    s: int
    n: int
    m: int


@dataclass(frozen=True)
class TwoParticleState:

    particles: tuple

    color_state: int


def generate_1p_basis(
    Nmax,
    Mj,
    K,
):

    basis = []

    s = 1 if Mj >= 0 else -1

    n = 0
    m = 0

    if n + abs(m) + 1 <= Nmax:

        state = SingleParticleState(
            k=K,
            s=s,
            n=n,
            m=m,
        )

        basis.append(state)

    return basis


def generate_gg_basis(
    Nmax,
    Mj,
    K,
    color_states
):

    basis = []

    for color_state in range(color_states):

        # k1 + k2 = K
        # gluon minimum k = 1

        for k1 in range(1, K):

            k2 = K - k1

            for s1 in (-1, +1):
                for s2 in (-1, +1):

                    # target_m = (Mj - s1 - s2) // 2
                    target_m = Mj // 2 - s1 - s2

                    max_n = (Nmax - 2) // 2

                    for n1 in range(max_n + 1):
                        for n2 in range(max_n - n1 + 1):

                            m_limit = Nmax

                            for m1 in range(-m_limit, m_limit + 1):

                                m2 = target_m - m1

                                N1 = 2*n1 + abs(m1) + 1
                                N2 = 2*n2 + abs(m2) + 1

                                if N1 + N2 > Nmax:
                                    continue

                                g1 = SingleParticleState(
                                    k=k1,
                                    s=s1,
                                    n=n1,
                                    m=m1
                                )

                                g2 = SingleParticleState(
                                    k=k2,
                                    s=s2,
                                    n=n2,
                                    m=m2
                                )

                                state = TwoParticleState(
                                    particles=(g1, g2),
                                    color_state=color_state
                                )

                                basis.append(state)

    return basis


# basis = generate_1p_basis(
#     Nmax=5,
#     Mj=2,
#     K=4
# )

# print(len(basis))


# basis = generate_gg_basis(
#     Nmax=5,
#     Mj=2,
#     K=4,
#     color_states=2
# )

# print("number of basis =", len(basis))
# print()

# for ibasis, state in enumerate(basis):

#     print(f"basis {ibasis}")

#     for i, p in enumerate(state.particles, start=1):

#         print(
#             f"particle {i}: "
#             f"k = {p.k:>3}  "
#             f"s = {p.s:>3}  "
#             f"n = {p.n:>3}  "
#             f"m = {p.m:>3}"
#         )

#     print(f"color_state = {state.color_state}")
#     print()


# =========================================================
# 4. Global basis (sector union)
# =========================================================

def build_global_basis(sector_basis):

    global_basis = []
    index_map = {}

    idx = 0

    for sector, basis in sector_basis.items():
        for state in basis:
            global_basis.append((sector, state))
            index_map[(sector, state)] = idx
            idx += 1

    return global_basis, index_map


# =========================================================
# 5. Hamiltonian (sparse dict form)
# =========================================================

def build_sparse_hamiltonian(global_basis):

    H = defaultdict(float)

    dim = len(global_basis)

    for i, (si, bra) in enumerate(global_basis):
        for j, (sj, ket) in enumerate(global_basis):

            val = matrix_element(si, sj, bra, ket)

            if abs(val) > 1e-14:
                H[(i, j)] += val

    return H, dim


# =========================================================
# 6. Interaction dispatcher (physics goes here)
# =========================================================

def matrix_element(sector_i, sector_j, bra, ket):

    # -------------------------
    # diagonal g sector
    # -------------------------
    if sector_i == "g" and sector_j == "g":
        return H_g_g(bra, ket)

    # -------------------------
    # diagonal gg sector
    # -------------------------
    if sector_i == "gg" and sector_j == "gg":
        return H_gg_gg(bra, ket)

    # -------------------------
    # interaction (ONLY compute one direction)
    # -------------------------
    if sector_i == "g" and sector_j == "gg":
        return H_g_gg(bra, ket)

    # Hermitian conjugate
    if sector_i == "gg" and sector_j == "g":
        return H_g_gg(ket, bra)

    return 0.0

# =========================================================
# 7. Physics kernels (PLACEHOLDERS)
# =========================================================

def H_g_g(bra, ket):
    return 0.0


def H_gg_gg(bra, ket):
    return 0.0

def H_g_gg(bra, ket):
    return 0.0



# =========================================================
# 8. Convert to scipy sparse matrix
# =========================================================

def to_sparse_matrix(H_dict, dim):

    rows = []
    cols = []
    data = []

    for (i, j), val in H_dict.items():
        rows.append(i)
        cols.append(j)
        data.append(val)

    return coo_matrix((data, (rows, cols)), shape=(dim, dim))


def build_hamiltonian(Nmax, K):

    # -----------------------------
    # fixed physics settings
    # -----------------------------
    Mj = 2
    color_states = 2

    # -----------------------------
    # basis generation
    # -----------------------------
    g_basis = generate_1p_basis(
        Nmax=Nmax,
        Mj=Mj,
        K=K
    )

    gg_basis = generate_gg_basis(
        Nmax=Nmax,
        Mj=Mj,
        K=K,
        color_states=color_states
    )

    sector_basis = {
        "g": g_basis,
        "gg": gg_basis
    }

    # -----------------------------
    # global basis
    # -----------------------------
    global_basis, index_map = build_global_basis(sector_basis)

    # -----------------------------
    # sparse Hamiltonian
    # -----------------------------
    H_dict, dim = build_sparse_hamiltonian(global_basis)

    H = to_sparse_matrix(H_dict, dim)

    return H, global_basis, index_map


# =========================================================
# 9. Example usage
# =========================================================

if __name__ == "__main__":

    H, basis, index_map = build_hamiltonian(
        Nmax=5,
        K=4
    )

    print("dim =", H.shape[0])
    print("nnz =", H.nnz)