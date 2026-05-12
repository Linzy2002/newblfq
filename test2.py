from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from math import log, sqrt, exp, floor, ceil
from scipy.sparse.linalg import eigsh

from TMC import TMC


# =========================================================
# 1. Physics parameters 
# =========================================================

@dataclass(frozen=True)
class PhysicsParams:
    couplings: float
    b: float
    mass_g: float
    mass_gg: float


# =========================================================
# 2. Basis states
# =========================================================

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


# =========================================================
# 3. 1-particle basis (g sector)
# =========================================================

def generate_1p_basis(Nmax, Mj, K):

    basis = []

    s = 1 if Mj >= 0 else -1

    n = 0
    m = 0

    if n + abs(m) + 1 <= Nmax:

        basis.append(
            SingleParticleState(
                k=K,
                s=s,
                n=n,
                m=m
            )
        )

    return basis


# =========================================================
# 4. 2-particle basis (gg sector)
# =========================================================

# def generate_gg_basis(Nmax, Mj, K, color_states):

#     basis = []

#     for color_state in range(color_states):

#         for k1 in range(1, K):
#             k2 = K - k1

#             for s1 in (-1, +1):
#                 for s2 in (-1, +1):

#                     target_m = Mj // 2 - s1 - s2
#                     max_n = (Nmax - 2) // 2

#                     for n1 in range(max_n + 1):
#                         for n2 in range(max_n - n1 + 1):

#                             m_limit = Nmax

#                             for m1 in range(-m_limit, m_limit + 1):

#                                 m2 = target_m - m1

#                                 N1 = 2 * n1 + abs(m1) + 1
#                                 N2 = 2 * n2 + abs(m2) + 1

#                                 if N1 + N2 > Nmax:
#                                     continue

#                                 g1 = SingleParticleState(k1, s1, n1, m1)
#                                 g2 = SingleParticleState(k2, s2, n2, m2)

#                                 basis.append(
#                                     TwoParticleState(
#                                         particles=(g1, g2),
#                                         color_state=color_state
#                                     )
#                                 )

#     return basis

def generate_gg_basis(Nmax, Mj, K, color_states):

    basis = []

    kmax = K - 1

    for color_state in range(color_states):

        # Fortran: do k1=1,kmax
        for k1 in range(1, kmax + 1):

            k2 = K - k1

            # Fortran:
            # do s2index=-1,1,2
            # do s1index=-1,1,2
            for s2 in (-1, +1):
                for s1 in (-1, +1):

                    target_m = Mj // 2 - s1 - s2

                    max_n = Nmax // 2

                    # Fortran:
                    # do n2
                    # do n1
                    for n2 in range(max_n + 1):
                        for n1 in range(max_n - n2 + 1):

                            # 完全照搬 Fortran 的 m1 上下限
                            # lower = -(
                            #     (Nmax - 2 - 2*n1 - 2*n2 - target_m) // 2
                            # )

                            # upper = (
                            #     (Nmax - 2 - 2*n1 - 2*n2 + target_m) // 2
                            # )

                            lower = -floor(
                                (Nmax - 2 - 2*n1 - 2*n2 - target_m) / 2
                            )

                            upper = ceil(
                                (Nmax - 2 - 2*n1 - 2*n2 + target_m) / 2
                            )

                            for m1 in range(lower, upper + 1):

                                m2 = target_m - m1

                                N1 = 2 * n1 + abs(m1) + 1
                                N2 = 2 * n2 + abs(m2) + 1

                                if N1 + N2 > Nmax:
                                    continue

                                g1 = SingleParticleState(k1, s1, n1, m1)
                                g2 = SingleParticleState(k2, s2, n2, m2)

                                basis.append(
                                    TwoParticleState(
                                        particles=(g1, g2),
                                        color_state=color_state
                                    )
                                )

    return basis


# =========================================================
# 5. Global basis
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
# 6. Sparse Hamiltonian builder
# =========================================================

def build_sparse_hamiltonian(global_basis, params):

    H = defaultdict(float)

    dim = len(global_basis)

    for i, (si, bra) in enumerate(global_basis):
        for j, (sj, ket) in enumerate(global_basis):

            val = matrix_element(si, sj, bra, ket, params)

            if abs(val) > 1e-14:
                H[(i, j)] += val

    return H, dim


# =========================================================
# 7. tool
# =========================================================

def kronecker(i, j):
    return 1 if i == j else 0


def fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1):
    return (kronecker(np1, nk1) *
            kronecker(mp1, mk1) *
            kronecker(sp1, sk1) *
            kronecker(kp1, kk1))


def adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1):
    if kronecker(mk1, mp1) == 0:
        return 0.0

    term1 = (1 + 2*np1 + abs(mp1)) * kronecker(nk1, np1)
    term2 = 0.0

    if max(nk1, np1) - min(nk1, np1) == 1:
        term2 = np.sqrt(max(nk1, np1) * (abs(mp1) + max(nk1, np1)))

    return (term1 - term2) * kronecker(sp1, sk1) * kronecker(kp1, kk1)

def adotas(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1):

    # KroneckerDelta(np1, nk1)
    term1 = (1 if np1 == nk1 else 0) * (2 * np1 + abs(mp1) + 1)

    # KroneckerDelta(|np1-nk1|, 1)
    term2 = 0.0
    if abs(np1 - nk1) == 1:
        term2 = np.sqrt(max(np1, nk1) * (max(np1, nk1) + abs(mp1)))

    # full structure
    result = (term1 + term2)

    result *= (1 if mp1 == mk1 else 0)
    result *= (1 if sp1 == sk1 else 0)
    result *= (1 if kp1 == kk1 else 0)

    return float(result)

def ifactor(n1, m1, n2, m2):

    def d(a, b):
        return 1 if a == b else 0

    # -----------------------------
    # Case 1: m1 >= 0 and m2 = m1 + 1
    # -----------------------------
    if (m1 >= 0) and (m2 == m1 + 1):

        term1 = -np.sqrt(float(n1)) * d(n1 - 1, n2)
        term2 = np.sqrt(float(1 + m1 + n1)) * d(n1, n2)

        return term1 + term2

    # -----------------------------
    # Case 2: m1 > 0 and m2 = m1 - 1
    # -----------------------------
    elif (m1 > 0) and (m2 == m1 - 1):

        term1 = np.sqrt(float(m1 + n1)) * d(n1, n2)
        term2 = -np.sqrt(float(1 + n1)) * d(n1 + 1, n2)

        return term1 + term2

    # -----------------------------
    # Case 3: m1 < 0 and m2 = m1 + 1
    # -----------------------------
    elif (m1 < 0) and (m2 == m1 + 1):

        term1 = np.sqrt(float(-m1 + n1)) * d(n1, n2)
        term2 = -np.sqrt(float(1 + n1)) * d(n1 + 1, n2)

        return term1 + term2

    # -----------------------------
    # Case 4: m1 <= 0 and m2 = m1 - 1
    # -----------------------------
    elif (m1 <= 0) and (m2 == m1 - 1):

        term1 = -np.sqrt(float(n1)) * d(n1 - 1, n2)
        term2 = np.sqrt(float(1 - m1 + n1)) * d(n1, n2)

        return term1 + term2

    # -----------------------------
    # default
    # -----------------------------
    else:
        return 0.0
    
def adotbq(np1, mp1, nk1, mk1,
           np2, mp2, nk2, mk2,
           sp1, sk1, sp2, sk2,
           kp1, kk1, kp2, kk2):

    def d(a, b):
        return 1 if a == b else 0

    # -------------------------
    # HO ladder factors
    # -------------------------
    term1 = ifactor(np1, mp1, nk1, mk1)
    term2 = ifactor(np2, mp2, nk2, mk2)

    # -------------------------
    # m coupling structure
    # -------------------------
    m_coupling = (
        d(mk1, 1 + mp1) * d(mk2, -1 + mp2)
        +
        d(mk1, -1 + mp1) * d(mk2, 1 + mp2)
    )

    # -------------------------
    # spin + longitudinal deltas
    # -------------------------
    spin = d(sp1, sk1) * d(sp2, sk2)
    kdelta = d(kp1, kk1) * d(kp2, kk2)

    # -------------------------
    # final expression (exact Fortran structure)
    # -------------------------
    return 0.5 * term1 * term2 * m_coupling * spin * kdelta


# =========================================================
# 7. Matrix element dispatcher
# =========================================================

def matrix_element(sector_i, sector_j, bra, ket, params):

    # -------------------------
    # g sector
    # -------------------------
    if sector_i == "g" and sector_j == "g":
        return H_g_g(bra, ket, params)

    # -------------------------
    # gg sector
    # -------------------------
    if sector_i == "gg" and sector_j == "gg":
        return H_gg_gg(bra, ket, params)

    # -------------------------
    # interaction (Hermitian)
    # -------------------------
    if sector_i == "g" and sector_j == "gg":
        return H_g_gg(bra, ket, params)

    if sector_i == "gg" and sector_j == "g":
        return H_g_gg(ket, bra, params)

    return 0.0


# =========================================================
# 8. Physics kernels
# =========================================================
def H_g_g(bra, ket, params):

    kp1 = bra.k
    sp1 = bra.s
    np1 = bra.n
    mp1 = bra.m
    
    
    kk1 = ket.k
    sk1 = ket.s
    nk1 = ket.n
    mk1 = ket.m

    # --------- selection rule (diagonal structure) ----------
    if not (kp1 == kk1 and sp1 == sk1 and np1 == nk1 and mp1 == mk1):
        return 0.0

    # --------- parameters ----------
    mass = params.mass_g
    b = params.b
    lag = 30.1

    Pplus = 6.0

    kt = kp1
    mj = 2
    
    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)


    kp1half = kp1 
    Pplus = kp1half  # 如果你有全局 Pplus，这里可以替换

    # --------- core matrix elements ----------
    Ememe = fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1)
    Ep1p1 = adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1)

    shift = 2 * Ememe

    kinetic1 = (mass**2 * Ememe +
                bee**2 * (kp1half / Pplus) * Ep1p1) / kp1half

    EnergySingle = kinetic1
    EnergyCM = bee**2 * (Ep1p1 * kp1half / Pplus) / Pplus

    fourierphasen = -np1 + nk1
    fourierphasem = -abs(mp1) + abs(mk1)

    fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)
    

    lagrangeterm = lag * bee**2 * (
        Ep1p1 * kp1half / Pplus +
        (Ep1p1 * kp1half / Pplus) * fourierphase -
        shift
    ) / Pplus

    hamiltonian = EnergySingle - EnergyCM + lagrangeterm

    return hamiltonian


def H_gg_gg(bra, ket, params):
    """
    two-gluon kinetic term (BLFQ-style)
    """


    g3, g4 = bra.particles

    g1, g2 = ket.particles

    kp1 = g3.k
    sp1 = g3.s
    np1 = g3.n
    mp1 = g3.m
    
    kp2 = g4.k
    sp2 = g4.s
    np2 = g4.n
    mp2 = g4.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m
    
    
    initialcolor = bra.color_state
    finalcolor   = ket.color_state

    if not (
        kp1 == kk1 and kp2 == kk2 and
        sp1 == sk1 and sp2 == sk2 
    ):
        return 0.0

    if initialcolor != finalcolor:
        return 0.0

    # -----------------------------
    # parameters
    # -----------------------------
    mass1 = params.mass_gg
    mass2 = params.mass_gg
    b = params.b
    
    lag = 30.1



    # -----------------------------
    # longitudinal momenta
    # -----------------------------
    kp1half = kp1 
    kp2half = kp2 
    kt = kp1 + kp2
    mj = 2 


##!!!!!!!!!!!!!!!!!!!!!!!总动量， 可能要调整

    Pplus = 6.0
    
    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)
    
    Pplus = kt
    
    

    # -----------------------------
    # Fourier phase (DIRECT translation)
    # -----------------------------
    # fourierphasen = -np1 - np2 + nk1 + nk2
    # fourierphasem = -abs(mp1) - abs(mp2) + abs(mk1) + abs(mk2)

    # fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    # -----------------------------
    # delta / matrix elements
    # -----------------------------
    Ememe = (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    Ep1p1 = (adotaq(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    Ep2p2 = (adotaq(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Es1s1 = (adotas(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
             fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    Es2s2 = (adotas(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2) *
             fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1))

    Ep1p2 = adotbq(np1, mp1, nk1, mk1,
                   np2, mp2, nk2, mk2,
                   sp1, sk1, sp2, sk2,
                   kp1, kk1, kp2, kk2)

    shift = 2.0 * (fidelta(np1, mp1, nk1, mk1, sp1, sk1, kp1, kk1) *
                   fidelta(np2, mp2, nk2, mk2, sp2, sk2, kp2, kk2))

    # -----------------------------
    # kinetic energy term
    # -----------------------------
    kinetic1 = ((mass1 ** 2) * Ememe +
                bee ** 2 * (kp1half / Pplus) * Ep1p1) / kp1half

    kinetic2 = ((mass2 ** 2) * Ememe +
                bee ** 2 * (kp2half / Pplus) * Ep2p2) / kp2half

    EnergySingle = kinetic1 + kinetic2

    EnergyCM = bee ** 2 * (
        Ep1p1 * kp1half / Pplus +
        2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
        Ep2p2 * kp2half / Pplus
    ) / Pplus

    lagrangeterm = lag * bee ** 2 * (
        (Ep1p1 * kp1half / Pplus +
         2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
         Ep2p2 * kp2half / Pplus)
        +
        (Ep1p1 * kp1half / Pplus +
         2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
         Ep2p2 * kp2half / Pplus) 
        - shift
    ) / Pplus
    
    # print("lag = ", lag, "b2 = ", bee**2, "Pplus = ", Pplus)
    





    # -----------------------------
    # Hamiltonian
    # -----------------------------
    hamiltonian = (
        EnergySingle
        - EnergyCM
        + lagrangeterm
    )
    
#     print("EnergySingle = ", EnergySingle, "EnergyCM = ", EnergyCM, "lagrangeterm = ", lagrangeterm)

    return hamiltonian


def H_g_gg(bra, ket, params):

    # =====================================================
    # enforce interpretation:
    # bra = g
    # ket = gg
    # =====================================================

    g_state = bra
    g1, g2 = ket.particles

    # -------------------------
    # unpack
    # -------------------------
    kp1 = g_state.k
    sp1 = g_state.s
    np1 = g_state.n
    mp1 = g_state.m

    kk1 = g1.k
    sk1 = g1.s
    nk1 = g1.n
    mk1 = g1.m

    kk2 = g2.k
    sk2 = g2.s
    nk2 = g2.n
    mk2 = g2.m


    finalcolor   = ket.color_state

    if finalcolor == 0:
        CF = -np.sqrt(3.0)

    elif finalcolor == 1:
        return 0.0

    else:
        raise ValueError(f"Unknown finalcolor = {finalcolor}")
    

    # -------------------------
    # params
    # -------------------------
    couplings = params.couplings
    b = params.b


    Pplus = 6.0

    kt = kp1

    mj = 2
    
    bee = b * np.sqrt((float(kt) + (mj % 2) / 2.0) / Pplus)

    Pplus = kp1 

    
    coupling_eff = couplings * CF / 2.0

    # -------------------------
    # selection rule
    # -------------------------
    if kp1 != kk1 + kk2:
        return 0.0

    # -------------------------
    # quantum number structure
    # -------------------------
    m = sp1 - (sk1 + sk2)

    n = nk1 + nk2 - np1 + (abs(mk1) + abs(mk2) - abs(mp1) - abs(m)) // 2
    
    if n < 0:
        return 0.0
    
    
    kp1h = float(kp1)
    kk1h = float(kk1)
    kk2h = float(kk2)

    constant = sqrt(2.0) * bee**2 * coupling_eff / (np.pi * Pplus)

    tandelta = sqrt(kk2h / kk1h)

    spinor = sqrt(n + 1.0) * ((-1.0) ** n)

    # -------------------------
    # TMC
    # -------------------------
    T = TMC(
        np1, mp1,
        n, m,
        nk1, mk1,
        nk2, mk2,
        tandelta
    )

    # -------------------------
    # terms
    # -------------------------
    c1 = c2 = c3 = 0.0

    if sp1 == sk2:
        longipart1 = sqrt(kk2h / (kk1h * kp1h))
        c1 = -longipart1 * T * spinor

    if sk1 == -sk2:
        longipart2 = sqrt(kk2h * kk1h / kp1h) / kp1h
        c2 = longipart2 * T * spinor

    if sp1 == sk1:
        longipart3 = sqrt(kk1h / (kk2h * kp1h))
        c3 = -longipart3 * T * spinor

    # -------------------------
    # final vertex
    # -------------------------
    interaction = constant * (c1 + c2 + c3)


    return interaction 


# =========================================================
# 9. sparse conversion
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


# =========================================================
# 10. FULL BUILD FUNCTION (FINAL API)
# =========================================================

def dump_gg_basis(
    basis,
    filename="Output/gg_basis.dat"
):
    """
    输出 gg basis 到 dat 文件
    """

    with open(filename, "w") as f:

        f.write(
            "# idx  color   "
            "k1 k2 s1 s2    "
            "n1 m1 n2 m2\n"
        )

        for idx, state in enumerate(basis):

            g1, g2 = state.particles

            f.write(
                f"{idx:6d} "
                f"{state.color_state:6d}   "
                f"{g1.k:3d} {g2.k:3d} {g1.s:3d} {g2.s:4d}   "
                f"{g1.n:3d} {g1.m:3d} {g2.n:3d} {g2.m:4d}\n"
            )

    print(f"gg basis saved to {filename}")


def build_hamiltonian(Nmax, K, params: PhysicsParams):

    Mj = 2
    color_states = 2

    # -------------------------
    # basis
    # -------------------------
    g_basis = generate_1p_basis(Nmax, Mj, K)

    gg_basis = generate_gg_basis(Nmax, Mj, K, color_states)

    # dump_gg_basis(
    #     gg_basis,
    #     "Output/gg_basis.dat"
    # )

    sector_basis = {
        "g": g_basis,
        "gg": gg_basis
    }

    # -------------------------
    # global basis
    # -------------------------
    global_basis, index_map = build_global_basis(sector_basis)

    # bra_sector, bra = global_basis[5]
    # ket_sector, ket = global_basis[6]

    # a = H_gg_gg(bra, ket, params)

    # print(a)


    # -------------------------
    # Hamiltonian
    # -------------------------
    H_dict, dim = build_sparse_hamiltonian(global_basis, params)

    H = to_sparse_matrix(H_dict, dim)
    # H=0

    return H, global_basis, index_map


def dump_hamiltonian_dense(H, filename="hamiltonian.dat"):
    """
    将稀疏/稠密 Hamiltonian 输出为完整方阵 dat 文件
    """

    # 转成 dense（关键一步）
    H_dense = H.toarray()

    dim = H_dense.shape[0]

    with open(filename, "w") as f:
        # 写维度
        f.write(f"# dimension = {dim}\n")

        for i in range(dim):
            for j in range(dim):
                val = H_dense[i, j]
                # 写成科学计数法，方便 Fortran / Python 读
                f.write(f"{i:6d} {j:6d} {val.real:20.12e} \n")

    print(f"saved to {filename}")




# =========================================================
# 11. Example
# =========================================================

if __name__ == "__main__":

    params = PhysicsParams(
        couplings=5.5,
        b=0.83,
        mass_g=0.0,
        mass_gg=0.0
    )

    H, basis, index_map = build_hamiltonian(
        Nmax=4,
        K=5,
        params=params
    )

    print("dim =", H.shape[0])
    print("nnz =", H.nnz)
    
        # 求最小的几个本征值
    vals, vecs = eigsh(
        H,
        k=3,          # 取最低6个态
        which='SA'    # Smallest Algebraic（最小本征值）
    )

    print("lowest eigenvalues:")
    print(vals)
    
    dump_hamiltonian_dense(H, "Output/h.dat")

    #     kt = 5

#     params = PhysicsParams(
#         couplings=5.5,
#         b=0.83,
#         mass_g=0.0,
#         mass_gg=0.0
#     )

#     H = build_hamiltonian(
#         Nmax=4,
#         K=kt,
#         params=params
#     )

#     print("dim =", H.shape[0])
#     print("nnz =", H.nnz)
    
#         # 求最小的几个本征值
#     vals, vecs = eigsh(H,k=3,which='SA')

#     eigenv1 = vals[0] * kt

#     renomass2 = sqrt(-1*eigenv1)

#     inputmass = renomass2

#     params = PhysicsParams(couplings=5.5,b=0.83,mass_g=inputmass,mass_gg=0.0)
#     H = build_hamiltonian(Nmax=4,K=kt,params=params)
#     vals, vecs = eigsh(H,k=3,which='SA')

#     eigenv2 = vals[0] * kt

#     loopnumber = 0
#     renomass1 = 0.0

#     loopnumbermax = 10

#     for loopnumber in range(loopnumbermax + 1):

#         # -------------------------
#         # 收敛判据
#         # -------------------------
#         if abs(eigenv2) < 1.0e-10:
#             break

#         # -------------------------
#         # 更新 renorm mass
#         # -------------------------
#         # renormass3 = np.sqrt((renomass2**2 / (eigenv2 - eigenv1)) * (-eigenv1))

#         renomass3 = np.sqrt(renomass1**2 + (renomass2**2 - renomass1**2) / (eigenv2 - eigenv1) * (-eigenv1)) 

#         inputmass =  renomass3

#         params = PhysicsParams(couplings=5.5,b=0.83,mass_g=inputmass,mass_gg=0.0)
#         H = build_hamiltonian(Nmax=4,K=kt,params=params)
#         vals, vecs = eigsh(H,k=3,which='SA')

#         eigenv1 = eigenv2
#         eigenv2 = vals[0] * kt

#         renomass1 = renomass2
#         renomass2 = renomass3

# print("renomass2 = ", renomass2)

