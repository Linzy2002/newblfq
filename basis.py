from dataclasses import dataclass
from math import  floor, ceil
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
