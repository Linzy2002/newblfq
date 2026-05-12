from dataclasses import dataclass
from bisect import bisect_right
from math import floor, ceil


# ============================================================
# 数据结构
# ============================================================

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


# ============================================================
# 旧代码
# ============================================================


def generate_gg_basis_old(Nmax, Mj, K, color_states):

    basis = []

    kmax = K - 1

    for color_state in range(color_states):

        for k1 in range(1, kmax + 1):

            k2 = K - k1

            for s2 in (-1, +1):
                for s1 in (-1, +1):

                    target_m = Mj // 2 - s1 - s2

                    max_n = Nmax // 2

                    for n2 in range(max_n + 1):
                        for n1 in range(max_n - n2 + 1):

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


# ============================================================
# 新 decoder
# ============================================================

@dataclass
class BasisBlock:

    s1: int
    s2: int

    n1: int
    n2: int

    target_m: int

    valid_m1: list

    offset: int
    count: int


class GGBasis:

    def __init__(self, Nmax, Mj, K, color_states):

        self.Nmax = Nmax
        self.Mj = Mj
        self.K = K

        self.color_states = color_states

        self.kmax = K - 1
        self.max_n = Nmax // 2

        self.spin_values = (-1, +1)

        self.blocks = []

        offset = 0

        for s2 in self.spin_values:
            for s1 in self.spin_values:

                target_m = Mj // 2 - s1 - s2

                for n2 in range(self.max_n + 1):
                    for n1 in range(self.max_n - n2 + 1):

                        valid_m1 = []

                        lower = -floor(
                            (Nmax - 2 - 2*n1 - 2*n2 - target_m) / 2
                        )

                        upper = ceil(
                            (Nmax - 2 - 2*n1 - 2*n2 + target_m) / 2
                        )

                        for m1 in range(lower, upper + 1):

                            m2 = target_m - m1

                            N1 = 2*n1 + abs(m1) + 1
                            N2 = 2*n2 + abs(m2) + 1

                            if N1 + N2 > Nmax:
                                continue

                            valid_m1.append(m1)

                        count = len(valid_m1)

                        if count == 0:
                            continue

                        self.blocks.append(
                            BasisBlock(
                                s1=s1,
                                s2=s2,
                                n1=n1,
                                n2=n2,
                                target_m=target_m,
                                valid_m1=valid_m1,
                                offset=offset,
                                count=count,
                            )
                        )

                        offset += count

        self.prefix = [block.offset for block in self.blocks]

        self.inner_size = offset

        self.color_size = self.kmax * self.inner_size

        self.dimension = self.color_states * self.color_size

    # ========================================================
    # decode
    # ========================================================

    def decode(self, index):

        color_state, rem = divmod(index, self.color_size)

        k1_index, rem = divmod(rem, self.inner_size)

        k1 = k1_index + 1
        k2 = self.K - k1

        block_id = bisect_right(self.prefix, rem) - 1

        block = self.blocks[block_id]

        local = rem - block.offset

        m1 = block.valid_m1[local]

        m2 = block.target_m - m1

        g1 = SingleParticleState(
            k=k1,
            s=block.s1,
            n=block.n1,
            m=m1
        )

        g2 = SingleParticleState(
            k=k2,
            s=block.s2,
            n=block.n2,
            m=m2
        )

        return TwoParticleState(
            particles=(g1, g2),
            color_state=color_state
        )

    def generate_basis(self):

        return [self.decode(i) for i in range(self.dimension)]


# ============================================================
# 输出工具
# ============================================================


def write_basis(filename, basis):

    with open(filename, "w") as f:

        for i, state in enumerate(basis):

            g1, g2 = state.particles

            f.write(
                f"{i:6d}  "
                f"{state.color_state:2d}  "
                f"{g1.k:2d} {g1.s:2d} {g1.n:2d} {g1.m:3d}   "
                f"{g2.k:2d} {g2.s:2d} {g2.n:2d} {g2.m:3d}\n"
            )


# ============================================================
# 比较函数
# ============================================================


def compare_basis(old_basis, new_basis):

    if len(old_basis) != len(new_basis):

        print("Dimension mismatch!")
        print(len(old_basis), len(new_basis))

        return False

    for i, (a, b) in enumerate(zip(old_basis, new_basis)):

        if a != b:

            print("Mismatch at index", i)

            print("OLD =", a)
            print("NEW =", b)

            return False

    return True


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    Nmax = 5
    K = 5
    Mj = 2
    color_states = 2

    # --------------------------------------------------------
    # old
    # --------------------------------------------------------

    old_basis = generate_gg_basis_old(
        Nmax,
        Mj,
        K,
        color_states
    )

    write_basis("Output/basis_old.dat", old_basis)

    # --------------------------------------------------------
    # new
    # --------------------------------------------------------

    basis_decoder = GGBasis(
        Nmax,
        Mj,
        K,
        color_states
    )

    new_basis = basis_decoder.generate_basis()

    write_basis("Output/basis_new.dat", new_basis)

    # --------------------------------------------------------
    # compare
    # --------------------------------------------------------

    ok = compare_basis(old_basis, new_basis)

    print()
    print("=" * 60)

    if ok:
        print("OLD and NEW basis are IDENTICAL")
    else:
        print("OLD and NEW basis are DIFFERENT")

    print("=" * 60)

    print()
    print("dimension =", len(old_basis))

