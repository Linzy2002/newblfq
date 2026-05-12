"""
Verify global index <-> basis mapping against basis.py enumeration order.

Uses Nmax=4, K=4, Mj=2, color_states=2 for gg; g sector has at most one state
under the same rules as generate_1p_basis.

PackedBasisTable: one O(dim_gg) preprocessing pass fills int32 columns; each
decode is O(1) indexing (no full-loop replay). Irregular N1+N2 truncation means
there is no single global "product + mod" formula; block_prefix gives coarse
offsets per (color_state, k1) for scheduling / parallelism.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from basis import (
    SingleParticleState,
    TwoParticleState,
    build_global_basis,
    generate_1p_basis,
    generate_gg_basis,
)


@dataclass(frozen=True)
class PackedBasisTable:
    """
    Precomputed flat table for global index -> quantum numbers.

    - build(): single walk, same order as generate_gg_basis + g block layout.
    - global_index_to_state(idx): O(1) after build.
    - gg_block_prefix[b]: cumulative gg count at start of block b, where
      b = color_state * kmax + (k1 - 1), kmax = K - 1. Length = n_blocks + 1.
    """

    Nmax: int
    Mj: int
    K: int
    color_states: int
    ng: int
    dim_gg: int
    g_k: int
    g_s: int
    g_n: int
    g_m: int
    gg_block_prefix: np.ndarray
    gg_color: np.ndarray
    gg_k1: np.ndarray
    gg_k2: np.ndarray
    gg_s1: np.ndarray
    gg_s2: np.ndarray
    gg_n1: np.ndarray
    gg_n2: np.ndarray
    gg_m1: np.ndarray
    gg_m2: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.ng + self.dim_gg)

    @property
    def kmax(self) -> int:
        return self.K - 1

    @property
    def n_gg_blocks(self) -> int:
        return int(self.color_states * self.kmax)

    @classmethod
    def build(cls, Nmax: int, Mj: int, K: int, color_states: int) -> PackedBasisTable:
        ng = one_p_basis_dim(Nmax, Mj, K)
        dim_gg = _count_gg_basis(Nmax, Mj, K, color_states)
        kmax = K - 1
        n_blocks = color_states * kmax

        if ng == 1:
            st = decode_1p_state(Nmax, Mj, K)
            g_k, g_s, g_n, g_m = st.k, st.s, st.n, st.m
        else:
            g_k = g_s = g_n = g_m = 0

        gg_block_prefix = np.zeros(n_blocks + 1, dtype=np.int64)
        gg_color = np.empty(dim_gg, dtype=np.int32)
        gg_k1 = np.empty(dim_gg, dtype=np.int32)
        gg_k2 = np.empty(dim_gg, dtype=np.int32)
        gg_s1 = np.empty(dim_gg, dtype=np.int32)
        gg_s2 = np.empty(dim_gg, dtype=np.int32)
        gg_n1 = np.empty(dim_gg, dtype=np.int32)
        gg_n2 = np.empty(dim_gg, dtype=np.int32)
        gg_m1 = np.empty(dim_gg, dtype=np.int32)
        gg_m2 = np.empty(dim_gg, dtype=np.int32)

        w = 0
        block_id = 0
        for color_state in range(color_states):
            for k1 in range(1, kmax + 1):
                gg_block_prefix[block_id] = w
                block_id += 1
                k2 = K - k1
                for s2 in (-1, +1):
                    for s1 in (-1, +1):
                        target_m = Mj // 2 - s1 - s2
                        max_n = Nmax // 2
                        for n2 in range(max_n + 1):
                            for n1 in range(max_n - n2 + 1):
                                lower = -floor(
                                    (Nmax - 2 - 2 * n1 - 2 * n2 - target_m) / 2
                                )
                                upper = ceil(
                                    (Nmax - 2 - 2 * n1 - 2 * n2 + target_m) / 2
                                )
                                for m1 in range(lower, upper + 1):
                                    m2 = target_m - m1
                                    N1 = 2 * n1 + abs(m1) + 1
                                    N2 = 2 * n2 + abs(m2) + 1
                                    if N1 + N2 > Nmax:
                                        continue
                                    gg_color[w] = color_state
                                    gg_k1[w] = k1
                                    gg_k2[w] = k2
                                    gg_s1[w] = s1
                                    gg_s2[w] = s2
                                    gg_n1[w] = n1
                                    gg_n2[w] = n2
                                    gg_m1[w] = m1
                                    gg_m2[w] = m2
                                    w += 1

        assert w == dim_gg
        assert block_id == n_blocks
        gg_block_prefix[n_blocks] = dim_gg

        return cls(
            Nmax=Nmax,
            Mj=Mj,
            K=K,
            color_states=color_states,
            ng=ng,
            dim_gg=dim_gg,
            g_k=g_k,
            g_s=g_s,
            g_n=g_n,
            g_m=g_m,
            gg_block_prefix=gg_block_prefix,
            gg_color=gg_color,
            gg_k1=gg_k1,
            gg_k2=gg_k2,
            gg_s1=gg_s1,
            gg_s2=gg_s2,
            gg_n1=gg_n1,
            gg_n2=gg_n2,
            gg_m1=gg_m1,
            gg_m2=gg_m2,
        )

    def global_index_to_state(
        self, idx: int
    ) -> tuple[str, SingleParticleState | TwoParticleState]:
        if idx < 0 or idx >= self.dim:
            raise IndexError(idx)
        if idx < self.ng:
            return (
                "g",
                SingleParticleState(
                    k=self.g_k, s=self.g_s, n=self.g_n, m=self.g_m
                ),
            )
        j = idx - self.ng
        return (
            "gg",
            TwoParticleState(
                particles=(
                    SingleParticleState(
                        int(self.gg_k1[j]),
                        int(self.gg_s1[j]),
                        int(self.gg_n1[j]),
                        int(self.gg_m1[j]),
                    ),
                    SingleParticleState(
                        int(self.gg_k2[j]),
                        int(self.gg_s2[j]),
                        int(self.gg_n2[j]),
                        int(self.gg_m2[j]),
                    ),
                ),
                color_state=int(self.gg_color[j]),
            ),
        )

    def gg_block_id(self, color_state: int, k1: int) -> int:
        return int(color_state * self.kmax + (k1 - 1))

    def gg_indices_for_block(self, color_state: int, k1: int) -> range:
        """Half-open range of gg flat indices j for this (color_state, k1)."""
        b = self.gg_block_id(color_state, k1)
        lo = int(self.gg_block_prefix[b])
        hi = int(self.gg_block_prefix[b + 1])
        return range(lo, hi)


def one_p_basis_dim(Nmax: int, Mj: int, K: int) -> int:
    """Same acceptance as generate_1p_basis: count is 0 or 1."""
    n, m = 0, 0
    return 1 if n + abs(m) + 1 <= Nmax else 0


def decode_1p_state(Nmax: int, Mj: int, K: int) -> SingleParticleState:
    """Only valid when one_p_basis_dim == 1 (same as generate_1p_basis)."""
    s = 1 if Mj >= 0 else -1
    n, m = 0, 0
    if not (n + abs(m) + 1 <= Nmax):
        raise ValueError("no 1p state for these parameters")
    return SingleParticleState(k=K, s=s, n=n, m=m)


def decode_gg_relative(rel_idx: int, Nmax: int, Mj: int, K: int, color_states: int) -> TwoParticleState:
    """
    rel_idx in [0, dim_gg): walk the same loops as generate_gg_basis in basis.py
    and return the rel_idx-th accepted TwoParticleState.
    """
    if rel_idx < 0:
        raise IndexError(rel_idx)

    kmax = K - 1
    countdown = rel_idx

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
                                (Nmax - 2 - 2 * n1 - 2 * n2 - target_m) / 2
                            )
                            upper = ceil(
                                (Nmax - 2 - 2 * n1 - 2 * n2 + target_m) / 2
                            )
                            for m1 in range(lower, upper + 1):
                                m2 = target_m - m1
                                n1v, n2v = n1, n2
                                N1 = 2 * n1v + abs(m1) + 1
                                N2 = 2 * n2v + abs(m2) + 1
                                if N1 + N2 > Nmax:
                                    continue
                                if countdown == 0:
                                    g1 = SingleParticleState(k1, s1, n1v, m1)
                                    g2 = SingleParticleState(k2, s2, n2v, m2)
                                    return TwoParticleState(
                                        particles=(g1, g2),
                                        color_state=color_state,
                                    )
                                countdown -= 1

    raise IndexError(f"gg rel_idx={rel_idx} out of range for Nmax={Nmax}, K={K}")


def global_index_to_state(
    idx: int,
    Nmax: int,
    Mj: int,
    K: int,
    color_states: int,
) -> tuple[str, SingleParticleState | TwoParticleState]:
    """
    Global row/column index in the same order as build_global_basis:
    sector_basis = {\"g\": g_basis, \"gg\": gg_basis} -> g block then gg block.

    Python idx runs 0 .. dim-1 (Fortran-style 1..dim would use idx = i - 1).
    """
    ng = one_p_basis_dim(Nmax, Mj, K)
    if idx < ng:
        return ("g", decode_1p_state(Nmax, Mj, K))
    return ("gg", decode_gg_relative(idx - ng, Nmax, Mj, K, color_states))


def basis_list_from_mapping(
    Nmax: int, Mj: int, K: int, color_states: int
) -> list[tuple[str, SingleParticleState | TwoParticleState]]:
    ng = one_p_basis_dim(Nmax, Mj, K)
    dim_gg = _count_gg_basis(Nmax, Mj, K, color_states)
    dim = ng + dim_gg
    return [global_index_to_state(i, Nmax, Mj, K, color_states) for i in range(dim)]


def _count_gg_basis(Nmax: int, Mj: int, K: int, color_states: int) -> int:
    """Count only; uses same rules as generate_gg_basis (no object allocation)."""
    kmax = K - 1
    count = 0
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
                                (Nmax - 2 - 2 * n1 - 2 * n2 - target_m) / 2
                            )
                            upper = ceil(
                                (Nmax - 2 - 2 * n1 - 2 * n2 + target_m) / 2
                            )
                            for m1 in range(lower, upper + 1):
                                m2 = target_m - m1
                                N1 = 2 * n1 + abs(m1) + 1
                                N2 = 2 * n2 + abs(m2) + 1
                                if N1 + N2 > Nmax:
                                    continue
                                count += 1
    return count


def run_check() -> None:
    Nmax, K, Mj, color_states = 4, 4, 2, 2

    g_basis = generate_1p_basis(Nmax, Mj, K)
    gg_basis = generate_gg_basis(Nmax, Mj, K, color_states)
    sector_basis = {"g": g_basis, "gg": gg_basis}
    global_basis_old, _ = build_global_basis(sector_basis)
    dim = len(global_basis_old)

    ng = one_p_basis_dim(Nmax, Mj, K)
    dim_gg = len(gg_basis)
    assert ng == len(g_basis)
    assert dim == ng + dim_gg
    assert dim_gg == _count_gg_basis(Nmax, Mj, K, color_states)

    global_basis_new = basis_list_from_mapping(Nmax, Mj, K, color_states)
    assert len(global_basis_new) == dim

    mismatches = 0
    for idx in range(dim):
        old_sector, old_state = global_basis_old[idx]
        new_sector, new_state = global_basis_new[idx]
        if old_sector != new_sector or old_state != new_state:
            mismatches += 1
            print(
                f"mismatch idx={idx} (1-based idx_f={idx + 1}):\n"
                f"  old: {old_sector!r}, {old_state!r}\n"
                f"  new: {new_sector!r}, {new_state!r}"
            )

    # Per-index direct decode (same as list from mapping)
    for idx in range(dim):
        dec = global_index_to_state(idx, Nmax, Mj, K, color_states)
        if dec != global_basis_old[idx]:
            mismatches += 1
            print(f"decode mismatch idx={idx}: {dec!r} vs {global_basis_old[idx]!r}")

    # O(1) packed table vs old basis
    packed = PackedBasisTable.build(Nmax, Mj, K, color_states)
    assert packed.dim == dim
    assert packed.ng == ng
    assert packed.dim_gg == dim_gg
    for idx in range(dim):
        if packed.global_index_to_state(idx) != global_basis_old[idx]:
            mismatches += 1
            print(f"packed mismatch idx={idx}")

    acc = 0
    for cs in range(color_states):
        for k1 in range(1, packed.kmax + 1):
            r = packed.gg_indices_for_block(cs, k1)
            assert r.start == acc
            acc = r.stop
    assert acc == dim_gg

    if mismatches:
        raise SystemExit(f"FAILED: {mismatches} mismatch(es)")

    print(
        f"OK: dim={dim} (ng={ng}, dim_gg={dim_gg}), slow map + PackedBasisTable "
        f"match; gg_blocks={packed.n_gg_blocks}."
    )


if __name__ == "__main__":
    run_check()
