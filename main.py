from dataclasses import dataclass
from math import sqrt, isnan
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


from basis import *
from hami import *
from Ctool import *

# =========================================================
# 1. Physics parameters 
# =========================================================

@dataclass(frozen=True)
class PhysicsParams:
    couplings: float
    b: float
    mass_g: float
    mass_gg: float




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

    return H



def renorm(Nmax, kt, b, coupling, loop_max=30, tol=1e-10):


    # -------------------------
    # 初始构造
    # -------------------------
    params = PhysicsParams(
        couplings=coupling,
        b=b,
        mass_g=0.0,
        mass_gg=0.0
    )

    H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)

    # dump_hamiltonian_dense(H, "Output/h.dat")

    vals, vecs = eigsh(H, k=3, which='SA')
    eigenv1 = vals[0] * kt

    # print("vals[0] = ", vals[0] , "kt = ", kt)

    renomass2 = np.sqrt(-eigenv1)
    inputmass = renomass2


    params = PhysicsParams(
        couplings=coupling,
        b=b,
        mass_g=inputmass,
        mass_gg=inputmass
    )

    H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)
    vals, vecs = eigsh(H, k=3, which='SA')

    eigenv2 = vals[0] * kt
    # print("eigenv1 = ", eigenv1)
    # print("eigenv2 = ", eigenv2)

    renomass1 = 0.0

    for _ in range(loop_max):

        if abs(eigenv2) < tol:
            break


        renomass3 = np.sqrt(
            renomass1**2 +
            (renomass2**2 - renomass1**2) / (eigenv2 - eigenv1) * (-eigenv1)
        )

        # print("renomass3 = ", renomass3)

        inputmass = renomass3

        params = PhysicsParams(
            couplings=coupling,
            b=b,
            mass_g=inputmass,
            mass_gg=inputmass
        )

        H = build_hamiltonian(Nmax=Nmax, K=kt, params=params)
        vals, vecs = eigsh(H, k=3, which='SA')

        # shift update
        eigenv1 = eigenv2
        eigenv2 = vals[0] * kt

        renomass1 = renomass2
        renomass2 = renomass3

    return renomass2

def scan_and_plot(
    coupling_range,
    b_range,
    Nmax=3,
    kt=2,
    savefile="scan.dat",
    plotfile=None
):
    """
    扫描 renorm 并画：
        横轴 = coupling
        纵轴 = renorm mass
        不同 b = 不同折线
    """

    coupling_min, coupling_max, coupling_step = coupling_range
    b_min, b_max, b_step = b_range

    couplings = np.arange(
        coupling_min,
        coupling_max + coupling_step,
        coupling_step
    )

    bs = np.arange(
        b_min,
        b_max + b_step,
        b_step
    )

    result = np.zeros((len(bs), len(couplings)))

    # =========================================
    # 扫描
    # =========================================
    with open(savefile, "w") as f:
        
        max_retry = 5
        for i, b in enumerate(bs):
            for j, coupling in enumerate(couplings):
                
                value = np.nan

                # 最多重复 max_retry 次
                for attempt in range(max_retry):

                    value = renorm(
                        Nmax=Nmax,
                        kt=kt,
                        b=b,
                        coupling=coupling
                    )

                    # 如果不是 nan 就退出
                    if not isnan(value):
                        break

                result[i, j] = value

                f.write(
                    f"{b:15.8f} "
                    f"{coupling:15.8f} "
                    f"{value:20.10e}\n"
                )

                print(
                    f"b={b:.3f}, "
                    f"coupling={coupling:.3f}, "
                    f"renorm={value:.6e}"
                )

    # =========================================
    # 画图
    # =========================================
    plt.figure(figsize=(8, 6))

    for i, b in enumerate(bs):

        plt.plot(
            couplings,
            result[i],
            marker='o',
            linewidth=2,
            markersize=5,
            label=fr"$b={b:.2f}$"
        )

    plt.xlabel(r"Coupling_g", fontsize=14)
    plt.ylabel(r"Smallest deltam", fontsize=14)

    plt.title(
        fr"$N_{{\max}}={Nmax},\ K_{{\max}}={kt}$",
        fontsize=15
    )

    plt.grid(alpha=0.3)

    plt.legend(
        fontsize=10,
        frameon=False,
        ncol=2
    )

    plt.tight_layout()

    if plotfile is not None:
        plt.savefig(plotfile, dpi=300)

    plt.show()

    return bs, couplings, result

# =========================================================
# 11. Example
# =========================================================

if __name__ == "__main__":

    
    # Nmax = 6  
    # Kmax = 6
    # b = 1.2
    # coupling = 1.5


    # output_file = "/home/linzy/glueball/MainProgram/renom.dat"

    # with open(output_file, "w") as f:

    #     for k in range(2, Kmax + 1):
    #         for n in range(3, Nmax + 1):

    #             renomass2 = renorm(
    #                 Nmax=n,
    #                 kt=k,
    #                 b=b,
    #                 coupling=coupling
    #             )

    #             # 只写三列：n k value
    #             f.write(f"{n:6d} {k:6d} {renomass2:.10e}\n")
    
    scan_and_plot(
    coupling_range=(0.5, 9.0, 1.0),
    b_range=(0.5, 1.6, 0.3),
    savefile="Output/scan.dat",
    plotfile="Output/scan.png"
)

    

