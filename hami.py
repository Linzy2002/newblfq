from math import  sqrt
from collections import defaultdict
from TMC import TMC
from tool import *

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

    fourierphasen = -np1-np2+nk1+nk2
    fourierphasem = -abs(mp1)-abs(mp2)+abs(mk1)+abs(mk2)

    fourierphase = ((-1) ** fourierphasen) * (1j ** fourierphasem)

    lagrangeterm = lag * bee ** 2 * (
        (Ep1p1 * kp1half / Pplus +
         2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
         Ep2p2 * kp2half / Pplus)
        +
        (Ep1p1 * kp1half / Pplus +
         2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
         Ep2p2 * kp2half / Pplus)*fourierphase 
        - shift
    ) / Pplus

    # lagrangeterm = lag * bee ** 2 * (
    #     (Ep1p1 * kp1half / Pplus +
    #      2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
    #      Ep2p2 * kp2half / Pplus)
    #     +
    #     (Ep1p1 * kp1half / Pplus +
    #      2 * Ep1p2 * np.sqrt(kp1half * kp2half / Pplus ** 2) +
    #      Ep2p2 * kp2half / Pplus)
    #     - shift
    #     ) / Pplus
    
    # print("lag = ", lag, "b2 = ", bee**2, "Pplus = ", Pplus)
    




    newlag = np.abs(lagrangeterm) if abs(lagrangeterm.imag) > 1e-12 else lagrangeterm.real
    # -----------------------------
    # Hamiltonian
    # -----------------------------
    hamiltonian = (
        EnergySingle
        - EnergyCM
        + newlag
    )
    
    # if(hamiltonian!=0):
    #     print("hamiltonian =", hamiltonian ,"EnergySingle = ", EnergySingle, "EnergyCM = ", EnergyCM, "lagrangeterm = ", lagrangeterm)

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
        # CF = -np.sqrt(3.0)
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

    
    coupling_eff = couplings * CF / sqrt(2.0)

    # coupling_eff = couplings * CF 

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