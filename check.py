import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh  

filename = "/home/linzy/glueball/Output/renohami.dat"
data = np.loadtxt(filename)
row_ptr = data[:, 0].astype(int)
col_ind = data[:, 1].astype(int)
values  = data[:, 2]

# 转为 0-based 索引（Fortran -> Python）
row_ptr -= 1
col_ind -= 1

nrows = 1 + 56

# 只保留有效部分
row_ptr = row_ptr[:nrows + 1]
nnz = row_ptr[-1]
col_ind = col_ind[:nnz]
values  = values[:nnz]

# 填充稀疏矩阵
ncols = col_ind.max() + 1
A = csr_matrix((values, col_ind, row_ptr), shape=(nrows, ncols))

# B = A.toarray()

eigvals, eigvecs = eigsh(A, k=6, which='SA') 

print(eigvals)


def load_dense_hamiltonian(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
    dim = None

    for line in lines:
        if line.startswith("#"):
            if "dimension" in line:
                dim = int(line.split("=")[1])
            continue

        i, j, re = line.split()
        data.append((int(i), int(j), float(re)))

    H = np.zeros((dim, dim), dtype=np.complex128)

    for i, j, re in data:
        H[i, j] = re 

    return H


def dump_matrix_diff(
    H_old,
    H_new,
    filename="Output/diff_matrix.dat",
    filename_sparse="Output/diff_sparse.dat",
    eps=1e-10
):
    """
    输出：

    1. 完整差分矩阵
       H_old - H_new

    2. 稀疏非零差分项
       只记录 abs(diff) > eps 的元素
    """

    diff = H_old - H_new
    dim = diff.shape[0]

    # =========================================================
    # 完整矩阵输出
    # =========================================================

    with open(filename, "w") as f:

        f.write("# Matrix: H_old - H_new (thresholded)\n")
        f.write(f"# dim = {dim}, eps = {eps}\n\n")

        for i in range(dim):

            line_real = ""

            for j in range(dim):

                val = diff[i, j]

                re = val.real
                im = val.imag

                if abs(re) < eps:
                    re = 0.0

                if abs(im) < eps:
                    im = 0.0

                line_real += f"{re:12.4e} "

            f.write(f"row {i:04d} RE: {line_real}\n")

    # =========================================================
    # 稀疏非零项输出
    # =========================================================

    count = 0

    with open(filename_sparse, "w") as f:

        f.write("# Nonzero entries of H_old - H_new\n")
        f.write(f"# eps = {eps}\n\n")

        f.write(
            f"{'i':>6} {'j':>6} "
            f"{'Re(diff)':>18} {'Im(diff)':>18} "
            f"{'Abs(diff)':>18}\n"
        )

        for i in range(dim):
            for j in range(dim):

                val = diff[i, j]

                re = val.real
                im = val.imag

                if abs(re) < eps:
                    re = 0.0

                if abs(im) < eps:
                    im = 0.0

                val_clean = re + 1j * im

                if abs(val_clean) > 0:

                    count += 1

                    f.write(
                        f"{i:6d} {j:6d} "
                        f"{re:18.10e}\n"
                    )

    print(f"diff matrix saved to {filename}")
    print(f"sparse diff saved to {filename_sparse}")
    print(f"nonzero entries = {count}")
    
    
def dump_matrix(H, filename="Output/matrix.dat"):
    """
    输出 H_old - H_new 的完整矩阵（适合小维度，比如 13x13）
    """

    diff = H
    dim = diff.shape[0]

    with open(filename, "w") as f:

        f.write("# Matrix: H\n")
        f.write(f"# dim = {dim}\n\n")

        for i in range(dim):

            line_real = ""
            line_imag = ""

            for j in range(dim):
                val = diff[i, j]

                # 实部
                line_real += f"{val.real:12.4e} "



            f.write(f"row {i:02d} RE: {line_real}\n")


    print(f"diff matrix saved to {filename}")
    
    
H_new = load_dense_hamiltonian("/home/linzy/Code/newblfq/Output/h.dat")

H_old = A.toarray() 


dump_matrix_diff(H_old, H_new)

dump_matrix( H_new)

dump_matrix( H_old, "Output/Hold.dat")
