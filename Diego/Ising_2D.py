from wcnf_matrix import *
from functools import reduce
from numpy import exp
import time

# Functions

def calc_pairs(n_spins, do_periodic_boundary_condition, _1D_ising_model):
    pairs = []
    if _1D_ising_model:
        for i in range(n_spins - 1):
            pairs.append((i, i + 1))
        if do_periodic_boundary_condition:
            pairs.append((0,n_spins - 1))
    else:
        for i in range(n_spins):
            for j in range(n_spins):
                index = i * n_spins + j
                if j < n_spins - 1:
                    right_index = i * n_spins + (j + 1)
                    pairs.append((index, right_index))
                elif do_periodic_boundary_condition:
                    right_index = i * n_spins
                    pairs.append((index, right_index))

                if i < n_spins - 1:
                    down_index = (i + 1) * n_spins + j
                    pairs.append((index, down_index))
                elif do_periodic_boundary_condition:
                    down_index = j
                    pairs.append((index, down_index))

    # we order the pairs so the first index is always smaller than the second one
    pairs = [(min(i, j), max(i, j)) for i, j in pairs]
    return list(set(pairs))

# Creation of the spin space and the P0, P1 and Id matrices

def calculate_the_Partition_Function (n_spins,J = 1.5, h = 0.5, beta = 1.0, boundary_condition = True, Dimension = False):
    local_space = Index (2)

    # P0 = (1 , 0)
    #      (0 , 0)
    P0 = ket ( local_space [0]) * bra ( local_space [0])

    # P1 = (0 , 0)
    #      (0 , 1)
    P1 = ket ( local_space [1]) * bra ( local_space [1])

    # Id = (1 , 0)
    #      (0 , 1)
    Id = ket ( local_space [0]) * bra ( local_space [0]) +   ket ( local_space [1]) * bra ( local_space [1])

    #  Qp = P0 ** P0 + P1 ** P1
    #  Qm = P0 ** P1 + P1 ** P0
    #  Z  = Qp - Qm
    #  Id = Qp + Qm
    if Dimension == True:
        N = n_spins
    elif Dimension == False:
        N = n_spins**2
    
    spins = [Reg(local_space) for _ in range(N)]

    # Preparation of the Identity Matrix
    id_factors = []
    for i in range(N):
        id_factors.append(Id | spins[i])
    EXPONENTIAl = reduce(lambda x, y: x * y, id_factors)

    # Identity2 = Id
    # for i in range(1,n_spins):
    #     Identity2 = Identity2 ** Id

    #Identity3 = WCNFMatrix.identity(local_space, n_spins)

    # Calculate the pairs that the Hamiltonian will take into account
    pairs = calc_pairs(n_spins, boundary_condition, Dimension)

    # Precompute the exponentials

    exp_pJbeta = exp( J * beta)
    exp_mJbeta = exp(-J * beta)
    exp_phbeta = exp( h * beta)
    exp_mhbeta = exp(-h * beta)



    for i,j in pairs:
        term_P0P0 = ((exp_pJbeta * P0) | spins[i]) * (P0 | spins[j])
        term_P1P1 = ((exp_pJbeta * P1) | spins[i]) * (P1 | spins[j])

        term_P0P1 = ((exp_mJbeta * P0) | spins[i]) * (P1 | spins[j])
        term_P1P0 = ((exp_mJbeta * P1) | spins[i]) * (P0 | spins[j])

        exp_JZZ_ij = term_P0P0 + term_P1P1 + term_P0P1 + term_P1P0
        EXPONENTIAl = EXPONENTIAl * exp_JZZ_ij


    exp_hZ_i = exp_phbeta * P0 + exp_mhbeta * P1

    for i in range(N):
        EXPONENTIAl = EXPONENTIAl * (exp_hZ_i | spins[i])

    return EXPONENTIAl

for nspins in [2,3,4]:

    # values
    J = 1.5
    h = 0.5
    beta = 1.0

    # settings
    do_periodic_boundary_condition = True
    _1D_ising_model = False

    N = nspins if _1D_ising_model else nspins**2

    # Prepare the exponential matrix
    logical_formula = calculate_the_Partition_Function(nspins, J, h, beta, do_periodic_boundary_condition, _1D_ising_model)
    cnf, weight_function = logical_formula.mat.trace_formula()

    # Calculate the trace
    start_time = time.time()
    result = weight_function(cnf)
    end_time = time.time()

    print(f"Partition function Z for N={N}: {result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")