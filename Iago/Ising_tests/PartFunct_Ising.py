import math
import time
from wcnf_matrix import *
from functools import reduce

set_model_counter(Cachet)

def build_partition_function(n_spins, J=1.5, h=0.5, beta=1.0):
    spin_space = Index(2)
    p0 = ket(spin_space[0]) * bra(spin_space[0])
    p1 = ket(spin_space[1]) * bra(spin_space[1])
    identity = p0 + p1
    spins = [Reg(spin_space) for i in range(n_spins)]

    # Start with a global identity state
    identity_factors = []
    for i in range(n_spins):
        identity_factors.append(identity | spins[i])

    global_operator = reduce(lambda x, y: x * y, identity_factors)

    # Multiply by Field Operators:
    exp_h = math.exp(beta * h) * p0 + math.exp(-beta * h) * p1

    for i in range(n_spins):
        global_operator = global_operator * (exp_h | spins[i])

    # Multiply by Interactions Operators:
    exp_plus = math.exp(beta * J)
    exp_minus = math.exp(-beta * J)
    for i in range(n_spins - 1):
        term_00 = ((exp_plus * p0) | spins[i]) * (p0 | spins[i + 1])
        term_11 = ((exp_plus * p1) | spins[i]) * (p1 | spins[i + 1])

        term_01 = ((exp_minus * p0) | spins[i]) * (p1 | spins[i + 1])
        term_10 = ((exp_minus * p1) | spins[i]) * (p0 | spins[i + 1])

        # Aligned spins get +J, anti-aligned spins get -J energy
        local_exp_J =term_00 + term_11 + term_01 + term_10
        global_operator = global_operator * local_exp_J

    return global_operator

# Benchmarking Loop
for N in [2, 4, 6, 8, 10]:
    print(f"Benchmarking N = {N}")
    logical_formula = build_partition_function(N)

    cnf, weight_function = logical_formula.mat.trace_formula()

    start_time = time.time()
    result = weight_function(cnf)
    end_time = time.time()

    print(f"Partition function Z for N={N}: {result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")