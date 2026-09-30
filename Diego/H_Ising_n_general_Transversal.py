from wcnf_matrix import *
from functools import reduce
import numpy as np

# Settings
do_periodic_boundary_condition = False

# Number of spins in the system
n_spins = 2

# Creation of the spin space and the X and Id matrices
local_space = Index (2)

# Z  = (1 , 0)
#      (0 ,-1)
Z  = ket ( local_space [0]) * bra ( local_space [0]) - ( ket ( local_space [1]) * bra ( local_space [1]) )

# X  = (0 , 1)
#      (1 , 0)
X = ket(local_space[0])*bra(local_space[1]) + ket(local_space[1])*bra(local_space[0])

# Id = (1 , 0)
#      (0 , 1)
Id = ket ( local_space [0]) * bra ( local_space [0]) +   ket ( local_space [1]) * bra ( local_space [1])

# physical parameters:
n_spins = 3 # Number of spins in the system

cte_matrix = True # If True, J_ij = J and h_i = h. If False, you must provide the matrix (h_i diagonal)

J = 1.5     # Interaction strength
h = 0.5     # External field

J_matrix = np.array(
    [[0.5 , 1.5 , 0   ], 
     [1.5 , 0.5 , 1.5 ], 
     [0   , 1.5 , 0.5 ]]
     )

if cte_matrix:
    Weight_Matrix = np.zeros((n_spins, n_spins))
    np.fill_diagonal(Weight_Matrix, h)
    for i in range(n_spins):
        for j in range(n_spins):
            if i == j+1 or i == j-1:
                Weight_Matrix[i][j] = J
else:
    Weight_Matrix = J_matrix

print("Weight Matrix:")
print(Weight_Matrix)

spins = [Reg(local_space) for _ in range(n_spins)]


interaction_terms = []
for i in range(n_spins):
    for j in range(i + 1, n_spins):
        Jij = Weight_Matrix[i][j]
        if Jij == 0:
            continue
        factors = []
        for k in range(n_spins):
            if k == i:
                factors.append((-Jij * Z) | spins[k])
            elif k == j:
                factors.append(Z | spins[k])
            else:
                factors.append(Id | spins[k])
        kron_product = reduce(lambda x, y: x * y, factors)
        interaction_terms.append(value(kron_product))

total_interaction = reduce(lambda x, y: x + y, interaction_terms)


field_terms = []
for i in range(n_spins):
    hi = Weight_Matrix[i][i]
    factors = []
    for k in range(n_spins):
        if k == i:
            factors.append((-hi * X) | spins[k])
        else:
            factors.append(Id | spins[k])
    kron_product = reduce(lambda x, y: x * y, factors)
    field_terms.append(value(kron_product))

total_field = reduce(lambda x, y: x + y, field_terms)

H = total_interaction + total_field
H = H.permutation(spins)
print(f"Total Hamiltonian for the {n_spins}-spin chain (transverse field):")
print(H)