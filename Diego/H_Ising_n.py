from wcnf_matrix import *
from functools import reduce

# Number of spins in the system
n_spins = 2

# Creation of the spin space and the X and Id matrices
local_space = Index (2)

# Z  = (1 , 0)
#      (0 ,-1)
Z  = ket ( local_space [0]) * bra ( local_space [0]) - ( ket ( local_space [1]) * bra ( local_space [1]) )

# Id = (1 , 0)
#      (0 ,1)
Id = ket ( local_space [0]) * bra ( local_space [0]) +   ket ( local_space [1]) * bra ( local_space [1])

# physical parameters:
n_spins = 3 # Number of spins in the system
J = 1.5     # Interaction strength
h = 0.5     # External field

spins = [Reg(local_space) for _ in range(n_spins)]

interaction_factors = []

for i in range(n_spins - 1):
    factors = []
    for j in range(n_spins):
        if j == i:
            factors.append((-J * Z) | spins[j])
        elif j == i + 1:
            factors.append(Z | spins[j])
        else:
            factors.append(Id | spins[j])

    neighbour_factor = reduce(lambda x, y: x * y, factors)
    interaction_factors.append(neighbour_factor)

interaction_term = reduce(lambda x, y: x + y, interaction_factors)



external_factors = []
for _ in range(n_spins):
    external_factors.append((-h * Z) | spins[_])
external_term = reduce(lambda x, y: x + y, external_factors)

H = (interaction_term + external_term)



print(f"Total Hamiltonian for the {n_spins}-spin chain:")
print( value(H) )