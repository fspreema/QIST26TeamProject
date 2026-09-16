from wcnf_matrix import *
from functools import reduce

spin_space = Index(2)
p0 = ket(spin_space[0]) * bra(spin_space[0])
p1 = ket(spin_space[1]) * bra(spin_space[1])
sigma_z = 1.0 * p0 + (-1.0) * p1
identity_matrix = p0 + p1

# Define our physical parameters:
n_spins = 3 # Number of spins in the system
J = 1.5     # Interaction strength
h = 0.5     # External field

spins = [Reg(spin_space) for i in range(n_spins)]

interaction_terms = []
for i in range(n_spins - 1):
    factors = []
    for j in range(n_spins):
        if j == i:
            factors.append((-J * sigma_z) | spins[j])
        elif j == i + 1:
            factors.append(sigma_z | spins[j])
        else:
            factors.append(identity_matrix | spins[j])

    Kronecker_product = reduce(lambda x, y: x * y, factors)
    interaction_terms.append(value(Kronecker_product))

field_terms = []
for i in range(n_spins):
    factors = []
    for j in range(n_spins):
        if i == j:
            factors.append((-h * sigma_z) | spins[j])
        else:
            factors.append(identity_matrix | spins[j])

    Kronecker_product = reduce(lambda x, y: x * y, factors)
    field_terms.append(value(Kronecker_product))

total_interaction = reduce(lambda x, y: x + y, interaction_terms)
total_field = reduce(lambda x, y: x + y, field_terms)

# Total Hamiltonian for the two-spin Ising model with external field
hamiltonian = total_interaction + total_field

print(f"Total Hamiltonian for the {n_spins}-spin chain:")
print(hamiltonian)

#This method takes a lot of time to evaluate the Hamiltonian for larger spin chains.
# For this reason, is better to use the partition function and use the WMC to evaluate the trace of the Hamiltonian, which is more efficient and faster.