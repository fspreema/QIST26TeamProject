from wcnf_matrix import *

spin_space = Index(2)
spin_a, spin_b = Reg(spin_space), Reg(spin_space)

p0 = ket(spin_space[0]) * bra(spin_space[0])
p1 = ket(spin_space[1]) * bra(spin_space[1])
sigma_z = 1.0 * p0 + (-1.0) * p1
identity_matrix = p0 + p1

# Define our physical parameters:
J = 1.5     # Interaction strength
h = 0.5     # External field

interaction_term = ((-J * sigma_z) | spin_a) * (sigma_z | spin_b)

field_a = ((-h * sigma_z) | spin_a) * (identity_matrix | spin_b)
field_b = (identity_matrix | spin_a) * ((-h * sigma_z) | spin_b)

# Total Hamiltonian for the two-spin Ising model with external field
hamiltonian = value(interaction_term) + value(field_a) + value(field_b)

print("Two-spin Ising Hamiltonian with external field evaluated via WMC:")
print(hamiltonian)