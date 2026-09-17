from wcnf_matrix import *

spin_space = Index(2)
spin_a, spin_b = Reg(spin_space), Reg(spin_space)

# Create basic projection operators for state 0 and 1
p0 = ket(spin_space[0]) * bra(spin_space[0])
p1 = ket(spin_space[1]) * bra(spin_space[1])

# Bind them to the registers
proj_a0 = p0 | spin_a
proj_a1 = p1 | spin_a

print("Projector operators initialized sucessfully")

# Construct sigma_z operator for a single spin
sigma_z = 1.0 * p0 + (-1.0) * p1

print("Sigma_z operator constructed successfully:")
print(value(sigma_z))

# Bind sigma_z to both registers
interaction_term = (sigma_z | spin_a) * (sigma_z | spin_b)

# Evaluate it using WMC:
evaluate_interaction = value(interaction_term)

print("Two-spin Ising interaction matrix (sigma_z ⊗ sigma_z) evaluated via WMC:")
print(evaluate_interaction)