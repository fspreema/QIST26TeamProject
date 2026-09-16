from wcnf_matrix import *

# 1. Define the 2-state space
spin_space = Index(2) 

# 2. Create two separate registers
spin_a, spin_b = Reg(spin_space), Reg(spin_space)

# 3. Create a matrix for a single spin (e.g., ket(0) * bra(0))
proj_a = (ket(spin_space[0]) * bra(spin_space[0])) | spin_a
proj_b = (ket(spin_space[1]) * bra(spin_space[1])) | spin_b

# 4. Combine them using the Kronecker product (*)
combined_operator = proj_a * proj_b

print("Combined operator evaluated via WMC:")
print(value(combined_operator))