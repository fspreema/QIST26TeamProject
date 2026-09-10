from wcnf_matrix import *

I = Index (2)
X = ket ( I [0]) * bra ( I [0]) - ( ket ( I [1]) * bra ( I [1]) )
Id = ket ( I [0]) * bra ( I [0]) + ket ( I [1]) * bra ( I [1])

# physical parameters:
n_spins = 2 # Number of spins in the system
J = 1.5     # Interaction strength
h = 0.5     # External field

X_1 = X**Id
X_2 = Id**X

H2 = - J * (X_1*X_2) - h * (X_1 + X_2)

print( value ( H2 ))