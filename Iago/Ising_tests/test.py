from wcnf_matrix import *
from functools import reduce

I = Index(3)
M1 = ket(I[0]) * bra(I[1])
M2 = ket(I[2]) * bra(I[0])

print(value(M1 * M2))
#print(value(M1 ** M2))

I = Index(2)
M1 = 2 * ket(I[0]) * bra(I[0]) + ket(I[1]) * bra(I[1])

print(value(M1))

M2 = reduce(lambda x, y: x ** y, [M1]*100)
cnf, weight_func = M2.trace_formula()

print(weight_func(cnf))

B = reduce(lambda x, y: x ** y, [bra(I[0])]*100)
K = reduce(lambda x, y: x ** y, [ket(I[0])]*100)

print(value(B * M2 * K)) # Retrieving the top left entry of M2