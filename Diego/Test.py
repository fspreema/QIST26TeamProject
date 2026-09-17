

local_space = Index(2)
Z = ket(local_space[0]) * bra(local_space[0]) - (ket(local_space[1]) * bra(local_space[1]))
X = ket(local_space[0]) * bra(local_space[1]) + ket(local_space[1]) * bra(local_space[0])
Id = ket(local_space[0]) * bra(local_space[0]) + ket(local_space[1]) * bra(local_space[1])

P0 = ket(local_space[0]) * bra(local_space[0])
P1 = ket(local_space[1]) * bra(local_space[1])

Qmas = P0 ** P0 + P1 ** P1
Qmenos = P1 ** P0 + P0 ** P1

print ("ZZ")
print ( value ( Z ** Z) )
print ( value ( Qmas - Qmenos ) )
print("IdId")
print ( value ( Id ** Id ) )
print ( value ( Qmas + Qmenos ) )
