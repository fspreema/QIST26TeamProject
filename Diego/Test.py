from wcnf_matrix import *

I = Index (2)
X = ket ( I [0]) * bra ( I [0]) - ( ket ( I [1]) * bra ( I [1]) )
Id = ket ( I [0]) * bra ( I [0]) + ket ( I [1]) * bra ( I [1])

J = 1.5
h = 0.5

X_1 = X**Id
X_2 = Id**X

H2 = - J * (X_1*X_2) - h * (X_1 + X_2)


n = 3

X_i = []

for i in range(n):
    if (i == 0):
        X_aux = X
    else:
        X_aux = Id

    for j in range(1,n):
        if (i == j):
            X_aux = X_aux ** X
        else:
            X_aux = X_aux ** Id

    X_i.append(X_aux)

H = Id
for i in range(1,n):
    H = H ** Id
for i in range(n):
    for j in range(i,n):
        if (i!=j):
            H = H - J * (X_i[i] * X_i[j])

for i in range(n):
    H = H - h * X_i[i]

print( value ( H ) )