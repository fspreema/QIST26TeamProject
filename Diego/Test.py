from wcnf_matrix import *

I = Index (2)
X = ket ( I [0]) * bra ( I [0]) - ( ket ( I [1]) * bra ( I [1]) )
Id = ket ( I [0]) * bra ( I [0]) + ket ( I [1]) * bra ( I [1])

J = 1.5
h = 0.5

X_1 = X**Id**Id
X_2 = Id**X**Id
X_3 = Id**Id**X


H3 = - J * (X_1*X_2 + X_2*X_3) - h * (X_1 + X_2 + X_3)

print ( value ( H3 ))

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

print( value ( X_i[0] - X_1))
print( value ( X_i[1] - X_2))
print( value ( X_i[2] - X_3))


H = - J * (X_i[0] * X_i[1])
print("1, 2")
for i in range(1, n - 1):
    H = H - J * (X_i[i] * X_i[i+1])
    print(f"{i +1 }, {i + 2}")

for i in range(n):
    H = H - h * X_i[i]
    print(f"{i + 1}")
H = - J * (X_i[0]*X_i[1] + X_i[1]*X_i[2]) - h * (X_i[0] + X_i[1] + X_i[2])

print( value ( H ) )