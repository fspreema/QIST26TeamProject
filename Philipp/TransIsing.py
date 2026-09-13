

from wcnf_matrix import *
import numpy as np
from functools import reduce
from itertools import product




class TransIsingModel:

    def __init__(self,  nodes, weights, I = Index(2), X_field = 0 , Z_field = 0):
            """ size:    number of nodes in the graph
                weights: a dictionary (node_1,node_2) -> edge 
                I:       the Index of the matrix
                extfield:Dictionary (node) -> external field or float(if homogenous) """
            if isinstance(X_field, float):
                self.X_field = dict({node: X_field for node in nodes})
            else: 
                self.X_field = X_field
            if isinstance(Z_field, float):
                self.Z_field = dict({node : Z_field for node in nodes})
            else:
                self.Z_field = Z_field
            self.nodes = nodes
            self.index = I
            self.registers = {node: Reg(self.index) for node in nodes}
            #Each register corresponds to the sub Hilbert space of one qubit
    
            self.weights = weights


    def hamiltonian(self):
        I = self.index
        Z = ket(I[0]) * bra(I[0]) - ket(I[1]) * bra(I[1])
        X = ket(I[0]) * bra(I[0]) - ket(I[1]) * bra(I[1])


        ListofFactors = [(self.weights[w]*Z|self.registers[w[0]])*(self.weights[w]*Z|self.registers[w[1]]) for w in self.weights]
        + [(self.Z_field[node]*Z) for node in self.Z_field] + [(self.X_field[node]*Z) for node in self.X_field]
        #This list saves all factors of the Ising model, takes very long

        self.matrix = reduce(lambda x, y: x+y , ListofFactors)
        #The outside field is zero

        return(self.matrix.mat)

    def partition_function(self, beta, accuracy = 10, concrete = False):

        I = self.index

        #The projectors P_00 and A_00
        P0 = ket(I[0]) * bra(I[0])
        P1 = ket(I[1]) * bra(I[1])

        #Hadamard using Bool Variables copied from \DiracWMC\experiments\quantum_ising\converter.py
        def hadamard(index: Index) -> WCNFMatrix:
            """ Returns a Hadamard matrix on the given index """
            x, y, r = BoolVar(), BoolVar(), BoolVar()
            # r <-> (x AND y)
            cnf = CNF([
                [-r, x],
                [-r, y],
                [r, -x, -y],
            ])
            weight_func = WeightFunction([x, y, r])
            weight_func.fill(1.0)
            weight_func[r, 0] = 1.0 / np.sqrt(2)
            weight_func[r, 1] = -1.0 / np.sqrt(2)
            x, y = LogVarRep(2, [x]), LogVarRep(2, [y])
            return WCNFMatrix(index, cnf, weight_func, [x], [y])
        

        def Zexp(theta, reg1, reg2):
            """Computes exp(theta Z_reg1  otimes theta Z_reg2)"""

            

            """exp(theta Z_reg1  otimes theta Z_reg2) = diag(exp(theta), exp(-theta), exp(-theta),exp(theta))
            """
            return (
                 (
                    (np.exp(theta ) *P0 | reg1) * (P0 | reg2) +
                    (np.exp(theta ) *P1 | reg1) * (P1 | reg2)
                )
                +
                 (
                    (np.exp(-theta ) *P0 | reg1) * (P1 | reg2) +
                    (np.exp(-theta ) *P1 | reg1) * (P0 | reg2)
                )
            )

        ListofFactors = []
        for w in self.weights:
            ListofFactors.append(Zexp(self.weights[w]*beta,self.registers[w[0]] , self.registers[w[1]]))
        #This list saves all factors of the Ising model

        for node in self.Z_field:
            ListofFactors.append((np.exp(self.Z_field[node]*beta)*P0 + np.exp(-self.Z_field[node]*beta)*P1 )|self.registers[node])
            print(" adding Z_field", value(ListofFactors[-1]))

        ListofFactors.extend([hadamard(I)|self.registers[node] for node in self.nodes])
        for node in self.X_field:
            ListofFactors.append((np.exp(self.X_field[node]*beta)*P0 + np.exp(-self.X_field[node]*beta)*P1 )|self.registers[node])

        ListofFactors.extend([hadamard(I)|self.registers[node] for node in self.nodes])

        Factorinlimit = reduce(lambda x, y: x*y , ListofFactors).mat
        self.matrix = Factorinlimit
        for i in range(accuracy):
            self.matrix = self.matrix * Factorinlimit
        


        return(self.matrix.trace_formula())


        
        



def LatticeTransIsing(dimensions, size, weights = 1, dimsubspace = 2, Z_field = 0, X_field = 0):
    """ This is currently slow ( Error for size >= 3)
        dimension: Dimension of the Lattice, 
        size: length of one side of the lattice, 
        weights: The weights accepts dict and int(Standard lattice with homogeneus weights = weights)
        dimsubspace: dimension of subspace of Each Quantum Dot, equal to 2 in standard
        Z_field: The external field in Z direction, dictionary or float for homogenous field
        X_fiedl: The external field in X direction, dictionary or float for homogenous field"""
    I = Index(dimsubspace)
    if isinstance(weights, int) or isinstance(weights, float):
        nodes = list(product(range(size), repeat=dimensions))


        dicweights = {
            (node,node[:i] + (((node[i] + 1) % size),) + node[i+1:]): weights
            for node in nodes
            for i in range(dimensions)}
    else:
        dicweights = weights
        nodes = range(size**dimensions)

    return TransIsingModel(nodes = nodes, weights = dicweights, I = I, X_field = X_field, Z_field = Z_field)









