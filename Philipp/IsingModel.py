

from wcnf_matrix import *
import numpy as np
from functools import reduce
from itertools import product
import networkx as nx

import matplotlib.pyplot as plt


class IsingModel:

    def __init__(self,  weights, I , nodes):
        """ size:    number of nodes in the graph
            weights: a dictionary (node_1,node_2) -> edge 
            I:       the Index of the matrix"""
        self.nodes = nodes

        self.registers = {node: Reg(I) for node in nodes}
        #Each register corresponds to the sub Hilbert space of one qubit

        self.weights = weights
        self.index = I


    def hamiltonian(self):
        I = self.index
        Z = ket(I[0]) * bra(I[0]) - ket(I[1]) * bra(I[1])


        ListofFactors = [(self.weights[w]*Z|self.registers[w[0]])*(self.weights[w]*Z|self.registers[w[1]]) for w in self.weights]
        #This list saves all factors of the Ising model, takes very long

        self.matrix = reduce(lambda x, y: x+y , ListofFactors)
        #The outside field is zero

        return(self.matrix)

    def partition_function(self, beta, concrete = False):

        I = self.index

        def Zexp(theta, reg1, reg2):
            """Computes exp(theta Z_reg1  otimes theta Z_reg2)"""

            P0 = ket(I[0]) * bra(I[0])
            P1 = ket(I[1]) * bra(I[1])

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

        self.matrix = reduce(lambda x, y: x*y , ListofFactors)
        #The outside field is zero

        return(self.matrix.mat.trace_formula())


        
        


def LatticeIsing(dimensions, size, weights = 1, dimsubspace = 2):
    """ dimension: Dimension of the Lattice, 
        size: length of one side of the lattice, 
        weights: The weights accepts dict and int(Standard lattice with homogeneus weights = weights)
        dimsubspace: dimension of subspace of Each Quantum Dot, equal to 2 in standard"""
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

    return IsingModel(nodes = nodes, weights = dicweights, I = I)









