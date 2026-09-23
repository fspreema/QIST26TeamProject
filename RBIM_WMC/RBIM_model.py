from wcnf_matrix import *
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce

SQRT2 = math.sqrt(2)

class RandomBondIsingModel:
    I : Index

    def __init__(self):
        self.I = Index()
    
    def exp_Z(theta: float) -> WCNFMatrix[float]: # taken from DiracWMC package
        x = BoolVar()

        cnf = CNF()
        weight_func = WeightFunction([x], weights={x: (np.exp(theta), np.exp(-theta))})
        exponentiated =  WCNFMatrix(self.I, cnf, weight_func, [LogVarRep(2, [x])],[LogVarRep(2, [x])])
        return exponentiated

    def exp_ZZ(theta: float) -> WCNFMatrix[float]:  # taken from DiracWMC package
        """ Returns a ZZ-rotation matrix with angle theta """
        x, y, z = BoolVar(), BoolVar(), BoolVar()
        # z <-> (x <-> y)
        cnf = CNF([
            [x, -y, -z],
            [x, y, z],
            [-x, y, -z],
            [-x, -y, z],
        ])
        weight_func = WeightFunction([x, y, z])
        weight_func.fill(1.0)
        weight_func[z, 0] = np.exp(-theta)
        weight_func[z, 1] = np.exp(theta)
        x, y = LogVarRep(2, [x]), LogVarRep(2, [y])
        return WCNFMatrix(self.I, cnf, weight_func, [x, y], [x, y])

    def hadamard() -> WCNFMatrix:  # taken from DiracWMC package
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
        weight_func[r, 0] = 1.0 / SQRT2
        weight_func[r, 1] = -1.0 / SQRT2
        x, y = LogVarRep(2, [x]), LogVarRep(2, [y])
        return WCNFMatrix(index, cnf, weight_func, [x], [y])

    def get_random_bond_ising_parameters(length: int, 
                           periodic: bool, 
                           dim: int,
                           p_ferro: float, 
                           seed: int = 42
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        - We set h=0
        - We sample J from a bimodal distrubtation given by:
            J_ij = p * delta(J_ij - 1) + (1-p) * delta(J_ij + 1)
        """ 
        # Init J and rng
        rng = np.random.default_rng(seed)
        num_spins = length ** dim
        j_interactions = np.zeros((num_spins, num_spins))

        # Get pairings for the selectzed lattice
        pairs = get_pairs(length= length, dim= dim, periodic= periodic)

        # J is sampled from a bimodal distribution
        for i,j in pairs:
            if rng.random() < p_ferro:
                j_interactions[i, j] = 1.0
            else:
                j_interactions[i, j] = -1.0

        return j_interactions

    def identity() -> WCNFMatrix:
        """ Returns an identity matrix on the given index """
        x = BoolVar()
        cnf = CNF()
        weight_func = WeightFunction([x])
        weight_func.fill(1.0)
        x = LogVarRep(2, [x])
        return WCNFMatrix(I, cnf, weight_func, [x], [x])

    def get_wcnf(self, n, beta: float = 1.0,
    trotter_layers: int = 5) -> tuple[CNF, WeightFunction]:
        """ Convert the given quantum Ising model to a CNF formula and weight
            function (approximation) using trotterization. Thwe ighted model count
            of the formula w.r.t. the weight function approximates the partition
            function of the quantum Ising model at inverse temperature beta """
        #n = len(model)
        regs = [Reg(index) for _ in range(n)]
        ident = reduce(lambda x, y: x * y, (self.identity(index) | reg for reg in regs))
        # Z-direction
        if len(list(self.get_random_bond_ising_parameters())) == 0:
            interactions = ident
        else:
            interactions = reduce(lambda x, y: x * y, (self.exp_ZZ(beta *strength / trotter_layers, index) | (regs[i], regs[j]) for i, j, strength in self.get_random_bond_ising_parameters()))
            
        external_field_z = reduce(lambda x, y: x * y, (self.exp_Z(beta *model.external_field_z / trotter_layers, index) | reg for reg in regs))
        # Hadamards
        hadamards = reduce(lambda x, y: x * y, (self.hadamard(index) | reg for reg in regs))
        # X-direction
        external_field_x = reduce(lambda x, y: x * y, (self.exp_Z(beta *model.external_field_x / trotter_layers, index) | regs[i] for i in range(n)))
        # Calculate final matrix
        cur = ident
        for _ in range(trotter_layers):
            cur *= (interactions * external_field_z * hadamards * external_field_x *
            hadamards)
        return cur.mat.trace_formula()