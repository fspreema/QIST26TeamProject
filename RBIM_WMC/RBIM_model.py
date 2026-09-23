from wcnf_matrix import *
import numpy as np
import matplotlib.pyplot as plt
import math

SQRT2 = math.sqrt(2)

class RandomBondIsingModel:
    x = 69

    def print_it(self):
        print(self.x)
    
    def exp_Z(theta: float, I: Index) -> WCNFMatrix[float]: # taken from DiracWMC package
        x = BoolVar()

        cnf = CNF()
        weight_func = WeightFunction([x], weights={x: (np.exp(theta), np.exp(-theta))})
        exponentiated =  WCNFMatrix(I, cnf, weight_func, [LogVarRep(2, [x])],[LogVarRep(2, [x])])
        return exponentiated

    def exp_zz_rotation(theta: float, I: Index) -> WCNFMatrix[float]:  # taken from DiracWMC package
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
        return WCNFMatrix(I, cnf, weight_func, [x, y], [x, y])

    def hadamard(index: Index) -> WCNFMatrix:  # taken from DiracWMC package
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