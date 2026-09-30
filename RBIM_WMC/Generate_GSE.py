from RBIM_model import RandomBondIsingModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import time
import numpy as np

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder,)


if __name__ == "__main__":
    
    L_list = [L for L in range(2, 6)]

    for solver in SOLVERS:
        bm = BenchMark(RandomBondIsingModel, solver)
        for L in L_list:
            shape = (L, L)
            bm.Generate_GoundStateEnergy(shape, average = 5)