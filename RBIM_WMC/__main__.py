from RBIM_model import RandomBondIsingModel, BaseModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder,)

def p_vs_runtime(shape: tuple[int,int] = (6,6), solver: ModelCounter = Cachet):
    bm = BenchMark(RandomBondIsingModel, solver)

    p_values = np.linspace(0, 1, 21)
    bm.runtime_vs_p_benchMark(shape, p_values, average = 50, save = True)


if __name__ == "__main__":

    # observable_test(2)

    #bm = BenchMark(RandomBondIsingModel, DPMC)

    # L = 2
    # betas = np.linspace(0.0, 3.0, 10)

    # start_1 = time.perf_counter()
    # bm.magnetization_vs_beta_benchMark(shape=(L, L), betas=betas, p_ferro=1.0)
    # end_1 = time.perf_counter()
    # time_mag = end_1 - start_1

    # start_2 = time.perf_counter()
    # bm.magnetization_squared_vs_beta_benchMark(shape=(L, L), betas=betas, p_ferro=1.0)
    # end_2 = time.perf_counter()
    # time_mag2 = end_2 - start_2

    # print("\n" + "="*40)
    # print("        Results        ")
    # print("="*40)
    # print(f"Time Magnetizaiton:   {time_mag:.4f} seconds")
    # print(f"Time Magnetization Squared:   {time_mag2:.4f} seconds")
    # # sides = np.arange(2,8,2)
    # # shapes = [(int(n),int(n)) for n in sides]
    # # bm.runtime_vs_lattice_benchMark(shapes, average = 4)
    
    # # # calculates error for random bond chain
    # # sides = np.arange(2,12,1)
    # # shapes = [(int(n),1) for n in sides]
    # # bm.error_vs_lattice_benchMark(shapes, average = 32)
    
    # p_vs_runtime((6,6), Cachet)
    
    # print("End of the program")
        
