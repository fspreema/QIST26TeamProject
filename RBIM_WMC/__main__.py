from RBIM_model import RandomBondIsingModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import time
import numpy as np

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder,)  


if __name__ == "__main__":

    # observable_test(2)

    bm = BenchMark(RandomBondIsingModel, DPMC)

    L = 2
    betas = np.linspace(0.0, 3.0, 10)

    start_1 = time.perf_counter()
    bm.magnetization_vs_beta_benchMark(shape=(L, L), betas=betas, p_ferro=1.0)
    end_1 = time.perf_counter()
    time_mag = end_1 - start_1

    start_2 = time.perf_counter()
    bm.magnetization_squared_vs_beta_benchMark(shape=(L, L), betas=betas, p_ferro=1.0)
    end_2 = time.perf_counter()
    time_mag2 = end_2 - start_2

    print("\n" + "="*40)
    print("        Results        ")
    print("="*40)
    print(f"Time Magnetizaiton:   {time_mag:.4f} seconds")
    print(f"Time Magnetization Squared:   {time_mag2:.4f} seconds")

