from RBIM_model import RandomBondIsingModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import time
import numpy as np

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder,)

def basic_RBIM_experiment(size: int):
    BETA = 1.0

    model = RandomBondIsingModel.generate_square_lattice(size) # Use same model for all solvers

    tstart_raw = time.time()
    true_value = model.partition_function(BETA)
    tend_raw = time.time()

    print(f"Direct calculation time: {(tend_raw - tstart_raw)}")

    problem = model.get_wcnf(BETA)

    for solver in SOLVERS:
        model_counter = solver()
        print(f" - Square lattice {solver.__name__}:")
        
        failed = False
        runtime = 0.0
        error = 0.0

        for result in model_counter.batch_model_count(problem):
            if not result.success:
                    failed = True
                    break
            runtime += result.runtime
            error += abs(result.model_count / true_value - 1)
            if failed:
                print("FAILURE")
                break
        print(f"({size}, {runtime}, {error})", end="\n", flush=True)

def observable_test(size : int):
    #model = RandomBondIsingModel.generate_square_lattice(size, p_ferro = 0.5)
    model = RandomBondIsingModel.generate_lattice((1,2), p_ferro = 1)
    model_counter = Cachet()

    for beta in np.logspace(1,5,10):

        problem_b = model.get_wcnf(beta)
        result_b = model_counter.model_count(*problem_b)
        value_b = result_b.model_count

        problem_a = model.get_wcnf_with_operator(beta)
        result_a = model_counter.model_count(*problem_a)
        value_a = result_a.model_count



        print(value_a/value_b)
     

     

if __name__ == "__main__":

    # observable_test(2)

    bm = BenchMark(RandomBondIsingModel, DPMC)

    L = 3

    bm.magnetization_vs_beta_benchMark(shape = (L,L), p_ferro = 1.0)
    bm.magnetization_squared_vs_beta_benchMark(shape = (L,L), p_ferro = 1.0)

