from RBIM_model import RandomBondIsingModel
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import time

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder,)

def basic_RBIM_experiment(size):
    BETA = 1.0

    model = RandomBondIsingModel.generate_square_lattice(size) # Use same model for all solvers

    tstart_raw = time.time()
    true_value = model.partition_function(BETA)
    tend_raw = time.time()

    print(f"Direct calculation time: {(tend_raw - tstart_raw)}")

    problem = model.get_wcnf_matrix(BETA).trace_formula()

    for solver in SOLVERS:
        model_counter = solver()
        print(f"\nSquare lattice {solver.__name__}:")
        
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
        print(f"({size}, {runtime}, {error})", end=" ", flush=True)
            

if __name__ == "__main__":

    size = 3
    basic_RBIM_experiment(size)  
