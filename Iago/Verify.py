import time
from wcnf_matrix import *

def verify_solver(solver_class, solver_name):
    print(f"\n--- Testing {solver_name} ---")
    try:
        set_model_counter(solver_class)
        
        # Build a trivial 1-spin identity trace
        space = Index(2)
        reg = Reg(space)
        identity = (ket(space[0]) * bra(space[0])) + (ket(space[1]) * bra(space[1]))
        
        cnf, weight_func = (identity | reg).mat.trace_formula()
        
        start = time.time()
        result = weight_func(cnf)
        end = time.time()
        
        print(f"✅ {solver_name} successfully executed in {end - start:.4f}s. Result: {result}")
    except FileNotFoundError:
        print(f"❌ {solver_name} executable not found in PATH.")
    except Exception as e:
        print(f"❌ {solver_name} failed with error: {e}")

if __name__ == "__main__":
    verify_solver(Cachet, "Cachet")
    verify_solver(DPMC, "DPMC")
    verify_solver(TensorOrder, "TensorOrder")