# import numpy as np
import argparse
from pathlib import Path

import numpy as np

from Fabian.TenPy.AccuracyBenchmarkRunner import AccuracyBenchmarkRunner
from Fabian.TenPy.RuntimeBenchmarkRunner import RuntimeBenchmarkRunner

RESULTS_DIR = Path(__file__).resolve().parent / "Results"

def make_model_config(lx, ly, lattice="Square", bc_x="open", bc_y="open"):
    return {
        "lattice": lattice,
        "Lx": lx,
        "Ly": ly,
        "p": 0.5, # Define fixed values of length and ferromagnetic order
        "bc_x": bc_x,
        "bc_y": bc_y,
        "bc_MPS": "finite",
    }

def runtime_benchmarks(repeats):

    #### RUNTIME AND TRUNCATION ####
    
    for curr_l in range(3,16):

        sampled_model_params = [make_model_config(curr_l, 3, "Square", "periodic", "open")]
        
        # Init Benchmarking
        RuntimeBenchmarkRunner(model_configs= sampled_model_params,
                        chi_limit= 600,
                        repeats= repeats,
                        ).run(file_name=str(RESULTS_DIR / "runtime_benchmarks.csv"))

    for curr_l in range(3,7):

        # Init Model params
        model_params_square_o = make_model_config(curr_l, curr_l, "Square", "open", "open")
        model_params_torus = make_model_config(curr_l, curr_l, "Square", "periodic", "periodic")
        model_params_triangle_torus = make_model_config(curr_l, curr_l, "Triangular", "periodic", "periodic")

        sampled_model_params = [model_params_square_o, model_params_torus, model_params_triangle_torus]

        # Init Benchmarking
        RuntimeBenchmarkRunner(model_configs= sampled_model_params,
                        chi_limit= 600,
                        repeats= repeats,
                        ).run(file_name=str(RESULTS_DIR / "runtime_benchmarks.csv"))

def accuracy_benchmarks(repeats):

    #### RUNTIME AND TRUNCATION ####
    
    for curr_l in range(3,7):

        sampled_model_params = [make_model_config(curr_l, 3, "Square", "periodic", "open")]
        
        # Init Benchmarking
        AccuracyBenchmarkRunner(model_configs= sampled_model_params,
                        chi_range= [100, 200, 400, 600],
                        repeats= repeats,
                        ).run(file_name=str(RESULTS_DIR / "accuracy_benchmarks.csv"))

    for curr_l in range(3,6):

        # Init Model params
        model_params_square_o = make_model_config(curr_l, curr_l, "Square", "open", "open")
        model_params_torus = make_model_config(curr_l, curr_l, "Square", "periodic", "periodic")
        model_params_triangle_torus = make_model_config(curr_l, curr_l, "Triangular", "periodic", "periodic")

        sampled_model_params = [model_params_square_o, model_params_torus, model_params_triangle_torus]

        # Init Benchmarking
        AccuracyBenchmarkRunner(model_configs= sampled_model_params,
                        chi_range= [100, 200, 400, 600],
                        repeats= repeats,
                        ).run(file_name=str(RESULTS_DIR / "accuracy_benchmarks.csv"))


def runtime_beta_dependence(repeats):
     
    ### RUNTIME FIXED LENGTH DEPENDENCE ON BETA ###
    # Define fixed values of length and ferromagnetic order
    
    l = 3
    for curr_beta in np.arange(0.5,5.5,0.5):

        sampled_model_params = [make_model_config(l, 3, "Square", "periodic", "open")]
        
        # Init Benchmarking
        RuntimeBenchmarkRunner(model_configs= sampled_model_params,
                chi_limit= 600, beta_max= curr_beta, repeats= repeats
                ).run(file_name=str(RESULTS_DIR / "runtime_beta_dependence.csv"))


def main():
    parser = argparse.ArgumentParser(description="Run Ising benchmarks.")

    parser.add_argument(
        "--benchmark",
        choices=["runtime", "accuracy", "beta", "all"],
        default="runtime",
        help="Benchmark to run (default: runtime).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions (default: 3).",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = {
        "runtime": runtime_benchmarks,
        "accuracy": accuracy_benchmarks,
        "beta": runtime_beta_dependence,
    }

    if args.benchmark == "all":
        for benchmark in benchmarks.values():
            benchmark(args.repeats)
    else:
        benchmarks[args.benchmark](args.repeats)

if __name__ == "__main__":
    main()