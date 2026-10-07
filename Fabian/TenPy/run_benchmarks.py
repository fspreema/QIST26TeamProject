# import numpy as np
from BenchmarkRunner import BenchmarkRunner

if __name__ == "__main__":

    #### RUNTIME AND TRUNCATION ####

    # Define fixed values of length and ferromagnetic order
    p = 0.5

    for curr_l in range(3,16):

        model_params_cylinder_long = {
        "lattice": "Square",
        "Lx": curr_l,
        "Ly": 3,
        "p": p,
        "bc_x": "periodic",
        "bc_y": "open",
        "bc_MPS": "finite",
        }

        sampled_model_params = [model_params_cylinder_long]
        
        # Init Benchmarking
        BenchmarkRunner(model_configs= sampled_model_params,
                        chi_limit= 600,
                        repeats= 3,
                        ).run(file_name= "benchmark_results_600.csv")

    for curr_l in range(3,7):

        # Init Model params
        model_params_square_o = {
            "lattice": "Square",
            "Lx": curr_l,
            "Ly": curr_l,
            "p": p,
            "bc_x": "open",
            "bc_y": "open",
            "bc_MPS": "finite",
        }
        model_params_torus = {
            "lattice": "Square",
            "Lx": curr_l,
            "Ly": curr_l,
            "p": p,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "bc_MPS": "finite",
        }
        model_params_triangle_torus = {
            "lattice": "Triangular",
            "Lx": curr_l,
            "Ly": curr_l,
            "p": p,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "bc_MPS": "finite",
        }

        sampled_model_params = [model_params_square_o, model_params_torus, model_params_triangle_torus]

        # Init Benchmarking
        BenchmarkRunner(model_configs= sampled_model_params,
                        chi_limit= 600,
                        repeats= 3,
                        ).run(file_name= "benchmark_results_600.csv")

    ### RUNTIME FIXED LENGTH DEPENDENCE ON BETA ###
    # Define fixed values of length and ferromagnetic order
    
    """
    l = 3
    p = 0.5
    for curr_beta in np.arange(0.5,5.5,0.5):

        model_params_cylinder_long = {
        "lattice": "Square",
        "Lx": l,
        "Ly": 3,
        "p": p,
        "bc_x": "open",
        "bc_y": "open",
        "bc_MPS": "finite",
        }

        sampled_model_params = [model_params_cylinder_long]
        
        # Init Benchmarking
        BenchmarkRunner(model_configs= sampled_model_params,
                chi_limit= 1_000,
                ).run(file_name= "benchmark_results_beta.csv")
    """
