import numpy as np
from datetime import datetime
from BenchmarkRunner import BenchmarkRunner

if __name__ == "__main__":

    #### RUNTIME AND TRUNCATION ####

    # Define fixed values of length and ferromagnetic order
    p = 0.5
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f" It is {time}")
    #Seedlist, with three different seeds
    seed_list = [range(0,10),range(10,20), range(20,30)]
    for seeds in seed_list:
        
        
        for curr_l in range(3,16):

            model_params_cylinder_long = {
            "lattice": "Square",
            "Lx": curr_l,
            "Ly": 3,
            "p": p,
            "bc_x": "periodic",
            "bc_y": "open",
            "bc_MPS": "finite",
            #"chi_max": chi_max
            "seeds" : seeds
            }

            

            sampled_model_params = [model_params_cylinder_long]
            
            # Init Benchmarking
            BenchmarkRunner(model_configs= sampled_model_params, seeds = seeds ).run(file_name= f"benchmark_results_chi_max=100 {time}")
    """for curr_l in range(3,7):
    
        for seeds in seed_list:

            # Init Model params
            model_params_square_o = {
                "lattice": "Square",
                "Lx": curr_l,
                "Ly": curr_l,
                "p": p,
                "bc_x": "open",
                "bc_y": "open",
                "bc_MPS": "finite",
                #"chi_max": chi_max
                "seeds" : seeds
            }
            model_params_torus = {
                "lattice": "Square",
                "Lx": curr_l,
                "Ly": curr_l,
                "p": p,
                "bc_x": "periodic",
                "bc_y": "periodic",
                "bc_MPS": "finite",
                #"chi_max": chi_max
                "seeds" : seeds
            }
            model_params_triangle_torus = {
                "lattice": "Triangular",
                "Lx": curr_l,
                "Ly": curr_l,
                "p": p,
                "bc_x": "periodic",
                "bc_y": "periodic",
                "bc_MPS": "finite",
                #"chi_max": chi_max
                "seeds" : seeds
            }

            sampled_model_params = [model_params_square_o, model_params_torus, model_params_triangle_torus]

            # Init Benchmarking
            BenchmarkRunner(model_configs= sampled_model_params).run(file_name= f"benchmark_results_chi_max=100 {time}")

    ### RUNTIME FIXED LENGTH DEPENDENCE ON BETA ###

    # Define fixed values of length and ferromagnetic order

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
        #"chi_max": chi_max
        "seeds" : seeds
        }

        sampled_model_params = [model_params_cylinder_long]
        
        # Init Benchmarking
        BenchmarkRunner(model_configs= sampled_model_params, beta_max= curr_beta).run(file_name= f"benchmark_results_chi_max=100 {time}")"""

