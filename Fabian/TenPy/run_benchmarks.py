from BenchmarkRunner import BenchmarkRunner

if __name__ == "__main__":

    # Define fixed values of length and ferromagnetic order
    p = 0.5

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
        model_params_cylinder_long = {
            "lattice": "Square",
            "Lx": curr_l,
            "Ly": 3,
            "p": p,
            "bc_x": "periodic",
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
        sampled_model_params = [model_params_square_o, model_params_torus, 
                                model_params_cylinder_long, model_params_triangle_torus]

        # Init Benchmarking
        BenchmarkRunner(model_configs= sampled_model_params).run()