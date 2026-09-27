import logging

import matplotlib.pyplot as plt
import numpy as np
from purification import BenchmarkModel
from rndm_bnd_ising import RandomBondIsing

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Init Run independent variables
    data = {}
    p_values = np.linspace(0, 1, 10)
    l = 4

    for curr_p in p_values:
        print(f"Running p = {curr_p:.2f}", flush=True)

        # Init Model params and Class
        model_params = {
            "lattice": "Square",
            "Lx": l,
            "Ly": l,
            "p": curr_p,
            "bc_x": "open",
            "bc_y": "open",
            "bc_MPS": "finite",
        }
        curr_model = RandomBondIsing(model_params)

        # Run Benchmark on current p
        benchmark = BenchmarkModel(
            model=curr_model,
            measure_obs="all",
            beta_max=7,
        )
        data[curr_p] = benchmark.run()


    # Rows correspond to p; columns correspond to beta.
    betas = np.asarray(data[p_values[0]]["beta"])

    energy = np.asarray([
        [m["Energy_per_spin"] for m in data[p]["all"]]
        for p in p_values
    ])

    mz_squared = np.asarray([
        [m["Mz_squared"] for m in data[p]["all"]]
        for p in p_values
    ])

    # Coordinates for the two surface plots.
    P, BETA = np.meshgrid(p_values, betas, indexing="ij")

    fig = plt.figure(figsize=(14, 6))

    # Plotting Energy
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(P, BETA, energy, cmap="viridis")
    ax.set_xlabel("p")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel(r"$\langle H\rangle/N$")
    ax.set_title("Energy per spin")

    # Plottign mz ** 2
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(P, BETA, mz_squared, cmap="plasma")
    ax.set_xlabel("p")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel(r"$\langle (M_z/N)^2\rangle$")
    ax.set_title("Squared magnetization per spin")
    plt.tight_layout()
    plt.show()