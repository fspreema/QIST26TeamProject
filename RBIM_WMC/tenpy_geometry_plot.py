from Tenpy_code.RandomBondIsing import RandomBondIsing as TenpyRBI
from datetime import datetime
from RBIM_model import RandomBondIsingModel, BaseModel
from BenchMark import BenchMark
from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    save_path = r"Results\Tenpy_model"

    filename = r"TensorOrder_MaxLength-16_Average-16_2026-10-09 01-03-45.csv"
    filename = r"DPMC_MaxLength-16_Average-16_2026-10-09 02-16-46.csv"
    #filename = r"Cachet_MaxLength-16_Average-16_2026-10-09 02-51-16.csv"
    
    use_old_cilinder_scaling = False
    plot_errorbars = False

    # =============================================

    full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), save_path, filename)

    df = pd.read_csv(full_path)

    sizes = df["Size"]


    SC_spins = [((s*s)//3) * 3 for s in sizes]
    if use_old_cilinder_scaling:
        SC_spins = [3*s for s in sizes]
    SO_spins = [s*s for s in sizes]
    ST_spins = [s*s for s in sizes]
    TT_spins = [s*s for s in sizes]

    if (False):
        SC_connections = []
        for s in sizes:
            model_params = {
                "lattice": "Square",
                "p": 0.5,
                "bc_x": "periodic",
                "bc_y": "open",
            }
            model_params["Lx"] = 3
            model_params["Ly"] = s
            tenpy_model = TenpyRBI(model_params)
            couplings = tenpy_model.coupling_terms['Sigmaz_i Sigmaz_j']
            num_couplings = len(couplings.to_TermList().terms)
            SC_connections.append(num_couplings)

        SO_connections = []
        for s in sizes:
            model_params = {
                "lattice": "Square",
                "p": 0.5,
                "bc_x": "open",
                "bc_y": "open",
            }
            model_params["Lx"] = s
            model_params["Ly"] = s
            tenpy_model = TenpyRBI(model_params)
            couplings = tenpy_model.coupling_terms['Sigmaz_i Sigmaz_j']
            num_couplings = len(couplings.to_TermList().terms)
            SO_connections.append(num_couplings)


        ST_connections = []
        for s in sizes:
            model_params = {
                "lattice": "Square",
                "p": 0.5,
                "bc_x": "periodic",
                "bc_y": "periodic",
            }
            model_params["Lx"] = s
            model_params["Ly"] = s
            tenpy_model = TenpyRBI(model_params)
            couplings = tenpy_model.coupling_terms['Sigmaz_i Sigmaz_j']
            num_couplings = len(couplings.to_TermList().terms)
            ST_connections.append(num_couplings)

        TT_connections = []
        for s in sizes:
            model_params = {
                "lattice": "Triangular",
                "p": 0.5,
                "bc_x": "periodic",
                "bc_y": "periodic",
            }
            model_params["Lx"] = s
            model_params["Ly"] = s
            tenpy_model = TenpyRBI(model_params)
            couplings = tenpy_model.coupling_terms['Sigmaz_i Sigmaz_j']
            num_couplings = len(couplings.to_TermList().terms)
            TT_connections.append(num_couplings)

    
    SC_runtimes = df["SC_runtimes"]
    SC_stds = df["SC_stds"]
    SO_runtimes = df["SO_runtimes"]
    SO_stds = df["SO_stds"]
    ST_runtimes = df["ST_runtimes"]
    ST_stds = df["ST_stds"]
    TT_runtimes = df["TT_runtimes"]
    TT_stds = df["TT_stds"]

    # plot results
    fig, ax = plt.subplots(figsize=(6, 5))
    if plot_errorbars:
        ax.errorbar(SC_spins, SC_runtimes, SC_stds, capsize=4, color = "tab:green", marker = "o", linestyle = "-", label = r"Square - cylinder ($L_x \times 3$)")
        ax.errorbar(SO_spins, SO_runtimes, SO_stds, capsize=4, color = "tab:blue", marker = "o", linestyle = "-", label = r"Square - open ($L_x \times L_x$)")
        ax.errorbar(ST_spins, ST_runtimes, ST_stds, capsize=4, color = "tab:orange", marker = "s", linestyle = "-", markersize = 9, markerfacecolor = "none", zorder = 3, label = r"Square - torus ($L_x \times L_x$)")
        ax.errorbar(TT_spins, TT_runtimes, TT_stds, capsize=4, color = "tab:red", marker = "^", linestyle = "--", markersize = 5, zorder = -1, label = r"Triangular - torus ($L_x \times L_x$)")
    else:
        ax.plot(SC_spins, SC_runtimes, color = "tab:green", marker = "o", linestyle = "-", label = r"Square - cylinder ($L_x \times 3$)")
        ax.plot(SO_spins, SO_runtimes, color = "tab:blue", marker = "o", linestyle = "-", label = r"Square - open ($L_x \times L_x$)")
        ax.plot(ST_spins, ST_runtimes, color = "tab:orange", marker = "s", linestyle = "-", markersize = 9, markerfacecolor = "none", zorder = 3, label = r"Square - torus ($L_x \times L_x$)")
        ax.plot(TT_spins, TT_runtimes, color = "tab:red", marker = "^", linestyle = "--", markersize = 5, zorder = -1, label = r"Triangular - torus ($L_x \times L_x$)")
    
    plt.title(r"Runtime for $Z$ using DPMC")
    ax.set_xlabel(r"Total number of spins $N=L_x L_y$")
    ax.set_ylabel("Mean runtime [s]")
    ax.set_yscale("log")
    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.1)
    ax.legend(fontsize=8, loc = "lower right")
    ax.set_ylim(10**-2, 2.5*10**0)
    ax.set_xlim(left = 0) #, right = 55)
    plt.tight_layout()
    plt.savefig(os.path.splitext(full_path)[0] + "_plotted" + ".pdf")
    plt.show()



