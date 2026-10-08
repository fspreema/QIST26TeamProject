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
    filename = r"MaxLength-16_Average-16_2026-10-08 03-26-56.csv"
    filename = r"DPMC_MaxLength-16_Average-16_2026-10-08 12-08-18.csv"
 
    full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), save_path, filename)

    df = pd.read_csv(full_path)

    sizes = df["Size"]
    SC_runtimes = df["SC_runtimes"]
    SC_stds = df["SC_stds"]
    SO_runtimes = df["SO_runtimes"]
    SO_stds = df["SO_stds"]
    ST_runtimes = df["ST_runtimes"]
    ST_stds = df["ST_stds"]
    TT_runtimes = df["TT_runtimes"]
    TT_stds = df["TT_stds"]

    # plot results
    fig, ax = plt.subplots()
    #ax.plot(sizes, SC_runtimes, color = "tab:green", marker = "o", linestyle = "-", label = r"Square - cylinder ($L_x \times 3$)")
    ax.errorbar(sizes, SC_runtimes, SC_stds, color = "tab:green", marker = "o", linestyle = "-", label = r"Square - cylinder ($L_x \times 3$)")
    ax.errorbar(sizes, SO_runtimes, SO_stds, color = "tab:blue", marker = "o", linestyle = "-", label = r"Square - open ($L_x \times L_x$)")
    ax.errorbar(sizes, ST_runtimes, ST_stds, color = "tab:orange", marker = "s", linestyle = "-", markersize = 9, markerfacecolor = "none", zorder = 3, label = r"Square - torus ($L_x \times L_x$)")
    ax.errorbar(sizes, TT_runtimes, TT_stds, color = "tab:red", marker = "^", linestyle = "--", markersize = 5, zorder = 4, label = r"Triangular - torus ($L_x \times L_x$)")
    ax.set_xlabel("Length $L_x$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime for Z")
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.splitext(full_path)[0] + "_plotted" + ".pdf")
    plt.show()



