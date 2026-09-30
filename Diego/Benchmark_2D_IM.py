import math
import time
import csv
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from wcnf_matrix import *
from functools import reduce
from numpy import exp

def calc_pairs(n_spins, do_periodic_boundary_condition, _1D_ising_model):
    pairs = []
    if _1D_ising_model:
        for i in range(n_spins - 1):
            pairs.append((i, i + 1))
        if do_periodic_boundary_condition:
            pairs.append((0,n_spins - 1))
    else:
        for i in range(n_spins):
            for j in range(n_spins):
                index = i * n_spins + j
                if j < n_spins - 1:
                    right_index = i * n_spins + (j + 1)
                    pairs.append((index, right_index))
                elif do_periodic_boundary_condition:
                    right_index = i * n_spins
                    pairs.append((index, right_index))

                if i < n_spins - 1:
                    down_index = (i + 1) * n_spins + j
                    pairs.append((index, down_index))
                elif do_periodic_boundary_condition:
                    down_index = j
                    pairs.append((index, down_index))

    # we order the pairs so the first index is always smaller than the second one
    pairs = [(min(i, j), max(i, j)) for i, j in pairs]
    return list(set(pairs))

# Creation of the spin space and the P0, P1 and Id matrices

def build_partition_function (n_spins, pairs, J = 1.5, h = 0.5, beta = 1.0, boundary_condition = True, Dimension = False):
    local_space = Index (2)

    # P0 = (1 , 0)
    #      (0 , 0)
    P0 = ket ( local_space [0]) * bra ( local_space [0])

    # P1 = (0 , 0)
    #      (0 , 1)
    P1 = ket ( local_space [1]) * bra ( local_space [1])

    # Id = (1 , 0)
    #      (0 , 1)
    Id = ket ( local_space [0]) * bra ( local_space [0]) +   ket ( local_space [1]) * bra ( local_space [1])

    #  Qp = P0 ** P0 + P1 ** P1
    #  Qm = P0 ** P1 + P1 ** P0
    #  Z  = Qp - Qm
    #  Id = Qp + Qm
    if Dimension == True:
        N = n_spins
    elif Dimension == False:
        N = n_spins**2
    
    spins = [Reg(local_space) for _ in range(N)]

    # Preparation of the Identity Matrix
    id_factors = []
    for i in range(N):
        id_factors.append(Id | spins[i])
    EXPONENTIAl = reduce(lambda x, y: x * y, id_factors)

    # Identity2 = Id
    # for i in range(1,n_spins):
    #     Identity2 = Identity2 ** Id

    #Identity3 = WCNFMatrix.identity(local_space, n_spins)

    # Precompute the exponentials

    exp_pJbeta = exp( J * beta)
    exp_mJbeta = exp(-J * beta)
    exp_phbeta = exp( h * beta)
    exp_mhbeta = exp(-h * beta)



    for i,j in pairs:
        term_P0P0 = ((exp_pJbeta * P0) | spins[i]) * (P0 | spins[j])
        term_P1P1 = ((exp_pJbeta * P1) | spins[i]) * (P1 | spins[j])

        term_P0P1 = ((exp_mJbeta * P0) | spins[i]) * (P1 | spins[j])
        term_P1P0 = ((exp_mJbeta * P1) | spins[i]) * (P0 | spins[j])

        exp_JZZ_ij = term_P0P0 + term_P1P1 + term_P0P1 + term_P1P0
        EXPONENTIAl = EXPONENTIAl * exp_JZZ_ij


    exp_hZ_i = exp_phbeta * P0 + exp_mhbeta * P1

    for i in range(N):
        EXPONENTIAl = EXPONENTIAl * (exp_hZ_i | spins[i])

    return EXPONENTIAl

def run_ising(spin_range, solvers, dim, J, h, beta):
    """
        Computes the trace of the partition function formula using the diferent solvers.
        As outputs gives the values of the partition function and the time 
    """
    
    timings = {name: [] for name in solvers}
    z_records = {name: [] for name in solvers}

    time_headers = " | ".join([f"Time {name:<10}" for name in solvers])

    print(f"{'L':<4} | {'N':<4} | {time_headers}")
    print("-" * (14 + 18 * len(solvers)))

    for L in spin_range:
        n_spins = L**2
        pairs = calc_pairs(L, False, False)
        
        logical_formula = build_partition_function(n_spins, pairs, J, h, beta)
        cnf, weight_function = logical_formula.mat.trace_formula()

        row_output = f"{L:<4} | {n_spins:<4} | "
        time_strs = []

        # Run each solver independently for the same CNF formula
        for name, solver_cls in solvers.items():
            set_model_counter(solver_cls)
            try:
                t0 = time.perf_counter()
                computed_z = weight_function(cnf)
                elapsed = time.perf_counter() - t0

                timings[name].append(elapsed)
                z_records[name].append(computed_z)

                time_strs.append(f"{elapsed:<15.4f}")
            except Exception as e:
                # Capture solver timeout, out-of-memory, or parse failures gracefully
                timings[name].append(None)
                z_records[name].append(None)
                time_strs.append(f"{'FAILED':<15}")

        row_output += " | ".join(time_strs)
        print(row_output)

    print("=" * (7 + 18 * len(solvers) * 2) + "\n")
    return timings, z_records

def save_and_plot(L_vals, dim, timings, solvers, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Number of total spins for each L
    n_spins = [L ** dim for L in L_vals]
    
    # Save raw timing data
    with open(f"{output_dir}/benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "N"] + list(solvers.keys()))
        for idx in range(len(L_vals)):
            writer.writerow([L_vals[idx], n_spins[idx]] + [timings[s][idx] for s in solvers])

    box_style = dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', alpha=0.8, edgecolor='silver')

    # ==== Plot 1: Time vs L (Side Lenght) ====
    plt.figure(figsize=(9, 5.5))
    for name in solvers:
        valid_points = [(L, t) for L, t in zip(L_vals, timings[name]) if t is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, marker='o', linewidth=1.8, label=name)

    plt.text(0.05, 0.95, f"{dim}D Lattice", transform=plt.gca().transAxes, 
             fontsize=11, fontweight='bold', verticalalignment='top', bbox=box_style)
    plt.xlabel("Linear Size $L$")
    plt.ylabel("Runtime ($s$)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/runtime_vs_L.png", dpi=300)
    plt.close()

    # ==== Plot 2: Time vs N (Total Spins) ====
    plt.figure(figsize=(9, 5.5))
    for name in solvers:
        valid_points = [(N, t) for N, t in zip(n_spins, timings[name]) if t is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, marker='o', linewidth=1.8, label=name)

    plt.text(0.05, 0.95, f"{dim}D Lattice", transform=plt.gca().transAxes, 
             fontsize=11, fontweight='bold', verticalalignment='top', bbox=box_style)
    plt.xlabel("Total number of spins")
    plt.ylabel("Runtime ($s$)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/runtime_vs_N.png", dpi=300)
    plt.close()

    print(f"\nSaved plots 'scaling_vs_L.png' and 'scaling_vs_N.png' to '{output_dir}'")

if __name__ == "__main__":

    active_solvers = {
        "Cachet": Cachet,
        "DPMC": DPMC,
        "TensorOrder": TensorOrder
    }
    
    J = 1.5
    h = 0.5
    beta = 1.0
    dim = 2
    L = [2, 3, 4, 5, 6, 7, 8]

    Output_dir = f"../Iago/Output/Ising_{dim}D/Run_3"

    timings, z_vals = run_ising(L, active_solvers, dim, J, h, beta)
    save_and_plot(L, dim, timings, active_solvers, Output_dir)