import math
import time
import csv
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from wcnf_matrix import *
from functools import reduce

def generate_grid_pairs(dim, L):
    """
    Generates nearest-neighbor pairs for a hypercube grid.
    dim: Number of dimensions (1 for line, 2 for square, 3 for cube).
    L: Number of spins along each side.
    """

    n_spins = L ** dim
    pairs = []

    # Generate every coordinate using the 'repeat' argument
    coords = list(itertools.product(range(L), repeat=dim))

    # Create a dictionary to map a coordinate like (0, 1) to a flat index like 1
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}

    # Loop through every coordinate to find its neighbours
    for coord in coords:
        idx = coord_to_idx[coord]

        for current_dim in range(dim):
            # Check if there's a neighbour to the right in this dimension
            if coord[current_dim] + 1 < L:
                neighbour_coord = list(coord)
                neighbour_coord[current_dim] += 1
                neighbour_idx = coord_to_idx[tuple(neighbour_coord)]

                # Add connection to our pairs list
                pairs.append((idx, neighbour_idx))

    return pairs, n_spins

def build_partition_function(n_spins, pairs, J, mu_z, mu_x, k, beta):
    """
    Builds the logical formula for the partition function for the TFIM
    """

    spin_space = Index(2)
    p0 = ket(spin_space[0]) * bra(spin_space[0])
    p1 = ket(spin_space[1]) * bra(spin_space[1])
    p2 = ket(spin_space[0]) * bra(spin_space[1])
    p3 = ket(spin_space[1]) * bra(spin_space[0])
    spins = [Reg(spin_space) for i in range(n_spins)]
    # spins_tuple = tuple(spins)

    identity_matrix = WCNFMatrix.identity(spin_space,n_spins)

    # Start with a global identity states
    Z_operator = identity_matrix | spins
    X_prime_operator = identity_matrix | spins
    H_global_operator = identity_matrix | spins

    # === Construct Z Operators ===
    # Multiply by Field Operators:
    exp_mu_z = math.exp(beta * mu_z / k) * p0 + math.exp(-beta * mu_z / k) * p1

    for i in range(n_spins):
        Z_operator = Z_operator * (exp_mu_z | spins[i])

    # Multiply by Interactions Operators:
    exp_plus = math.exp(beta * J / k)
    exp_minus = math.exp(-beta * J / k)

    for i, j in pairs:
        term_00 = ((exp_plus * p0) | spins[i]) * (p0 | spins[j])
        term_11 = ((exp_plus * p1) | spins[i]) * (p1 | spins[j])

        term_01 = ((exp_minus * p0) | spins[i]) * (p1 | spins[j])
        term_10 = ((exp_minus * p1) | spins[i]) * (p0 | spins[j])

        # Aligned spins get +J, anti-aligned spins get -J energy
        local_exp_J =term_00 + term_11 + term_01 + term_10
        Z_operator = Z_operator * local_exp_J

    # === Construct X Operators ===
    # Multiply by Field Operators:
    
    exp_mu_x_prime = math.exp(beta * mu_x / k) * p0 + math.exp(-beta * mu_x / k) * p1
    
    for i in range(n_spins):
        X_prime_operator = X_prime_operator * (exp_mu_x_prime | spins[i])

    # === Construct H Operators ===
    H_op = 1/math.sqrt(2) * (1.0 * p0 - 1.0 * p1 + 1.0 * p2 + 1.0 * p3)
    for i in range(n_spins):
        H_global_operator = H_global_operator * (H_op | spins[i])

    # === Trotterization Loop ===
    Trotter_slice = Z_operator * H_global_operator * X_prime_operator * H_global_operator
    global_operator = Trotter_slice

    for kk in range(k - 1):
        global_operator = global_operator * Trotter_slice

    return global_operator

def run_ising(spin_range, solvers, dim, J, mu_z, mu_x, k, beta):
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
        pairs, n_spins = generate_grid_pairs(dim, L)

        logical_formula = build_partition_function(n_spins, pairs, J, mu_z, mu_x, k, beta)
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
    
    J = 1.0
    mu_z = 1.0
    mu_x = 1.0
    k = 2
    # k_vals = [1, 2, 4, 8, 16, 32]
    beta = 1.0
    dim = 1
    L = [10, 20, 30, 40, 50]

    Output_dir = f"../Iago/Output/Transverse_Ising_{dim}D/Run_2"

    timings, z_vals = run_ising(L, active_solvers, dim, J, mu_z, mu_x, k, beta)
    save_and_plot(L, dim, timings, active_solvers, Output_dir)