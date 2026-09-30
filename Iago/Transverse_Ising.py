import math
import time
import csv
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from wcnf_matrix import *
from functools import reduce
import scipy.linalg as la

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
    
    q_plus = p0 ** p0 + p1 ** p1
    q_minus = p0 ** p1 + p1 ** p0

    # Aligned spins get +J, anti-aligned spins get -J energy
    local_exp_J = exp_plus * q_plus + exp_minus * q_minus

    for i, j in pairs:
        Z_operator = Z_operator * (local_exp_J | (spins[i], spins[j]))

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

def exact_partition_function(n_spins, pairs, J, mu_z, mu_x, beta):
    """
    Calculates the exacta value of Z using the hola exponential matrix of the Hamiltonian
    """
    I = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])

    def apply_local_op(op, target_idx):
        # Constructs Kronecker product:
        ops = [I] * n_spins
        ops[target_idx] = op
        global_op = ops[0]
        for i in range(1, n_spins):
            global_op = np.kron(global_op, ops[i])
        return global_op

    H = np.zeros((2**n_spins, 2**n_spins))

    # Pairs interaction (-J * Z_i * Z_j)
    for i, j in pairs:
        Z_i = apply_local_op(Z, i)
        Z_j = apply_local_op(Z, j)
        H -= J * (Z_i @ Z_j)

    # Magnetic fields interactions
    for i in range(n_spins):
        H -= mu_z * apply_local_op(Z, i)
        H -= mu_x * apply_local_op(X, i)

    # Partition function: Z = Tr(e^(-beta * H))
    exp_H = la.expm(-beta * H)
    Z_exact = np.trace(exp_H)
    
    return np.real(Z_exact)

def run_ising(spin_range, solvers, dim, J, mu_z, mu_x, k, beta):
    """
    Computes the trace of the partition function formula using the diferent solvers.
    As outputs gives the values of the partition function and the time 
    """

    timings = {name: {L: [] for L in spin_range} for name in solvers}
    z_records = {name: {L: [] for L in spin_range} for name in solvers}
    z_exact_records = {L: None for L in spin_range}
    dpmc_errors = {L: [] for L in spin_range}

    time_headers = " | ".join([f"Time {name:<10}" for name in solvers])
    header = f"{'L':<4} | {'N':<4} | {'k':<4} | {time_headers} | {'DPMC Error':<12}"

    print(header)
    print("-" * len(header))

    for L in spin_range:
        pairs, n_spins = generate_grid_pairs(dim, L)

        try:
            z_exact = exact_partition_function(n_spins, pairs, J, mu_z, mu_x, beta)
            z_exact_records[L] = z_exact
        except MemoryError:
            z_exact = None
            z_exact_records[L] = None

        # Loop for all k values
        for k in k_vals:
            logical_formula = build_partition_function(n_spins, pairs, J, mu_z, mu_x, k, beta)
            cnf, weight_function = logical_formula.mat.trace_formula()

            row_output = f"{L:<4} | {n_spins:<4} | {k:<4} | "
            time_strs = []

            # Run each solver independently for the same CNF formula
            for name, solver_cls in solvers.items():
                try:            
                    solver = solver_cls()
                    result = solver.model_count(cnf, weight_function)
                    computed_z = result.model_count
                    elapsed = result.runtime

                    if elapsed == -1.0 or computed_z == 0.0:
                        timings[name][L].append(None)
                        z_records[name][L].append(None)
                        time_strs.append(f"{'FAILED':<15}")
                    else:
                        timings[name][L].append(elapsed)
                        z_records[name][L].append(computed_z)
                        time_strs.append(f"{elapsed:<15.4f}")
                except Exception as e:
                    # Capture solver timeout, out-of-memory, or parse failures gracefully
                    timings[name][L].append(None)
                    z_records[name][L].append(None)
                    time_strs.append(f"{'FAILED':<15}")

            row_output += " | ".join(time_strs)

            if z_exact is not None and "DPMC" in z_records and z_records["DPMC"][L][-1] is not None:
                dpmc_z = z_records["DPMC"][L][-1]
                rel_error = abs(dpmc_z - z_exact) / z_exact
                dpmc_errors[L].append(rel_error)
                row_output += f" | {rel_error:<12.2e}"
            else:
                dpmc_errors[L].append(None)
                row_output += f" | {'N/A':<12}"
            
            print(row_output)

        print("-" * len(header))

    print("=" * len(header) + "\n")
    return timings, z_records, z_exact_records, dpmc_errors

def save_and_plot(L_vals, k_vals, dim, timings, solvers, dpmc_errors, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Number of total spins for each L
    n_spins = [L ** dim for L in L_vals]
    
    # Save raw timing data
    with open(f"{output_dir}/benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "N", "k"] + list(solvers.keys()) + ["DPMC_Error"])
        for idx, L in enumerate(L_vals):
            for i, k in enumerate(k_vals):
                row = [L, n_spins[idx], k]
                for s in solvers:
                    if i < len(timings[s][L]):
                        row.append(timings[s][L][i])
                    else:
                        row.append(None)
                
                if i < len(dpmc_errors[L]):
                    row.append(dpmc_errors[L][i])
                else:
                    row.append(None)
                writer.writerow(row)

    box_style = dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', alpha=0.8, edgecolor='silver')

    # ==== Plot 1: Time vs L (Side Lenght) ====
    plt.figure(figsize=(9, 5.5))
    for name in solvers:
        # We use the Runtime of the tird "k" evaluated
        y_vals = [timings[name][L][2] for L in L_vals if len(timings[name][L]) > 2 and timings[name][L][2] is not None]
        x_vals = [L for L in L_vals if len(timings[name][L]) > 2 and timings[name][L][2] is not None]
        if x_vals:
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
        # We use the Runtime of the tird "k" evaluated
        y_vals = [timings[name][L][2] for L in L_vals if len(timings[name][L]) > 2 and timings[name][L][2] is not None]
        x_vals = [n_spins[idx] for idx, L in enumerate(L_vals) if len(timings[name][L]) > 2 and timings[name][L][2] is not None]
        if x_vals:
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

    # ==== Plot 3: DPMC Runtime vs Relative Error ====
    plt.figure(figsize=(9, 5.5))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, L in enumerate(L_vals):
        n_spins = L ** dim
        valid_points = [(err, t) for err, t in zip(dpmc_errors[L], timings["DPMC"][L]) if err is not None and t is not None]
        
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, marker=markers[idx % len(markers)], linewidth=1.8, label=f"$n={n_spins}$")

    plt.xlabel("Relative Error")
    plt.ylabel("Runtime (s)")
    plt.xscale("log")
    plt.yscale("log")
    
    # Invert x axis:
    plt.gca().invert_xaxis()
    
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/relative_error_convergence.png", dpi=300)
    plt.close()

    print(f"\nSaved plots 'scaling_vs_L.png' and 'scaling_vs_N.png' to '{output_dir}'")

if __name__ == "__main__":

    solvers = {
        "Cachet": Cachet,
        "DPMC": DPMC,
        "TensorOrder": TensorOrder
    }
    
    J = 1.0
    mu_z = 1.0
    mu_x = 1.0
    # k = 2 
    beta = 1.0
    dim = 1

    L_vals = [2, 3, 4, 5, 6, 7, 8]

    k_vals = [1, 2, 4, 8, 16]

    Output_dir = f"../Iago/Output/Transverse_Ising_{dim}D/Run_4"

    timings, z_vals, z_exact_records, dpmc_errors = run_ising(L_vals, solvers, dim, J, mu_z, mu_x, k_vals, beta)
    save_and_plot(L_vals, k_vals, dim, timings, solvers, dpmc_errors, Output_dir)