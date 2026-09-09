import math
import time
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from wcnf_matrix import *
from functools import reduce

def build_partition_function(n_spins, J, h, beta):
    spin_space = Index(2)
    p0 = ket(spin_space[0]) * bra(spin_space[0])
    p1 = ket(spin_space[1]) * bra(spin_space[1])
    identity = p0 + p1
    spins = [Reg(spin_space) for i in range(n_spins)]

    # Start with a global identity state
    identity_factors = []
    for i in range(n_spins):
        identity_factors.append(identity | spins[i])

    global_operator = reduce(lambda x, y: x * y, identity_factors)

    # Multiply by Field Operators:
    exp_h = math.exp(beta * h) * p0 + math.exp(-beta * h) * p1

    for i in range(n_spins):
        global_operator = global_operator * (exp_h | spins[i])

    # Multiply by Interactions Operators:
    exp_plus = math.exp(beta * J)
    exp_minus = math.exp(-beta * J)
    for i in range(n_spins - 1):
        term_00 = ((exp_plus * p0) | spins[i]) * (p0 | spins[i + 1])
        term_11 = ((exp_plus * p1) | spins[i]) * (p1 | spins[i + 1])

        term_01 = ((exp_minus * p0) | spins[i]) * (p1 | spins[i + 1])
        term_10 = ((exp_minus * p1) | spins[i]) * (p0 | spins[i + 1])

        # Aligned spins get +J, anti-aligned spins get -J energy
        local_exp_J =term_00 + term_11 + term_01 + term_10
        global_operator = global_operator * local_exp_J

    return global_operator

def analytic_exact_Z(n_spins, J, h, beta):
    # Calculate Z via 2x2 transfer matrix for ground validation
    if n_spins == 1:
        return 2.0 * math.cosh(beta * h)

    T = np.array([
        [math.exp(beta * (J + h)),  math.exp(-beta * J)],
        [math.exp(-beta * J),  math.exp(beta * (J - h))]
    ], dtype=np.float64)

    v_left = np.array([math.exp(beta * h), math.exp(-beta * h)], dtype=np.float64)
    v_right = np.array([1.0, 1.0], dtype=np.float64)

    T_pow = np.linalg.matrix_power(T, n_spins - 1)
    return float(v_left @ T_pow @ v_right)

def run_ising(spin_range, solvers, J, h, beta):
    timings = {name: [] for name in solvers}
    z_records = {name: [] for name in solvers}
    exact_records = []

    err_headers = " | ".join([f"Err {name:<11}" for name in solvers])
    time_headers = " | ".join([f"Time {name:<10}" for name in solvers])

    print(f"{'N':<4} | {err_headers} | {time_headers}")
    print("-" * (7 + 18 * len(solvers) * 2))

    for N in spin_range:
        z_exact = analytic_exact_Z(N, J, h, beta)
        exact_records.append(z_exact)

        logical_formula = build_partition_function(N, J, h, beta)
        cnf, weight_function = logical_formula.mat.trace_formula()

        row_output = f"{N:<4} | "
        err_strs = []
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

                rel_err = abs(computed_z - z_exact) / z_exact if z_exact != 0 else abs(computed_z)

                err_strs.append(f"{rel_err:<15.2e}")
                time_strs.append(f"{elapsed:<15.4f}")
            except Exception as e:
                # Capture solver timeout, out-of-memory, or parse failures gracefully
                timings[name].append(None)
                z_records[name].append(None)
                err_strs.append(f"{'FAILED':<15}")
                time_strs.append(f"{'FAILED':<15}")

        row_output += " | ".join(err_strs) + " | " + " | ".join(time_strs)
        print(row_output)

    print("=" * (7 + 18 * len(solvers) * 2) + "\n")
    return timings, z_records, exact_records

def save_and_plot(spin_range, timings, solvers, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Save raw timing data
    with open(f"{output_dir}/benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N"] + list(solvers.keys()))
        for idx, N in enumerate(spin_range):
            writer.writerow([N] + [timings[s][idx] for s in solvers])

    # Plot scaling curve
    plt.figure(figsize=(9, 5.5))
    for name in solvers:
        valid_points = [(N, t) for N, t in zip(spin_range, timings[name]) if t is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, marker='o', linewidth=1.8, label=name)

    plt.xlabel("Number of Spins ($N$)")
    plt.ylabel("ExecutionTime (seconds)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_benchmark.png", dpi=300)
    print("\nSaved plot to 'scaling_benchmark.png' and raw data to 'benchmark_results.csv'.")

if __name__ == "__main__":
    test_spins = [10, 30, 50, 70, 90, 110, 130]
    active_solvers = {
        "Cachet": Cachet,
        "DPMC": DPMC,
        "TensorOrder": TensorOrder
    }

    Output_dir = "../Iago/Output/Run_3"

    timings, z_vals, ground_truth = run_ising(test_spins, active_solvers, J=1.5, h=0.5, beta=1.0)
    save_and_plot(test_spins, timings, active_solvers, Output_dir)