"""
Run this for recreating the results shown in the report
-> If fresh CSV file is needed run run_bechnmarks.py (Attention: Computationally expensive!!)
"""

import ast
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

if __name__ == "__main__":

    groups = defaultdict(list)

    with open("benchmark_results.csv", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            lx = int(row["Lx"])
            ly = int(row["Ly"])

            # Determine boundary condition
            if row["bc_x"] == "open" and row["bc_y"] == "open":
                boundary = "open"
            else:
                boundary = "torus"

            # Name the geometry
            if row["bc_x"] == "open" and row["bc_y"] == "open":
                boundary = "open"
            elif row["bc_x"] == "periodic" and row["bc_y"] == "periodic":
                boundary = "torus"
            else:
                boundary = "long cylinder"

            if boundary == "long cylinder":
                label = rf"{row['lattice']} - cylinder ($L_y = 3$)"
            else:
                label = f"{row['lattice']} - {boundary}"

            max_chi_values = ast.literal_eval(row["max_chi"])
            max_chi = max(max_chi_values)
            runtime = float(row["runtime_s"])

            groups[label].append((lx, max_chi, runtime))

    # Create plots
    fig, (ax_chi, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

    for label, data in groups.items():
        data.sort()

        lengths = [item[0] for item in data]
        max_chis = [item[1] for item in data]
        runtimes = [item[2] for item in data]

        ax_chi.plot(lengths, max_chis, "o-", label=label)
        ax_time.plot(lengths, runtimes, "o-", label=label)

    # Chi plot
    ax_chi.set_title("Maximum Bond Dimension: Random Bond Ising")
    ax_chi.set_xlabel(r"Length $L_x$")
    ax_chi.set_ylabel(r"Maximum $\chi$")
    ax_chi.grid(alpha=0.3)

    # Runtime plot
    ax_time.set_title("Maximum Runtime for $\log{Z}$: Random Bond Ising")
    ax_time.set_xlabel(r"Length $L_x$")
    ax_time.set_ylabel("Runtime [s]")
    ax_time.grid(alpha=0.3)

    ax_time.legend(fontsize=8)

    plt.tight_layout()
    plt.show()