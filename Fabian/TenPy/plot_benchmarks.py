"""
Run this for recreating the results shown in the report
-> If fresh CSV file is needed run run_bechnmarks.py (Attention: Computationally expensive!!)
"""

import ast
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def curve_style(label):
    if "Triangular" in label:
        return {
            "color": "tab:red", "marker": "^", "linestyle": "--",
            "markersize": 5, "zorder": 4,
        }
    if "torus" in label:
        return {
            "color": "tab:orange", "marker": "s", "linestyle": "-",
            "markersize": 9, "markerfacecolor": "none", "zorder": 3,
        }
    if "cylinder" in label:
        return {"color": "tab:green", "marker": "o", "linestyle": "-"}

    return {"color": "tab:blue", "marker": "o", "linestyle": "-"}

if __name__ == "__main__":

    ### RUNTIME AND TRUNCATION PLOTS AND CSV ###    

    groups = defaultdict(list)

    with open("benchmark_results.csv", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            lx = int(row["Lx"])
            ly = int(row["Ly"])

            # Determine boundary condition
            if row["bc_x"] == "open" and row["bc_y"] == "open":
                boundary = "open"
            elif row["bc_x"] == "periodic" and row["bc_y"] == "periodic":
                boundary = "torus"
            else:
                boundary = "cylinder"

            if boundary == "cylinder":
                label = rf"{row['lattice']} - cylinder ($L_x \times {ly}$)"
            else:
                label = rf"{row['lattice']} - {boundary} ($L_x \times L_x$)"

            max_chi_values = ast.literal_eval(row["max_chi"])
            max_chi = max(max_chi_values)
            runtime = float(row["runtime_s"])

            groups[label].append((lx, max_chi, runtime))

    # Bond dimension plot
    fig_chi, ax_chi = plt.subplots(figsize=(6, 5))

    for label, data in groups.items():
        data.sort()

        lengths = [item[0] for item in data]
        max_chis = [item[1] for item in data]

        ax_chi.plot(lengths, max_chis, label=label, **curve_style(label))

    # Set maximum chi line
    ax_chi.axhline(
        400,
        color="0.5",
        linestyle=":",
        linewidth=1.2,
        label=r"Imposed cap: $\chi_{\max}=400$",
        zorder=0,
    )
    ax_chi.set_ylabel(r"Maximum retained bond dimension $\chi$")
    ax_chi.set_ylim(bottom=0)

    # Plot disc and prop
    ax_chi.set_title("Maximum Bond Dimension: Random Bond Ising")
    ax_chi.set_xlabel(r"Length $L_x$")
    ax_chi.set_ylabel(r"Maximum retained bond dimension $\chi$")
    ax_chi.grid(alpha=0.3)
    ax_chi.legend(fontsize=8)
    fig_chi.tight_layout()
    fig_chi.savefig("Scaling_Bond_Dimension.pdf", format="pdf")

    # Runtime plot
    fig_time, ax_time = plt.subplots(figsize=(6, 5))

    for label, data in groups.items():
        data.sort()

        lengths = [item[0] for item in data]
        runtimes = [item[2] for item in data]

        ax_time.plot(lengths, runtimes, label=label, **curve_style(label))

    ax_time.set_title(r"Runtime for $\log Z$: Random Bond Ising")
    ax_time.set_xlabel(r"Length $L_x$")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=8)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Runtime [s]")
    ax_time.grid(which="major", alpha=0.3)
    ax_time.grid(which="minor", alpha=0.1)
    fig_time.tight_layout()
    fig_time.savefig("Scaling_Runtime.pdf", format="pdf")


    ### RUNTIME VS BETA FOR FIXED GEOMETRY ###

    beta_runtimes = []

    with open("benchmark_results_beta.csv", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            beta_values = ast.literal_eval(row["betas"])
            max_beta = max(beta_values)
            runtime = float(row["runtime_s"])

            beta_runtimes.append((max_beta, runtime))

    # Sort by beta
    beta_runtimes.sort()

    betas = [item[0] for item in beta_runtimes]
    runtimes = [item[1] for item in beta_runtimes]

    # Runtime vs beta plot
    fig_time, ax_time = plt.subplots(figsize=(6, 5))

    ax_time.plot(
        betas,
        runtimes,
        "o-",
        label=r"Square - open ($L_x=L_y=3$)",
    )

    ax_time.set_title(r"Runtime versus inverse temperature: Random Bond Ising")
    ax_time.set_xlabel(r"Final inverse temperature $\beta_{\max}$")
    ax_time.set_ylabel("Runtime [s]")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=8)

    fig_time.tight_layout()
    fig_time.savefig("Scaling_Runtime_Beta.pdf", format="pdf")