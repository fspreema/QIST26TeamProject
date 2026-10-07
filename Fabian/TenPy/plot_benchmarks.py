"""
Run this for recreating the results shown in the report
-> If fresh CSV file is needed run run_bechnmarks.py (Attention: Computationally expensive!!)
"""

import ast
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

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


def boundary_condition(row):
    # Determine boundary condition
    return {
        ("open", "open"): "open",
        ("periodic", "open"): "cylinder",
        ("periodic", "periodic"): "torus",
    }[(row["bc_x"], row["bc_y"])]


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def set_title(ax, quantity, beta, dt):
    ax.set_title(
        f"{quantity}\n"
        rf"$\beta_{{\max}}={beta}$, $\Delta\tau={dt:g}$",
        fontsize=11,
        pad=8,
        linespacing=1.2,
    )


def plot_results():
    results_dir = Path(__file__).resolve().parent / "Results"

    ### RUNTIME AND TRUNCATION PLOTS AND CSV ###

    groups = defaultdict(lambda: defaultdict(list))

    for row in read_csv(results_dir / "benchmark_results_600.csv"):
        lx = int(row["Lx"])
        ly = int(row["Ly"])
        boundary = boundary_condition(row)

        if boundary == "cylinder":
            label = rf"{row['lattice']} - cylinder ($L_x \times {ly}$)"
        else:
            label = rf"{row['lattice']} - {boundary} ($L_x \times L_x$)"

        max_chi_values = ast.literal_eval(row["max_chi"])
        max_chi = max(max_chi_values)
        runtime = float(row["runtime_s"])

        groups[label][lx].append((max_chi, runtime))

    # Bond dimension plot
    fig_chi, ax_chi = plt.subplots(figsize=(6, 5))

    for label, data in groups.items():
        lengths = sorted(data)
        max_chis = [
            mean(chi for chi, runtime in data[lx])
            for lx in lengths
        ]

        ax_chi.plot(
            lengths, max_chis, label=label, **curve_style(label)
        )

    # Set maximum chi line
    ax_chi.axhline(
        600,
        color="0.5",
        linestyle=":",
        linewidth=1.2,
        label=r"Imposed cap: $\chi_{\max}=600$",
        zorder=0,
    )
    ax_chi.set_ylim(bottom=0)

    # Plot disc and prop
    set_title(ax_chi, "Maximum Bond Dimension", beta="0.5", dt=0.05)
    ax_chi.set_xlabel(r"Length $L_x$")
    ax_chi.set_ylabel(r"Mean maximum retained bond dimension $\chi$")
    ax_chi.grid(alpha=0.3)
    ax_chi.legend(fontsize=8)
    fig_chi.tight_layout()
    fig_chi.savefig(results_dir / "Scaling_Bond_Dimension.pdf")
    plt.close(fig_chi)

    # Runtime plot
    fig_time, ax_time = plt.subplots(figsize=(6, 5))

    for label, data in groups.items():
        lengths = sorted(data)
        runtimes = [
            mean(runtime for chi, runtime in data[lx])
            for lx in lengths
        ]

        ax_time.plot(
            lengths, runtimes, label=label, **curve_style(label)
        )

    set_title(ax_time, r"Runtime for $\log Z$", beta="0.5", dt=0.05)
    ax_time.set_xlabel(r"Length $L_x$")
    ax_time.set_ylabel("Mean runtime [s]")
    ax_time.set_yscale("log")
    ax_time.grid(which="major", alpha=0.3)
    ax_time.grid(which="minor", alpha=0.1)
    ax_time.legend(fontsize=8)
    fig_time.tight_layout()
    fig_time.savefig(results_dir / "Scaling_Runtime.pdf")
    plt.close(fig_time)

    ### RUNTIME VS BETA FOR FIXED GEOMETRY ###

    beta_runtimes = defaultdict(list)

    for row in read_csv(results_dir / "benchmark_results_beta.csv"):
        beta_values = ast.literal_eval(row["betas"])
        max_beta = round(beta_values[-1], 12)
        runtime = float(row["runtime_s"])

        beta_runtimes[max_beta].append(runtime)

    # Sort by beta
    betas = sorted(beta_runtimes)
    runtimes = [mean(beta_runtimes[beta]) for beta in betas]

    # Runtime vs beta plot
    fig_time, ax_time = plt.subplots(figsize=(6, 5))

    ax_time.plot(
        betas,
        runtimes,
        "o-",
        label=r"Square - open ($L_x=L_y=3$)",
    )

    set_title(
        ax_time,
        "Runtime versus Inverse Temperature",
        beta=rf"{betas[0]:g}\,\mathrm{{to}}\,{betas[-1]:g}",
        dt=0.05,
    )
    ax_time.set_xlabel(r"Final inverse temperature $\beta_{\max}$")
    ax_time.set_ylabel("Mean runtime [s]")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=8)
    fig_time.tight_layout()
    fig_time.savefig(results_dir / "Scaling_Runtime_Beta.pdf")
    plt.close(fig_time)

    ### ERROR IN FINAL LOG Z ###

    error_groups = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in read_csv(results_dir / "accuracy_benchmarks_dt00625.csv"):
        lattice = row["lattice"]
        boundary = boundary_condition(row)
        size = (int(row["Lx"]), int(row["Ly"]))
        chi_limit = int(row["chi_limit"])

        final_log_z = ast.literal_eval(row["log_z_approx"])[-1]
        exact_log_z = float(row["log_z_exact"])

        # Calculate each run's error before averaging.
        error = abs(final_log_z - exact_log_z) / abs(exact_log_z)

        # Separate plots by lattice, boundary conditions, and settings.
        error_groups[(lattice, boundary)][size][chi_limit].append(error)

    for (lattice, boundary), size_groups in sorted(error_groups.items()):
        fig_error, ax_error = plt.subplots(figsize=(6, 5))

        for (lx, ly), data in sorted(size_groups.items()):
            chi_limits = sorted(data)
            errors = [mean(data[chi]) for chi in chi_limits]
            error_std = [stdev(data[chi]) for chi in chi_limits]

            ax_error.errorbar(
                chi_limits,
                errors,
                yerr=error_std,
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                capsize=4,
                elinewidth=1.2,
                capthick=1.2,
                label=rf"${lx} \times {ly}$",
            )

        set_title(
            ax_error,
            rf"Relative Error in $\log Z$: {lattice} - {boundary}",
            beta="0.5",
            dt=0.00625,
        )
        ax_error.set_xlabel(r"Imposed bond-dimension cap $\chi_{\max}$")
        ax_error.set_ylabel(
            r"Mean $|\log Z_{\mathrm{approx}}-\log Z_{\mathrm{exact}}|"
            r"/|\log Z_{\mathrm{exact}}|$"
        )

        chi_ticks = sorted({
            chi for data in size_groups.values() for chi in data
        })
        ax_error.set_xscale("log", base=2)
        ax_error.set_xticks(chi_ticks)
        ax_error.set_xticklabels([str(chi) for chi in chi_ticks])
        ax_error.grid(alpha=0.3)
        ax_error.legend(title=r"$L_x \times L_y$", fontsize=8)

        fig_error.tight_layout()
        fig_error.savefig(
            results_dir / f"Relative_Error_Log_Z_{lattice}_{boundary}.pdf"
        )
        plt.close(fig_error)


if __name__ == "__main__":
    plot_results()