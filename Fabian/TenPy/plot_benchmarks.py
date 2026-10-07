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


def summarize(points):
    """Mean and sample standard deviation across individual runs."""
    xs = sorted(points)
    averages = [mean(points[x]) for x in xs]
    deviations = [
        stdev(points[x]) if len(points[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, averages, deviations


def boundary_condition(row):
    # Determine boundary condition
    if row["bc_x"] == "open" and row["bc_y"] == "open":
        return "open"
    if row["bc_x"] == "periodic" and row["bc_y"] == "periodic":
        return "torus"
    return "cylinder"


def plot_results(relative_error=True):
    results_dir = Path(__file__).resolve().parent / "Results"

    ### RUNTIME AND TRUNCATION PLOTS AND CSV ###

    groups = defaultdict(lambda: defaultdict(list))

    with open(results_dir / "benchmark_results_600.csv", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
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
        points = {
            lx: [chi for chi, runtime in runs]
            for lx, runs in data.items()
        }
        lengths, max_chis, _ = summarize(points)

        ax_chi.plot(
            lengths,
            max_chis,
            label=label,
            **curve_style(label),
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
    ax_chi.set_ylabel(r"Maximum retained bond dimension $\chi$")
    ax_chi.set_ylim(bottom=0)

    # Plot disc and prop
    ax_chi.set_title("Maximum Bond Dimension: Random Bond Ising")
    ax_chi.set_xlabel(r"Length $L_x$")
    ax_chi.set_ylabel(r"Mean maximum retained bond dimension $\chi$")
    ax_chi.grid(alpha=0.3)
    ax_chi.legend(fontsize=8)
    fig_chi.tight_layout()
    fig_chi.savefig(results_dir / "Scaling_Bond_Dimension.pdf", format="pdf")
    plt.close(fig_chi)

    # Runtime plot
    fig_time, ax_time = plt.subplots(figsize=(6, 5))

    for label, data in groups.items():
        points = {
            lx: [runtime for chi, runtime in runs]
            for lx, runs in data.items()
        }
        lengths, runtimes, _ = summarize(points)

        ax_time.plot(
            lengths,
            runtimes,
            label=label,
            **curve_style(label),
        )

    ax_time.set_title(r"Runtime for $\log Z$: Random Bond Ising")
    ax_time.set_xlabel(r"Length $L_x$")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=8)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Mean runtime [s]")
    ax_time.grid(which="major", alpha=0.3)
    ax_time.grid(which="minor", alpha=0.1)
    fig_time.tight_layout()
    fig_time.savefig(results_dir / "Scaling_Runtime.pdf", format="pdf")
    plt.close(fig_time)

    ### RUNTIME VS BETA FOR FIXED GEOMETRY ###

    beta_runtimes = defaultdict(list)

    with open(results_dir / "benchmark_results_beta.csv", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            beta_values = ast.literal_eval(row["betas"])
            max_beta = round(max(beta_values), 12)
            runtime = float(row["runtime_s"])

            beta_runtimes[max_beta].append(runtime)

    # Sort by beta
    betas, runtimes, _ = summarize(beta_runtimes)

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
    ax_time.set_ylabel("Mean runtime [s]")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=8)

    fig_time.tight_layout()
    fig_time.savefig(results_dir / "Scaling_Runtime_Beta.pdf", format="pdf")
    plt.close(fig_time)

    ### ERROR IN FINAL LOG Z ###

    error_groups = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    with open(
        results_dir / "accuracy_benchmarks_dt00625.csv", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            lx = int(row["Lx"])
            ly = int(row["Ly"])
            boundary = boundary_condition(row)

            final_log_z = ast.literal_eval(row["log_z_approx"])[-1]
            exact_log_z = float(row["log_z_exact"])

            # Calculate each run's error before averaging.
            error = abs(final_log_z - exact_log_z)

            if relative_error:
                if exact_log_z == 0:
                    raise ValueError(
                        "Relative error is undefined when log_z_exact is zero."
                    )
                error /= abs(exact_log_z)

            chi_limit = int(row["chi_limit"])
            beta_final = round(ast.literal_eval(row["betas"])[-1], 12)

            # Separate plots by lattice, boundary conditions, and settings.
            geometry_key = (
                row["lattice"],
                boundary,
                row["bc_x"],
                row["bc_y"],
                row["bc_MPS"],
                float(row["p"]),
                float(row["dt"]),
                beta_final,
            )

            error_groups[geometry_key][(lx, ly)][chi_limit].append(error)

    error_type = "Relative" if relative_error else "Absolute"
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for geometry_key, size_groups in sorted(error_groups.items()):
        lattice, boundary, bc_x, bc_y, bc_mps, p, dt, beta_final = geometry_key

        fig_error, ax_error = plt.subplots(figsize=(6, 5))

        for index, (size, points) in enumerate(sorted(size_groups.items())):
            lx, ly = size
            chi_limits, errors, error_std = summarize(points)

            ax_error.errorbar(
                chi_limits,
                errors,
                yerr=error_std,
                marker=markers[index % len(markers)],
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                capsize=4,
                elinewidth=1.2,
                capthick=1.2,
                label=rf"${lx} \times {ly}$",
            )

        ax_error.set_title(
            f"{lattice} - {boundary}\n"
            rf"$\beta={beta_final:g}$, $dt={dt:g}$"
            " — mean ± sample SD"
        )
        ax_error.set_xlabel(r"Imposed bond-dimension cap $\chi_{\max}$")

        if relative_error:
            ax_error.set_ylabel(
                r"Mean $|\log Z_{\mathrm{approx}}-\log Z_{\mathrm{exact}}|"
                r"/|\log Z_{\mathrm{exact}}|$"
            )
        else:
            ax_error.set_ylabel(
                r"Mean $|\log Z_{\mathrm{approx}}-\log Z_{\mathrm{exact}}|$"
            )

        chi_ticks = sorted({
            chi
            for points in size_groups.values()
            for chi in points
        })

        ax_error.set_xscale("log", base=2)
        ax_error.set_xticks(chi_ticks)
        ax_error.set_xticklabels([str(chi) for chi in chi_ticks])
        ax_error.grid(alpha=0.3)
        ax_error.legend(title=r"$L_x \times L_y$", fontsize=8)

        fig_error.tight_layout()

        filename = (
            f"{error_type}_Error_Log_Z_{lattice}_{boundary}"
            f"_bcx-{bc_x}_bcy-{bc_y}_{bc_mps}"
            f"_p-{p:g}_dt-{dt:g}_beta-{beta_final:g}.pdf"
        )
        fig_error.savefig(results_dir / filename, format="pdf")
        plt.close(fig_error)


if __name__ == "__main__":
    # Set False for absolute differences instead of relative errors.
    plot_results(relative_error=True)