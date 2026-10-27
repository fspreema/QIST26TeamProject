"""
PLotting function to get the joint plot of WMC and TenPy package for comparison of the results.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def curve_style(label, wmc=False):

    if wmc:
        color = "tab:blue"
    else:
        color = "tab:orange"

    if "Triangular" in label:
        return {
            "color": color, "marker": "^", "linestyle": "--",
            "markersize": 5, "zorder": 4,
        }
    
    if "torus" in label:
        return {
            "color": color, "marker": "s", "linestyle": "-",
            "markersize": 9, "markerfacecolor": "none", "zorder": 3,
        }
    if "cylinder" in label:
        return {"color": color, "marker": "o", "linestyle": "-"}

    return {"color": color, "marker": "o", "linestyle": "-"}


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
    results_dir_tenpy = Path(__file__).resolve().parent / "Results"
    results_dir_wmc = Path(__file__).resolve().parents[2] / "RBIM_WMC" / "Results" / "Tenpy_model"

    ### RUNTIME AND TRUNCATION PLOTS AND CSV ###


    # Use groupings for the TeNPy results
    groups = defaultdict(lambda: defaultdict(list))

    for row in read_csv(results_dir_tenpy / "benchmark_results_600.csv"):
        lx = int(row["Lx"])
        ly = int(row["Ly"])
        boundary = boundary_condition(row)

        if boundary == "cylinder":
            label = rf"{row['lattice']} - cylinder ($L_x \times {ly}$)"
        else:
            label = rf"{row['lattice']} - {boundary} ($L_x \times L_x$)"

        runtime = float(row["runtime_s"])

        n_spins = lx * ly
        groups[label][n_spins].append(runtime)

    # For WMC load in as dataframe
    # -> I know this is sketchy but I dind't do the WMC plotting and csv generation so I don't want to change it too much
    df = pd.read_csv(results_dir_wmc / "TensorOrder_MaxLength-16_Average-16_2026-10-09 01-03-45.csv")

    # Get num of spins for wmc
    sizes = df["Size"]
    SC_spins = [((s*s)//3) * 3 for s in sizes if ((s*s)//3) * 3 <= 50]
    SO_spins = [s*s for s in sizes if s*s <= 50]
    ST_spins = [s*s for s in sizes if s*s <= 50]
    TT_spins = [s*s for s in sizes if s*s <= 50]

    # read out runtime -> stds we do not need here
    SC_runtimes = df["SC_runtimes"][:len(SC_spins)]
    SO_runtimes = df["SO_runtimes"][:len(SO_spins)]
    ST_runtimes = df["ST_runtimes"][:len(ST_spins)]
    TT_runtimes = df["TT_runtimes"][:len(TT_spins)]


    # Runtime plot
    fig_time, ax_time = plt.subplots(figsize=(6, 5))

    # Plot the TeNPy results
    for label, data in groups.items():
        spin_counts = sorted(data)
        runtimes = [
            mean(runtime for runtime in data[n])
            for n in spin_counts
        ]

        ax_time.plot(
            spin_counts, runtimes, label="_nolegend_", **curve_style(label)
        )

    # Plot the WMC results
    ax_time.plot(SC_spins, SC_runtimes, label="_nolegend_", **curve_style("Square - cylinder", wmc=True))
    ax_time.plot(SO_spins, SO_runtimes, label="_nolegend_", **curve_style("Square - open", wmc=True))
    ax_time.plot(ST_spins, ST_runtimes, label="_nolegend_", **curve_style("Square - torus", wmc=True))
    ax_time.plot(TT_spins, TT_runtimes, label="_nolegend_", **curve_style("Triangular - torus", wmc=True))

    geometry_specs = [
        (r"Square - cylinder ($L_x \times 3$)", "Square - cylinder"),
        (r"Square - open ($L_x \times L_x$)", "Square - open"),
        (r"Square - torus ($L_x \times L_x$)", "Square - torus"),
        (r"Triangular - torus ($L_x \times L_x$)", "Triangular - torus"),
    ]
    geometry_handles = []
    for legend_label, style_label in geometry_specs:
        style = curve_style(style_label)
        style["color"] = "0.35"
        geometry_handles.append(
            Line2D([], [], label=legend_label, **style)
        )

    source_handles = [
        Line2D([], [], color=curve_style("", wmc=False)["color"], label="TeNPy"),
        Line2D([], [], color=curve_style("", wmc=True)["color"], label="WMC"),
    ]

    set_title(ax_time, r"Runtime DiracWMC vs TeNPy for $\log Z$", beta="0.5", dt=0.05)
    ax_time.set_xlabel(r"Total number of spins $N=L_x L_y$")
    ax_time.set_ylabel("Mean runtime [s]")
    ax_time.set_yscale("log")
    ax_time.grid(which="major", alpha=0.3)
    ax_time.grid(which="minor", alpha=0.1)
    geometry_legend = fig_time.legend(
        handles=geometry_handles,
        title="Geometry",
        loc="lower center",
        bbox_to_anchor=(0.34, 0.01),
        ncol=2,
        fontsize=8,
    )
    fig_time.add_artist(geometry_legend)
    fig_time.legend(
        handles=source_handles,
        title="Data source",
        loc="lower center",
        bbox_to_anchor=(0.83, 0.01),
        ncol=1,
        fontsize=8,
    )
    fig_time.tight_layout(rect=(0, 0.18, 1, 1))
    fig_time.savefig(results_dir_tenpy / "Scaling_Runtime_TeNPy_vs_WMC.pdf")
    plt.close(fig_time)

if __name__ == "__main__":
    plot_results()