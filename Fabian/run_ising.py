import argparse
import sys
import traceback
from pathlib import Path
from time import perf_counter

# Make the local DiracWMC checkout available to this test script.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "external" / "DiracWMC" / "wcnf_matrix"))

import matplotlib.pyplot as plt
import wcnf_matrix as wmc
from ising_model import IsingModelWMC, IsingParameters

# Shared solver selection
solvers = {
    "DPMC": wmc.DPMC,
    "Cachet": wmc.Cachet,
    "TensorOrder": wmc.TensorOrder,
}

# Shared plotting function
def plot_runtime(lengths, runtime, *, title, xlabel, filename):
    """Plot and save a solver runtime comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, times in runtime.items():
        ax.plot(lengths, times, marker="o", linewidth=2, markersize=6, label=name)
    ax.legend(title="Solver", frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total runtime (s)")
    ax.set_title(title)
    ax.set_xticks(lengths)
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / filename
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Plot saved to {output_path}")

# 1D experiment
def run_1d_experiment():
    """
    Run and plot the 1D Ising experiment with respect to runtime and chain size
    """
    lengths = list(range(3, 7))
    runtime = {name: [] for name in solvers}

    print("\n========================================================\n1D ISING EXPERIMENT\n========================================================", flush=True)

    # Run each solver on the same model parameters.
    for name, solver_class in solvers.items():
        print(f"\n--- {name} ---", flush=True)
        wmc.set_model_counter(solver_class)

        for curr_size in lengths:
            h_interactions, j_interactions = IsingParameters(
                length=curr_size, periodic=False
            ).get_1dim_parameters()

            model = IsingModelWMC(
                num_spins=curr_size,
                j_interactions=j_interactions,
                h_field=h_interactions,
                beta=1.0,
            )

            start = perf_counter()
            print(f"\n[{name}] START L={curr_size}, spins={curr_size}; "
                  "solver limit=300 s, container wait=330 s", flush=True)
            try:
                partition_function = model.get_partition_function()
            except Exception:
                print(f"[{name}] FAILED L={curr_size} after "
                      f"{perf_counter() - start:.2f} s", file=sys.stderr, flush=True)
                traceback.print_exc()
                # Leave a gap in the plot and continue with the other cases.
                runtime[name].append(float("nan"))
            else:
                elapsed = perf_counter() - start
                runtime[name].append(elapsed)
                print(f"[{name}] OK L={curr_size}: Z={partition_function}, "
                      f"total elapsed={elapsed:.2f} s", flush=True)

    plot_runtime(
        lengths, runtime,
        title="1D Ising model runtime",
        xlabel="Chain length L",
        filename="ising_1dim.pdf",
    )

# 2D experiment
def run_2d_experiment():
    """
    Run and plot the 2D Ising experiment with respect of runtime and lattice size
    """
    lengths = list(range(3, 7))
    runtime = {name: [] for name in solvers}

    print("\n========================================================\n2D ISING EXPERIMENT\n========================================================", flush=True)

    # Run each solver on the same model parameters.
    for name, solver_class in solvers.items():
        print(f"\n--- {name} ---", flush=True)
        wmc.set_model_counter(solver_class)

        for curr_size in lengths:
            h_interactions, j_interactions = IsingParameters(
                length=curr_size, periodic=False
            ).get_2dim_parameters()

            model = IsingModelWMC(
                num_spins=curr_size ** 2,
                j_interactions=j_interactions,
                h_field=h_interactions,
                beta=1.0,
            )

            start = perf_counter()
            print(f"\n[{name}] START L={curr_size}, spins={curr_size ** 2}; "
                  "solver limit=300 s, container wait=330 s", flush=True)
            try:
                partition_function = model.get_partition_function()
            except Exception:
                print(f"[{name}] FAILED L={curr_size} after "
                      f"{perf_counter() - start:.2f} s", file=sys.stderr, flush=True)
                traceback.print_exc()
                # Leave a gap in the plot and continue with the other cases.
                runtime[name].append(float("nan"))
            else:
                elapsed = perf_counter() - start
                runtime[name].append(elapsed)
                print(f"[{name}] OK L={curr_size}: Z={partition_function}, "
                      f"total elapsed={elapsed:.2f} s", flush=True)

    plot_runtime(
        lengths, runtime,
        title="2D Ising model runtime",
        xlabel="Lattice side length (L x L spins)",
        filename="ising_2dim.pdf",
    )

# Run either experiment, or both
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Ising solver runtimes.")
    parser.add_argument("--experiment", choices=("1d", "2d", "all"), default="all")
    args = parser.parse_args()

    if args.experiment in ("1d", "all"):
        run_1d_experiment()
    if args.experiment in ("2d", "all"):
        run_2d_experiment()
