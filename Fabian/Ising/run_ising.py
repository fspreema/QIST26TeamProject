import argparse
import csv
import sys
from pathlib import Path

# Make the local DiracWMC checkout available to this test script.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "external" / "DiracWMC" / "wcnf_matrix"))

import matplotlib.pyplot as plt
import numpy as np
import wcnf_matrix as wmc
from ising_model import (
    IsingModel,
    TransversalIsingModel,
    get_lattice_parameters,
    get_pairs,
    get_random_bond_ising_parameters,
    get_random_graph_parameters,
)
from local_solvers import TensorOrderLocal

# Shared solver selection
available_solvers = {
    "DPMC": wmc.DPMC,
    "Cachet": wmc.Cachet,
    "TensorOrder": wmc.TensorOrder,
}

# Shared plotting function
def plot_runtime(lengths, runtime, *, title, xlabel, filename):
    """
    Save solver runtimes as CSV and plot them as PDF.
    """

    output_path = Path(__file__).resolve().parent / "Results" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")

    # One row per size, with one runtime column for each solver.
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([xlabel, *runtime])
        for index, size in enumerate(lengths):
            row = [size]
            for solver_name in runtime:
                seconds = runtime[solver_name][index]
                row.append(seconds if np.isfinite(seconds) else "FAILED")
            writer.writerow(row)

    # If no runtime values exist, don't save any plot
    if not any(np.isfinite(t) and t > 0 for times in runtime.values() for t in times):
        print(f"Saved {csv_path}; no positive runtimes to plot", flush=True)
        return

    # Plot figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, times in runtime.items():
        ax.plot(lengths, times, marker="o", linewidth=2, markersize=6, label=name)
    ax.legend(title="Solver", frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Solver runtime (s)")
    ax.set_title(title)
    ax.set_xticks(lengths)
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved {csv_path.name} and {output_path.name} in {output_path.parent}")

def run_model_with_solvers(model, *, case, selected_solvers=None):
    """
    Build one formula, then evaluate it sequentially with each solver.
    """

    if selected_solvers is None:
        selected_solvers = available_solvers
    runtimes = {name: None for name in selected_solvers}
    if not selected_solvers:
        return runtimes

    # Build once and reuse the formula for every solver.
    try:
        formula = model.build_partition_formula()
    except Exception as error:
        print(f"{case}: formula construction failed: {error}", file=sys.stderr)
        return runtimes

    for name, solver_class in selected_solvers.items():
        try:
            result = model.get_partition_function(solver_class, formula=formula)
        except Exception as error:
            print(f"[{name}] {case}: failed: {error}", file=sys.stderr)
        else:
            runtimes[name] = result.runtime
            print(f"[{name}] {case}: Z={result.model_count:.6g}, "
                  f"runtime={result.runtime:.6f} s", flush=True)

    return runtimes


def run_random_graph_experiment():
    """
    Average solver runtimes over five graphs with expected degree three.

    Couplings are uniform on [-1, 1], with no external field. Each graph is
    built once and passed to every solver still active at that size.
    """

    # Init parameters & runtime
    num_spins_list = range(5,85,10)
    num_runs = 5
    runtime = {name: [] for name in available_solvers}
    print("\nRANDOM GRAPH ISING EXPERIMENT", flush=True)

    # Loop over graph sizes
    for curr_size in num_spins_list:

        curr_runtimes = {name: [] for name in available_solvers}
        active_solvers = dict(available_solvers)

        # Loop over the 5 runs per size
        for run in range(num_runs):

            # Create for all 5 runs individual sampled interactions (Rndm. Samlping)
            # -> Therefore needing individual seeds for different sampling
            h_interactions, j_interactions = get_random_graph_parameters(
                length=curr_size,
                seed= run
            )

            # Load model
            model = IsingModel(
                num_spins=curr_size,
                j_interactions=j_interactions,
                h_field=h_interactions,
                beta=1.0,
            )

            # Get results depending on solvers
            results = run_model_with_solvers(
                model,
                case=f"spins={curr_size}, run={run + 1}",
                selected_solvers=active_solvers,
            )

            # Safe runtime
            for name, solver_runtime in results.items():

                if solver_runtime is None:
                    # A failed solver cannot contribute a five-run mean at
                    # this size; continue the remaining instances for others.
                    del active_solvers[name]
                else:
                    curr_runtimes[name].append(solver_runtime)

        # If all 5 runs of same n had finsihed, take mean
        # Else, runtime is NaN and not plotted
        for name, times in curr_runtimes.items():

            if len(times) == num_runs:
                runtime[name].append(np.mean(times))
            else:
                runtime[name].append(float("nan"))

    plot_runtime(
        num_spins_list, runtime,
        title="Random graph Ising model runtime",
        xlabel="Number of spins",
        filename="ising_random_graph.pdf",
    )

def run_2dim_rndm_bnd_ising():
    """
    Average solver runtimes over five graphs with expected degree three.

    Couplings are uniform on [-1, 1], with no external field. Each graph is
    built once and passed to every solver still active at that size.
    """

    # Init parameters & runtime
    lenghts_list = range(2, 8, 2)
    num_runs = 5
    runtime = {name: [] for name in available_solvers}
    print("\n2DIM RANDOM BOND ISING", flush=True)

    # Loop over graph sizes
    for curr_length in lenghts_list:

        curr_runtimes = {name: [] for name in available_solvers}
        active_solvers = dict(available_solvers)
        curr_num_spins = curr_length ** 2

        # Loop over the 5 runs per size
        for run in range(num_runs):

            # Create for all 5 runs individual sampled interactions (Rndm. Samlping)
            # -> Therefore needing individual seeds for different sampling
            j_interactions = get_random_bond_ising_parameters(
                length=curr_length,
                periodic=False,
                dim=2,
                p_ferro=0.8,
                seed= run
            )

            # Load model
            model = IsingModel(
                num_spins=curr_num_spins,
                j_interactions=j_interactions,
                h_field=np.zeros(curr_num_spins),
                beta=1.0,
            )

            # Get results depending on solvers
            results = run_model_with_solvers(
                model,
                case=f"spins={curr_num_spins}, run={run + 1}",
                selected_solvers=active_solvers,
            )

            # Safe runtime
            for name, solver_runtime in results.items():

                if solver_runtime is None:
                    # A failed solver cannot contribute a five-run mean at
                    # this size; continue the remaining instances for others.
                    del active_solvers[name]
                else:
                    curr_runtimes[name].append(solver_runtime)

        # If all 5 runs of same n had finsihed, take mean
        # Else, runtime is NaN and not plotted
        for name, times in curr_runtimes.items():

            if len(times) == num_runs:
                runtime[name].append(np.mean(times))
            else:
                runtime[name].append(float("nan"))

    plot_runtime(
        [length ** 2 for length in lenghts_list],
        runtime,
        title="Random Bond 2dim Ising model runtime",
        xlabel="Number of spins",
        filename="rndm_2dim_ising_bond.pdf",
    )

def run_lattice_experiment(dimensions, lengths, *, title, xlabel, filename):
    """
    Construct one lattice model per size and compare solvers on its formula.
    """

    runtime = {name: [] for name in available_solvers}
    print(f"\n{dimensions}D ISING EXPERIMENT", flush=True)

    for curr_size in lengths:

        # For one fixed length get parameters
        h_interactions, j_interactions = get_lattice_parameters(
            length=curr_size, dim=dimensions, periodic=False
        )

        num_spins = curr_size ** dimensions

        # Load Up model and get part function with all solvers
        model = IsingModel(
            num_spins=num_spins,
            j_interactions=j_interactions,
            h_field=h_interactions,
            beta=1.0,
        )
        results = run_model_with_solvers(
            model, case=f"L={curr_size}, spins={num_spins}"
        )

        # Collect runtime for each solver, if None, set to nan
        for name, solver_runtime in results.items():
            runtime[name].append(
                solver_runtime if solver_runtime is not None else float("nan")
            )

    plot_runtime(lengths, runtime, title=title, xlabel=xlabel, filename=filename)

def run_1d_experiment():
    """
    Compare solvers for open chains.
    """

    run_lattice_experiment(
        1, list(range(2, 20)),
        title="1D Ising model runtime",
        xlabel="Chain length L",
        filename="ising_1dim.pdf",
    )

def run_2d_experiment():
    """
    Compare solvers for open square lattices.
    """

    run_lattice_experiment(
        2, list(range(2, 14, 2)),
        title="2D Ising model runtime",
        xlabel="Lattice side length (L x L spins)",
        filename="ising_2dim.pdf",
    )

def run_transversal_experiment():
    """
    Compare solvers for open transverse-field chains at fixed Trotter steps.
    """
    lengths = list(range(2, 9))
    trotter_steps = 15
    runtime = {name: [] for name in available_solvers}
    print("\nTRANSVERSE-FIELD ISING EXPERIMENT", flush=True)

    for length in lengths:
        pairs = get_pairs(length=length, dim=1, periodic=False)
        model = TransversalIsingModel(
            num_spins=length,
            j_constant=1.0,
            g_constant=1.0,
            beta=1.0,
            trotter_steps=trotter_steps,
            pairs=pairs,
        )
        results = run_model_with_solvers(
            model, case=f"L={length}, Trotter steps={trotter_steps}"
        )
        for name, solver_runtime in results.items():
            runtime[name].append(
                solver_runtime if solver_runtime is not None else float("nan")
            )

    plot_runtime(
        lengths, runtime,
        title=f"Transverse-field Ising runtime ({trotter_steps} Trotter steps)",
        xlabel="Chain length L",
        filename="ising_transversal.pdf",
    )

# Run either experiment, or all
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Ising solver runtimes.")
    parser.add_argument(
        "--experiment",
        choices=("1d", "2d", "random", "transversal", "all", "2dim_bond"),
        default="all"
    )
    parser.add_argument(
        "--solvers", nargs="+", choices=tuple(available_solvers),
        default=list(available_solvers),
        help="Default: DPMC, TensorOrder, and Cachet.",
    )
    args = parser.parse_args()
    available_solvers = {name: available_solvers[name] for name in args.solvers}
    
    if args.experiment in ("1d", "all"):
        run_1d_experiment()

    if args.experiment in ("2d", "all"):
        run_2d_experiment()

    if args.experiment in ("random", "all"):
        run_random_graph_experiment()

    if args.experiment in ("2dim_bond", "all"):
        run_2dim_rndm_bnd_ising()

    if args.experiment in ("transversal", "all"):
        run_transversal_experiment()
