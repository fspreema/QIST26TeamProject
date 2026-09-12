import argparse
import csv
import sys
import traceback
from pathlib import Path
from time import perf_counter

# Make the local DiracWMC checkout available to this test script.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "external" / "DiracWMC" / "wcnf_matrix"))

import matplotlib.pyplot as plt
import numpy as np
import wcnf_matrix as wmc
from ising_model import IsingModelWMC, IsingParameters
from local_solvers import CachetExperimental, TensorOrderLocal

# Shared solver selection
available_solvers = {
    "DPMC": wmc.DPMC,
    "Cachet": wmc.Cachet,
    "TensorOrder": TensorOrderLocal,
    "CachetExperimental": CachetExperimental,
}
# Include all three counter families; retain the experimental Cachet label.
default_solvers = ("DPMC", "TensorOrder", "Cachet")
solvers = {name: available_solvers[name] for name in default_solvers}

# Shared plotting function
def plot_runtime(lengths, runtime, *, title, xlabel, filename):
    """
    Save Runtime aagainst lattice size wuth CSV and print Plot as PDF
    """

    # Preserve every selected counter in the output, including failed batches.
    csv_path = Path(__file__).resolve().parent / Path(filename).with_suffix(".csv")
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([xlabel, *runtime])
        print(f"\n{title}: solver runtime summary", flush=True)
        print(" | ".join([xlabel, *runtime]), flush=True)
        for index, size in enumerate(lengths):
            values = [times[index] if np.isfinite(times[index]) else "FAILED"
                      for times in runtime.values()]
            writer.writerow([size, *values])
            print(" | ".join([str(size), *[f"{v:.6f}" if isinstance(v, (float, np.floating))
                                         else str(v) for v in values]]), flush=True)
    print(f"Results saved to {csv_path}", flush=True)
    if not any(np.isfinite(t) and t > 0 for times in runtime.values() for t in times):
        print(f"No positive, finite runtimes for {title}; plot not saved", flush=True)
        return
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
    output_path = Path(__file__).resolve().parent / filename
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Plot saved to {output_path}")

def run_model_with_solvers(model, *, case, selected_solvers=None):
    """
    Build one formula, then evaluate it sequentially with each solver.
    """

    # a) Fall back to the full registry of solvers if the caller didn't
    #    hand-pick a subset (e.g. when running the full DPMC/Cachet/TensorOrder sweep)
    if selected_solvers is None:
        selected_solvers = solvers

    # Every solver starts as None -> stays None if it never successfully
    # returns a runtime (formula failure, solver crash, or invalid Z)
    runtimes = {name: None for name in selected_solvers}
    print(f"\n--- {case} ---", flush=True)


    # b) Build the WCNF formula once for this case. It's identical for every
    # solver -> building & reusing it below avoids compute
    build_start = perf_counter()

    try:
        formula = model.build_partition_formula()
    except Exception:
        # No formula -> no solver can run this case at all, so log once and return early
        print(f"Formula construction FAILED for {case}", file=sys.stderr, flush=True)
        traceback.print_exc()

        return runtimes
    
    build_elapsed = perf_counter() - build_start

    # c) Run each solver against the shared formula
    for name, solver_class in selected_solvers.items():

        start = perf_counter()
        limit = getattr(solver_class, "default_timeout", 300)
        print(f"[{name}] START {case}; solver limit={limit} s, container wait={limit + 30} s",
              flush=True)
        
        try:
            # formula=formula reuses the shared build instead of having
            # get_partition_function build its own
            result = model.get_partition_function(solver_class, formula=formula)

        except Exception:
            # Catches solver-side crashes
            print(f"[{name}] FAILED {case} after {perf_counter() - start:.2f} s",
                  file=sys.stderr, flush=True)
            traceback.print_exc()

        else:
            # If no crash, report time, log results & time, print timing info
            elapsed = perf_counter() - start
            runtimes[name] = result.runtime
            print(
                f"[{name}] OK {case}: Z={result.model_count}, "
                f"solver runtime={result.runtime:.6f} s, "
                f"solver call elapsed={elapsed:.2f} s "
                f"(shared formula build={build_elapsed:.2f} s)",
                flush=True,
            )

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
    runtime = {name: [] for name in solvers}
    print("\nRANDOM GRAPH ISING EXPERIMENT", flush=True)

    # Loop over graph sizes
    for curr_size in num_spins_list:

        curr_runtimes = {name: [] for name in solvers}
        active_solvers = dict(solvers)

        # Loop over the 5 runs per size
        for run in range(num_runs):

            # Create for all 5 runs individual sampled interactions (Rndm. Samlping)
            h_interactions, j_interactions = IsingParameters(
                length=curr_size, periodic=False
            ).random_sampling(seed=run)

            # Load model
            model = IsingModelWMC(
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
                print(f"[{name}] spins={curr_size}: incomplete five-run batch; "
                      "no average plotted", flush=True)

    plot_runtime(
        num_spins_list, runtime,
        title="Random graph Ising model runtime",
        xlabel="Number of spins",
        filename="ising_random_graph.pdf",
    )


def run_lattice_experiment(dimensions, lengths, *, title, xlabel, filename):
    """
    Construct one lattice model per size and compare solvers on its formula.
    """

    runtime = {name: [] for name in solvers}
    print(f"\n{dimensions}D ISING EXPERIMENT", flush=True)

    for curr_size in lengths:

        # For one fixed length get parameters
        parameters = IsingParameters(length=curr_size, periodic=False)

        if dimensions == 1:
            h_interactions, j_interactions = parameters.get_1dim_parameters()
        else:
            h_interactions, j_interactions = parameters.get_2dim_parameters()

        num_spins = curr_size ** dimensions

        # Load Up model and get part function with all solvers
        model = IsingModelWMC(
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

# Run either experiment, or all
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Ising solver runtimes.")
    parser.add_argument(
        "--experiment",
        choices=("1d", "2d", "random", "all"),
        default="all"
    )
    parser.add_argument(
        "--solvers", nargs="+", choices=tuple(available_solvers),
        default=list(default_solvers),
        help="Default: DPMC, fixed TensorOrder, and experimental Cachet (30 s limit).",
    )
    args = parser.parse_args()
    
    if args.experiment in ("1d", "all"):
        run_1d_experiment()

    if args.experiment in ("2d", "all"):
        run_2d_experiment()

    if args.experiment in ("random", "all"):
        run_random_graph_experiment()
