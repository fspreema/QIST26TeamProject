from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field

_here = os.path.dirname(os.path.abspath(__file__))
_dirac_root = os.path.join(_here, "..", "external", "DiracWMC")
sys.path.insert(0, _dirac_root)
sys.path.insert(0, os.path.join(_dirac_root, "wcnf_matrix"))

from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder

from experiments.ising.ising_model import IsingModel
from experiments.ising.converter import ising_to_wcnf 

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder)
BETA = 1.0


@dataclass
class SolverRunRecord:
    solver: str
    lattice: str
    size: int
    num_vars: int
    num_clauses: int
    success: bool
    avg_runtime_s: float | None
    avg_rel_error: float | None
    error_message: str | None
    per_run_runtimes: list[float] = field(default_factory=list)


def generate_square_lattice(side: int, seed: int) -> IsingModel:
    rng = random.Random(seed)
    n = side * side
    model = IsingModel(n)
    for i in range(n):
        if i + side < n:
            model[i, i + side] = rng.normalvariate(0, 1)
        if i % side != side - 1:
            model[i, i + 1] = rng.normalvariate(0, 1)
    for i in range(n):
        model[i] = rng.normalvariate(0, 1)
    return model


def generate_random_graph(size: int, expected_degree: float, seed: int) -> (
IsingModel):
    rng = random.Random(seed)
    model = IsingModel(size)
    for i in range(size):
        for j in range(i):
            if rng.uniform(0, 1) < expected_degree / size:
                model[i, j] = rng.uniform(-1, 1)
    return model

MAX_EXACT_SPINS = 20


def run_one(solver_cls: type[ModelCounter], model: IsingModel, lattice: str,
size: int, runs: int) -> SolverRunRecord:
    cnf, weight_func = ising_to_wcnf(model, BETA)
    num_vars = len(list(weight_func))
    num_clauses = cnf.num_clauses
    n_spins = len(model)
    if n_spins <= MAX_EXACT_SPINS and hasattr(model, "partition_function"):
        exact = model.partition_function(BETA)
    else:
        exact = None
        if n_spins > MAX_EXACT_SPINS:
            print(f"  (skipping exact brute-force check: {n_spins} spins, "
            f"2**{n_spins} configurations is not tractable in pure Python)")

    runtimes: list[float] = []
    rel_errors: list[float] = []
    error_message: str | None = None

    try:
        counter = solver_cls()
    except Exception as exc:
        return SolverRunRecord(solver_cls.__name__, lattice, size, num_vars,
        num_clauses, False, None, None,
        f"{type(exc).__name__} constructing solver: {exc}")

    for _ in range(runs):
        try:
            result = counter.model_count(cnf, weight_func)
        except Exception:
            error_message = "exception during model_count:\n" + \
            traceback.format_exc(limit=2)
            break
        if not result.success:
            error_message = "solver returned ModelCounterResult(success=False)"
            break
        runtimes.append(result.runtime)
        if exact:
            rel_errors.append(abs(result.model_count / exact - 1))

    success = len(runtimes) == runs and runs > 0
    return SolverRunRecord(
        solver=solver_cls.__name__, lattice=lattice, size=size,
        num_vars=num_vars, num_clauses=num_clauses, success=success,
        avg_runtime_s=(sum(runtimes) / len(runtimes)) if runtimes else None,
        avg_rel_error=(sum(rel_errors) / len(rel_errors)) if rel_errors else None,
        error_message=error_message, per_run_runtimes=runtimes,
    )


def main():
    parser = argparse.ArgumentParser(description="Three-solver speed benchmark on the classical Ising model.")
    parser.add_argument("--lattice", choices=["square", "random", "both"],
    default="both")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=8,
    help="Side length for square lattice, node count (x10) for random graph")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lattices = ["square", "random"] if args.lattice == "both" else [args.lattice]
    records: list[SolverRunRecord] = []

    for lattice in lattices:
        for solver_cls in SOLVERS:
            print(f"\n=== {lattice} lattice, {solver_cls.__name__} ===")
            stop_growing = False
            for size in range(args.min_size, args.max_size + 1):
                if stop_growing:
                    break
                if lattice == "square":
                    model = generate_square_lattice(size, args.seed)
                    label = size
                else:
                    n_nodes = size * 10
                    model = generate_random_graph(n_nodes, 3.0, args.seed)
                    label = n_nodes
                t0 = time.time()
                rec = run_one(solver_cls, model, lattice, label, args.runs)
                records.append(rec)
                wall = time.time() - t0
                if rec.success:
                    err = f"{rec.avg_rel_error:.2e}" if rec.avg_rel_error is not None else "n/a"
                    print(f"  size={label:4d}  vars={rec.num_vars:4d}  "
                    f"avg_runtime={rec.avg_runtime_s:.4f}s  rel_error={err}  "
                    f"(wall {wall:.1f}s)")
                else:
                    print(f"  size={label:4d}  vars={rec.num_vars:4d}  "
                    f"FAILED: {rec.error_message}")
                    stop_growing = True

    out_csv = os.path.join(_here, "ising_benchmark_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "lattice", "size", "num_vars",
        "num_clauses", "success", "avg_runtime_s", "avg_rel_error",
        "error_message"])
        for r in records:
            writer.writerow([r.solver, r.lattice, r.size, r.num_vars,
            r.num_clauses, r.success,
            r.avg_runtime_s if r.avg_runtime_s is not None else "",
            r.avg_rel_error if r.avg_rel_error is not None else "",
            (r.error_message or "").replace("\n", " | ")])
    print(f"\nWrote {out_csv}")

    make_plot(records, lattices, os.path.join(_here, "ising_benchmark_plot.png"))


def make_plot(records, lattices, out_path: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(lattices), figsize=(6.5 * len(lattices), 5),
    squeeze=False)
    axes = axes[0]

    for ax, lattice in zip(axes, lattices):
        by_solver: dict[str, list] = {}
        for r in records:
            if r.lattice == lattice:
                by_solver.setdefault(r.solver, []).append(r)
        for solver, recs in by_solver.items():
            xs = [r.size for r in recs if r.success]
            ys = [r.avg_runtime_s for r in recs if r.success]
            if xs:
                ax.plot(xs, ys, marker="o", label=solver)
        ax.set_xlabel("System size (nodes)")
        ax.set_ylabel("Average runtime (s)")
        ax.set_title(f"Classical Ising, {lattice} graph")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
