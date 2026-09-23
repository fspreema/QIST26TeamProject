from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

_here = os.path.dirname(os.path.abspath(__file__))
_dirac_root = os.path.join(_here, "..", "external", "DiracWMC")
sys.path.insert(0, _dirac_root)
sys.path.insert(0, os.path.join(_dirac_root, "wcnf_matrix"))

from wcnf_matrix import ModelCounter, DPMC, Cachet, TensorOrder

from experiments.quantum_ising.quantum_ising_model import ( 
    QuantumIsingModel,
)
from experiments.quantum_ising.converter import quantum_ising_to_wcnf

SOLVERS: tuple[type[ModelCounter], ...] = (DPMC, Cachet, TensorOrder)
BETA = 1.0


@dataclass
class SolverRunRecord:
    solver: str
    n_spins: int
    trotter_layers: int
    num_vars: int
    num_clauses: int
    success: bool
    avg_runtime_s: float | None
    avg_rel_error: float | None
    error_message: str | None
    per_run_runtimes: list[float] = field(default_factory=list)


def make_chain_model(n_spins: int) -> QuantumIsingModel:
    model = QuantumIsingModel(n_spins)
    for i in range(n_spins - 1):
        model[i, i + 1] = 1.0
    model.external_field_x = 0.5
    model.external_field_z = 0.1
    return model


def run_one(solver_cls: type[ModelCounter], n_spins: int, trotter_layers: int,
runs: int) -> SolverRunRecord:
    MAX_EXACT_SPINS = 12
    model = make_chain_model(n_spins)
    exact = model.partition_function(BETA) if n_spins <= MAX_EXACT_SPINS else None
    cnf, weight_func = quantum_ising_to_wcnf(model, BETA, trotter_layers)
    num_vars = len(list(weight_func))
    num_clauses = cnf.num_clauses

    runtimes: list[float] = []
    rel_errors: list[float] = []
    error_message: str | None = None

    try:
        counter = solver_cls()
    except Exception as exc:
        return SolverRunRecord(solver_cls.__name__, n_spins, trotter_layers,
        num_vars, num_clauses, False, None, None,
        f"{type(exc).__name__} constructing solver: {exc}")

    for _ in range(runs):
        try:
            result = counter.model_count(cnf, weight_func)
        except Exception:
            error_message = "exception during model_count:\n" + \
            traceback.format_exc(limit=2)
            break
        if not result.success:
            error_message = ("solver returned ModelCounterResult(success="
            "False) -- check the solver's own container log for the root "
            "cause (this is the expected outcome for Cachet/TensorOrder on "
            "TFIM, see module docstring)")
            break
        runtimes.append(result.runtime)
        if exact:
            rel_errors.append(abs(result.model_count / exact - 1))

    success = len(runtimes) == runs and runs > 0
    return SolverRunRecord(
        solver=solver_cls.__name__,
        n_spins=n_spins,
        trotter_layers=trotter_layers,
        num_vars=num_vars,
        num_clauses=num_clauses,
        success=success,
        avg_runtime_s=(sum(runtimes) / len(runtimes)) if runtimes else None,
        avg_rel_error=(sum(rel_errors) / len(rel_errors)) if rel_errors else None,
        error_message=error_message,
        per_run_runtimes=runtimes,
    )


def main():
    parser = argparse.ArgumentParser(description="Three-solver speed/capability benchmark on the transverse-field Ising model (TFIM).")
    parser.add_argument("--min-spins", type=int, default=2)
    parser.add_argument("--max-spins", type=int, default=8)
    parser.add_argument("--trotter-layers", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3,
    help="Repetitions per (solver, size), averaged")
    args = parser.parse_args()

    records: list[SolverRunRecord] = []
    for solver_cls in SOLVERS:
        print(f"\n=== {solver_cls.__name__} ===")
        stop_growing = False
        for n_spins in range(args.min_spins, args.max_spins + 1):
            if stop_growing:
                break
            t0 = time.time()
            rec = run_one(solver_cls, n_spins, args.trotter_layers, args.runs)
            records.append(rec)
            wall = time.time() - t0
            if rec.success:
                print(f"  n_spins={n_spins:3d}  vars={rec.num_vars:4d}  "
                f"avg_runtime={rec.avg_runtime_s:.4f}s  "
                f"rel_error={rec.avg_rel_error:.2e}  (wall {wall:.1f}s)")
            else:
                print(f"  n_spins={n_spins:3d}  vars={rec.num_vars:4d}  "
                f"FAILED: {rec.error_message}")
                stop_growing = True

    out_csv = os.path.join(_here, "tfim_benchmark_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "n_spins", "trotter_layers", "num_vars",
        "num_clauses", "success", "avg_runtime_s", "avg_rel_error",
        "error_message"])
        for r in records:
            writer.writerow([r.solver, r.n_spins, r.trotter_layers,
            r.num_vars, r.num_clauses, r.success,
            r.avg_runtime_s if r.avg_runtime_s is not None else "",
            r.avg_rel_error if r.avg_rel_error is not None else "",
            (r.error_message or "").replace("\n", " | ")])
    print(f"\nWrote {out_csv}")

    make_plot(records, os.path.join(_here, "tfim_benchmark_plot.png"))


def make_plot(records: list[SolverRunRecord], out_path: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    by_solver: dict[str, list[SolverRunRecord]] = {}
    for r in records:
        by_solver.setdefault(r.solver, []).append(r)

    any_success = False
    for solver, recs in by_solver.items():
        xs = [r.n_spins for r in recs if r.success]
        ys = [r.avg_runtime_s for r in recs if r.success]
        if xs:
            any_success = True
            ax.plot(xs, ys, marker="o", label=solver)
        n_fail = sum(1 for r in recs if not r.success)
        if n_fail:
            first_fail = next(r for r in recs if not r.success)
            ax.annotate(f"{solver}: fails from n={first_fail.n_spins}",
            xy=(0.02, 0.98 - 0.06 * list(by_solver).index(solver)),
            xycoords="axes fraction", fontsize=8, va="top")

    ax.set_xlabel("Number of spins (chain length)")
    ax.set_ylabel("Average runtime (s)")
    ax.set_title("TFIM: DPMC vs. Cachet vs. TensorOrder")
    if any_success:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
