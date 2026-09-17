from wcnf_matrix import *
import numpy as np
import time
import matplotlib.pyplot as plt

SOLVERS = {
    "DPMC": DPMC,
    "Cachet": Cachet,
    "TensorOrder": TensorOrder,
}

def ising_partition_function_wmc(n_spins, interaction, external_field, beta, solver_name):
    spin_vars = [BoolVar() for _ in range(n_spins)]
    interactions = list(interaction.items())
    aux_vars = [BoolVar() for _ in interactions]

    clauses = []
    for ((i, j), _), aux in zip(interactions, aux_vars):
        vi, vj = spin_vars[i], spin_vars[j]
        clauses.append([vi, -vj, -aux])
        clauses.append([vi, vj, aux])
        clauses.append([-vi, vj, -aux])
        clauses.append([-vi, -vj, aux])

    cnf = CNF(clauses)
    all_vars = spin_vars + aux_vars
    weight_func = WeightFunction(all_vars)

    for var, h in zip(spin_vars, external_field):
        weight_func[var, True] = np.exp(beta * h)
        weight_func[var, False] = np.exp(-beta * h)

    for ((_, _), J), aux in zip(interactions, aux_vars):
        weight_func[aux, True] = np.exp(beta * J)
        weight_func[aux, False] = np.exp(-beta * J)

    counter_cls = SOLVERS[solver_name]
    counter = counter_cls()
    result = counter.model_count(cnf, weight_func)
    return result.model_count


def make_chain_ising(n_spins, J=1.5, h=0.5, periodic=False):
    interaction = {(i, i + 1): J for i in range(n_spins - 1)}
    if periodic and n_spins > 2:
        interaction[(0, n_spins - 1)] = J
    external_field = [h] * n_spins
    return interaction, external_field


def warmup_solvers(solver_names, n_warmup=3, beta=1.0, J=1.5, h=0.5):
    """Ejecuta una llamada 'de mentira' por solver para absorber el coste
    de arranque (p.ej. Docker) antes de medir tiempos en serio."""
    interaction, external_field = make_chain_ising(n_warmup, J=J, h=h)
    print("Calentando solvers...")
    for solver_name in solver_names:
        t0 = time.perf_counter()
        ising_partition_function_wmc(n_warmup, interaction, external_field, beta, solver_name)
        elapsed = time.perf_counter() - t0
        print(f"  {solver_name:12s} warm-up: {elapsed:.4f}s (descartado)")
    print()

def benchmark(n_values, solver_names, beta=1.0, J=1.5, h=0.5, promediar=1):
    results = []

    for n in n_values:
        interaction, external_field = make_chain_ising(n, J=J, h=h)

        for solver_name in solver_names:
            times = []
            Z = None

            # Repetir la medida "promediar" veces
            for _ in range(promediar):
                t0 = time.perf_counter()

                Z = ising_partition_function_wmc(
                    n, interaction, external_field,
                    beta, solver_name
                )

                elapsed = time.perf_counter() - t0
                times.append(elapsed)

            # Media y desviación estándar
            mean_time = np.mean(times)

            # Si solo hay una medida, no hay error estadístico
            if promediar > 1:
                std_time = np.std(times, ddof=1)
            else:
                std_time = 0.0

            results.append({
                "n": n,
                "solver": solver_name,
                "Z": Z,
                "time_seconds": mean_time,
                "time_error": std_time
            })

            print(
                f"n={n:3d} | {solver_name:12s} | "
                f"Z={Z:.6f} | "
                f"t={mean_time:.4f} ± {std_time:.4f}s"
            )

    return results


def plot_benchmark(results, log_y=True, guardar=False):
    solver_names = sorted(set(r["solver"] for r in results))

    plt.figure(figsize=(9, 6))

    for solver_name in solver_names:
        pts = [
            (r["n"], r["time_seconds"], r["time_error"])
            for r in results
            if r["solver"] == solver_name
        ]
        pts.sort()

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errors = [p[2] for p in pts]

        plt.errorbar(
            xs,
            ys,
            yerr=errors,
            marker="o",
            capsize=4,
            label=solver_name
        )

    if log_y:
        plt.yscale("log")

    plt.xlabel("Number of Spins (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("WMC solver comparison — 1D classical Ising model (longitudinal field)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if guardar:
        plt.savefig(
            "Diego/Figures/benchmark_ising_classical_wmc.png",
            dpi=150
        )

    plt.show()


# --- Ejecución ---
if __name__ == "__main__":
    n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    solver_names = ["DPMC", "Cachet", "TensorOrder"]

    promediar = 5

    warmup_solvers(solver_names)

    results = benchmark(
        n_values,
        solver_names,
        promediar=promediar
    )

    plot_benchmark(
        results,
        guardar=True
    )