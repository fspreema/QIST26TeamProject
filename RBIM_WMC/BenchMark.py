from RBIM_model import RandomBondIsingModel
import time
import matplotlib as plt


class BenchMark:

    def __init__(self, Model: RandomBondIsingModel, solver: str):
        self._Model:RandomBondIsingModel = Model
        self._solver = solver

    def runtime_vs_lattice_benchMark(self, sizes: list, average: int = 5, beta: float = 1.0):

        runtime_values = []
        # runtime_errors = []

        for size in sizes:
            models = []
            for _ in range(average):
                models.append(self._Model.generate_lattice(size)) # We first create various models with the same size so we can average the time it takes to solve them
            runtime_values.append(sum(self.calculate_runtime(models, beta)) / average)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(sizes, runtime_values, marker="o", linewidth=1.5, markersize=5)

        ax.set_yscale("log")
        ax.set_xlabel("N (# spins)")
        ax.set_ylabel("Runtime average (s)")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        fig.tight_layout()
        fig.savefig(f"RBMI_WMC/Plots/Runtime_vs_Lattice_B_{beta}.png", dpi=150)
        plt.show()

    def calculate_runtime(self, models: list[RandomBondIsingModel], beta: float) -> list[float]:
        results = []
        for m in models:
            tstart_raw = time.time()
            true_value = m.partition_function(beta)
            tend_raw = time.time()
            result = tend_raw - tstart_raw
            results.append(result)
        return results

    def calculate_err(self, model: RandomBondIsingModel, beta: float, real_value: float) -> float:

        exp_value = model.partition_function(beta)
        return abs(exp_value - real_value)/real_value


