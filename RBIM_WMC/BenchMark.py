from base_model import BaseModel
from wcnf_matrix import ModelCounter
import time
import matplotlib.pyplot as plt
import numpy as np


class BenchMark:

    def __init__(self, Model: BaseModel, solver: ModelCounter):
        self._Model:BaseModel = Model
        self._model_counter = solver()

    def runtime_vs_lattice_benchMark(self, shapes: list, average: int = 5, beta: float = 1.0):

        runtime_values = []
        # runtime_errors = []

        print(" = Start Benchmark =")
        for shape in shapes:
            models = []
            for _ in range(average):
                models.append(self._Model.generate_lattice(shape)) # We first create various models with the same size so we can average the time it takes to solve them
            
            measured_runtimes = self.calculate_runtimes(models, beta)
            average_runtime = np.average(measured_runtimes)
            runtime_values.append(average_runtime)

            print(f"shape = {shape} -> {average_runtime} s")

        number_of_spins = [shape[0] * shape[1] for shape in shapes]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(number_of_spins, runtime_values, marker="o", linewidth=1.5, markersize=5)

        ax.set_yscale("log")
        ax.set_xlabel("N (# spins)")
        ax.set_ylabel("Runtime average (s)")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        fig.tight_layout()
        #fig.savefig(f"RBMI_WMC/Plots/Runtime_vs_Lattice_B_{beta}.png", dpi=150)
        plt.show()

    def calculate_runtimes(self, models: list[BaseModel], beta: float) -> list[float]:

        problems = [model.get_wcnf() for model in models]

        runtimes = []

        failed = False
        for result in self._model_counter.batch_model_count(*problems):
            if not result.success:
                    failed = True
                    break
            if failed:
                print("FAILURE")
                break
            runtimes.append(float(result.runtime))
        return runtimes
    

    def calculate_err(self, model: BaseModel, beta: float, real_value: float) -> float:

        exp_value = model.partition_function(beta)
        return abs(exp_value - real_value)/real_value


