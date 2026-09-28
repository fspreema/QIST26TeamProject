from base_model import BaseModel
from wcnf_matrix import ModelCounter
import time
import matplotlib.pyplot as plt
import numpy as np
import os


class BenchMark:

    def __init__(self, Model: BaseModel, solver: ModelCounter):
        self._Model:BaseModel = Model
        self._model_counter = solver()

    def runtime_vs_lattice_benchMark(self, shapes: list, average: int = 5, beta: float = 1.0):

        runtime_values = []

        print(" = Start Benchmark =")
        for shape in shapes:
            models = []
            for _ in range(average):
                models.append(self._Model.generate_lattice(shape)) # We first create various models with the same size so we can average the time it takes to solve them

            measured_runtimes, _ = self.run_solver(models, beta)
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

    def error_vs_lattice_benchMark(self, shapes: list, average: int = 5, beta: float = 1.0):

        runtime_values = []
        error_values = []

        print(" = Start Benchmark =")
        for shape in shapes:
            models = []
            for _ in range(average):
                models.append(self._Model.generate_lattice(shape)) # We first create various models with the same size so we can average the time it takes to solve them

            true_values = []
            try:
                true_values = [model.partition_function(beta) for model in models]
            except np._core._exceptions._ArrayMemoryError:
                print(f"partition function for shape {shape} was to large to fit into memory")
                break

            measured_runtimes, measured_errors = self.run_solver(models, beta, true_values)
            average_runtime = np.average(measured_runtimes)
            average_error = np.average(measured_errors)
            runtime_values.append(average_runtime)
            error_values.append(average_error)

            print(f"shape = {shape} -> {average_runtime} s , {average_error}")

        number_of_spins = [shape[0] * shape[1] for shape in shapes]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(number_of_spins, error_values, marker="o", linewidth=1.5, markersize=5)

        ax.set_yscale("log")
        ax.set_xlabel("N (# spins)")
        ax.set_ylabel("Relative error compared to exact solution")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        fig.tight_layout()
        #fig.savefig(f"RBMI_WMC/Plots/Runtime_vs_Lattice_B_{beta}.png", dpi=150)
        plt.show()

    def magnetization_vs_beta_benchMark(self, shape: list, p_ferro: float = 1.0):
        print("=== Calculating Magnetization ===")

        model = self._Model.generate_lattice((shape[0], shape[1]), p_ferro=p_ferro)
    
        magnetization = []
    
        for beta in np.linspace(0.0, 5.0, 10):
    
            print("Computing M for beta = ", beta)
    
            problem_Z = model.get_wcnf(beta)
            result_Z = self._model_counter.model_count(*problem_Z)
            value_Z = result_Z.model_count
    
            magnetization_sum = 0.0
            n_spins = len(model)
    
            for i in range(n_spins):
                problem_op = model.get_wcnf_single_spin(target_spin = i, beta=beta)
                result_op = self._model_counter.model_count(*problem_op)
                value_Z_op = result_op.model_count
    
                magnetization_sum += (value_Z_op/value_Z)
    
            magnetization.append(magnetization_sum / n_spins)

        betas = np.linspace(0.0, 5.0, 10)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(betas, magnetization, marker="o", linewidth=1.5, markersize=5)
    
        ax.set_xlabel("Beta")
        ax.set_ylabel("Magnetization")
    
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    
        fig.tight_layout()

        output_dir = "Plots/Magnetization_vs_Beta"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/Magnetization_vs_Beta_{shape}.png", dpi=150)
    
        return magnetization

    def magnetization_squared_vs_beta_benchMark(self, shape: list, p_ferro: float):
        print("=== Calculating Magnetization Squared ===")

        model = self._Model.generate_lattice((shape[0], shape[1]), p_ferro=p_ferro)
    
        magnetization_squared = []
    
        for beta in np.linspace(0.0, 5.0, 10):
    
            print("Computing M^2 for beta = ", beta)
    
            problem_Z = model.get_wcnf(beta)
            result_Z = self._model_counter.model_count(*problem_Z)
            value_Z = result_Z.model_count
    
            magnetization_sum = 0.0
            n_spins = len(model)
    
            for i in range(n_spins):
                for j in range(n_spins):
                    if i == j: magnetization_sum += 1.0; continue # For <s_i s_i> = 1, no calculation needed
                    problem_op = model.get_wcnf_double_spin(target_spins = (i,j), beta=beta)
                    result_op = self._model_counter.model_count(*problem_op)
                    value_Z_op = result_op.model_count
    
                    magnetization_sum += (value_Z_op/value_Z)
    
            magnetization_squared.append(magnetization_sum / (n_spins**2))

        magnetization = np.sqrt(magnetization_squared)

        betas = np.linspace(0.0, 5.0, 10)
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(betas, magnetization, marker="o", linewidth=1.5, markersize=5)
    
        ax.set_xlabel("Beta")
        ax.set_ylabel(r"$\sqrt{\text{M}^2}$")
    
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    
        fig.tight_layout()

        output_dir = "Plots/Magnetization_squared_vs_Beta"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/Magnetization_squared_vs_Beta_{shape}.png", dpi=150)
    
        return magnetization


    def run_solver(self, models: list[BaseModel], beta: float, true_values: list[float] = None) -> tuple[list[float], list[float]]:
        """ Solve the provided models by taking the WCNF and running it through
         the solver. Returns runtimes, and relative errors when true values are
          provided. """
        problems = [model.get_wcnf(beta) for model in models]

        runtimes = []
        rel_errors = []

        failed = False
        for i, result in enumerate(self._model_counter.batch_model_count(*problems)):
            if not result.success:
                    failed = True
                    break
            if failed:
                print("FAILURE")
                break

            runtimes.append(float(result.runtime))
            if true_values is not None:
                rel_error = abs(result.model_count / true_values[i] - 1)
                rel_errors.append(rel_error)

        return runtimes, rel_errors