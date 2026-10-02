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

    def runtime_vs_lattice_benchMark(self, shapes: list, average: int = 5, beta: float = 1.0, plot: bool = False):

        runtime_values = []

        print(" = Start Benchmark =")
        for shape in shapes:
            models = []
            for _ in range(average):
                models.append(self._Model.generate_lattice(shape)) # We first create various models with the same size so we can average the time it takes to solve them

            measured_runtimes, _, _ = self.run_solver(models, beta)
            average_runtime = np.average(measured_runtimes)
            runtime_values.append(average_runtime)

            print(f"shape = {shape} -> {average_runtime} s")

        number_of_spins = [shape[0] * shape[1] for shape in shapes]

        if plot:
            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(number_of_spins, runtime_values, marker="o", linewidth=1.5, markersize=5)

            ax.set_yscale("log")
            ax.set_xlabel("N (# spins)")
            ax.set_ylabel("Runtime average (s)")

            ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

            fig.tight_layout()

            output_dir = "Plots/Runtime_vs_Lattice"
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/Runtime_vs_Lattice_B_{beta}.png", dpi=150)

            plt.show()

        return runtime_values

    def error_vs_lattice_benchMark(self, shapes: list, average: int = 5, beta: float = 1.0, plot: bool = False):

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

            measured_runtimes, measured_errors, _ = self.run_solver(models, beta, true_values)
            average_runtime = np.average(measured_runtimes)
            average_error = np.average(measured_errors)
            runtime_values.append(average_runtime)
            error_values.append(average_error)

            print(f"shape = {shape} -> {average_runtime} s , {average_error}")

        number_of_spins = [shape[0] * shape[1] for shape in shapes]

        if (plot):
            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(number_of_spins, error_values, marker="o", linewidth=1.5, markersize=5)

            ax.set_yscale("log")
            ax.set_xlabel("N (# spins)")
            ax.set_ylabel("Relative error compared to exact solution")

            ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

            fig.tight_layout()
            fig.savefig(f"RBMI_WMC/Plots/Runtime_vs_Lattice_B_{beta}.png", dpi=150)
            plt.show()

        return runtime_values, error_values

    def Generate_GoundStateEnergy(self, shape: list, average: int = 5):
        """This function calculates the Ground State Energy from a system by performing
        the discrete derivative of -dln(Z)/dbeta for large beta"""

        print(" = Start Benchmark =")

        betas = np.arange(0.0, 5.1, 0.2)
        ln_Z_values = []

        for beta in betas:
            models = []
            for _ in range(average):
                models.append(self._Model.generate_lattice(shape)) # We first create various models with the same size so we can average the time it takes to solve them

            measured_runtimes, measured_errors, Z_values = self.run_solver(models, float(beta))

            mean_Z = np.mean(Z_values)
            ln_Z = np.log(mean_Z)
            ln_Z_values.append(ln_Z)

            print(f"shape = {shape} | beta = {beta:.1f} -> ln(Z) = {ln_Z:.4f}")

        State_Energy = -np.gradient(ln_Z_values, betas)

        solver_name = self._model_counter.__class__.__name__
        label_text = f"{solver_name} (L={shape})"

        # --- Plot ln(Z) vs beta ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(betas, ln_Z_values, marker="o", linewidth=1.5, markersize=5, label=label_text)

        ax.set_xlabel("Beta")
        ax.set_ylabel("ln(Z)")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        ax.legend(loc="best")
        fig.tight_layout()

        output_dir = "Plots/ln(Z)_vs_Beta"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/ln(Z)_vs_Beta_{shape}_{solver_name}.png", dpi=150)

        # --- Plot Energy vs beta ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(betas, State_Energy, marker="o", linewidth=1.5, markersize=5, label=label_text)

        # ax.set_yscale("log")
        ax.set_xlabel("Beta")
        ax.set_ylabel("Energy")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        ax.legend(loc="best")
        fig.tight_layout()
        
        output_dir = "Plots/Energy_vs_Beta"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/Energy_vs_Beta_{shape}_{solver_name}.png", dpi=150)

    def runtime_vs_p_benchMark(self, shape: tuple, p_values: list, average: int = 5, beta: float = 1.0, save: bool = False):

        runtime_values = []

        print(" = Start Benchmark =")
        for p in p_values:
            models = []
            for i in range(average):
                models.append(self._Model.generate_lattice(shape, p_ferro=float(p), seed=i)) # We first create various models with the same p so we can average the time it takes to solve them

            measured_runtimes, _, _ = self.run_solver(models, beta)
            average_runtime = np.average(measured_runtimes)
            runtime_values.append(average_runtime)

            print(f"p = {p:.2f} -> {average_runtime} s")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(p_values, runtime_values, marker="o", linewidth=1.5, markersize=5)

        ax.set_xlabel("p (probability of J = +1)")
        ax.set_ylabel("Runtime average (s)")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        fig.tight_layout()
        if(save):
            solver_name = self._model_counter.__class__.__name__
            output_dir = "Plots/Runtime_vs_p"
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/Runtime_vs_p_{shape}_{solver_name}.png", dpi=150)
        plt.show()

        return runtime_values

    def magnetization_vs_beta_benchMark(self, shape: list, betas: list, p_ferro: float = 1.0):
        print("=== Calculating Magnetization ===")

        model = self._Model.generate_lattice((shape[0], shape[1]), p_ferro=p_ferro)
    
        magnetization = []
    
        for beta in betas:
            print("Computing M for beta = ", beta)
            magnetization.append(model.compute_expected_M(self._model_counter, beta))

        # --- Plot ---
        solver_name = self._model_counter.__class__.__name__
        label_text = f"{solver_name} (L={shape})"
        
        fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(betas, magnetization, marker="o", linewidth=1.5, markersize=5, label=label_text)
    
        ax.set_xlabel("Beta")
        ax.set_ylabel(r"$\langle \text{M} \rangle$")
    
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        ax.legend(loc="best")
        fig.tight_layout()

        output_dir = "Plots/Magnetization_vs_Beta"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/Magnetization_vs_Beta_{shape}.png", dpi=150)
    
        return magnetization

    def magnetization_squared_vs_beta_benchMark(self, shape: list, betas: list, p_ferro: float = 1.0):
        print("=== Calculating Magnetization Squared ===")

        model = self._Model.generate_lattice((shape[0], shape[1]), p_ferro=p_ferro)
    
        magnetization_squared = []
    
        for beta in betas:
            print("Computing M^2 for beta = ", beta)    
            magnetization_squared.append(model.compute_expected_M_squared(self._model_counter, beta))

        magnetization = np.sqrt(magnetization_squared)
        
        # --- Plot ---
        solver_name = self._model_counter.__class__.__name__
        label_text = f"{solver_name} (L={shape})"

        fig, ax = plt.subplots(figsize=(8, 5))
    
        ax.plot(betas, magnetization, marker="o", linewidth=1.5, markersize=5, label=label_text)
    
        ax.set_xlabel("Beta")
        ax.set_ylabel(r"$\sqrt{\langle \text{M}^2 \rangle}$")
    
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        ax.legend(loc="best")
        fig.tight_layout()

        output_dir = "Plots/Magnetization_squared_vs_Beta"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/Magnetization_squared_vs_Beta_{shape}.png", dpi=150)
    
        return magnetization

    def runtime_vs_p_benchMark(self, shape: tuple, p_values: list, average: int = 5, beta: float = 1.0, save: bool = False):

        runtime_values = []

        print(" = Start Benchmark =")
        for p in p_values:
            models = []
            for i in range(average):
                models.append(self._Model.generate_lattice(shape, p_ferro=float(p), seed=i)) # We first create various models with the same p so we can average the time it takes to solve them

            measured_runtimes, _, _ = self.run_solver(models, beta)
            average_runtime = np.average(measured_runtimes)
            runtime_values.append(average_runtime)

            print(f"p = {p:.2f} -> {average_runtime} s")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(p_values, runtime_values, marker="o", linewidth=1.5, markersize=5)

        ax.set_xlabel("p (probability of J = +1)")
        ax.set_ylabel("Runtime average (s)")

        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

        fig.tight_layout()
        if(save):
            solver_name = self._model_counter.__class__.__name__
            output_dir = "Plots/Runtime_vs_p"
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f"{output_dir}/Runtime_vs_p_{shape}_{solver_name}.png", dpi=150)
        plt.show()

    def run_solver(self, models: list[BaseModel], beta: float, true_values: list[float] = None) -> tuple[list[float], list[float]]:
        """ Solve the provided models by taking the WCNF and running it through
         the solver. Returns runtimes, and relative errors when true values are
          provided. """
        problems = [model.get_wcnf(beta) for model in models]

        runtimes = []
        rel_errors = []
        Z_values = []

        failed = False
        for i, result in enumerate(self._model_counter.batch_model_count(*problems)):
            if not result.success:
                print(f"FAILURE (model {i} of {len(models)})")
                break

            runtimes.append(float(result.runtime))
            Z_values.append(float(result.model_count))
            if true_values is not None:
                rel_error = abs(result.model_count / true_values[i] - 1)
                rel_errors.append(rel_error)

        return runtimes, rel_errors, Z_values