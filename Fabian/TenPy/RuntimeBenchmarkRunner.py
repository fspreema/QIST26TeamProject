import csv
import statistics
import timeit

import numpy as np
from PurificationSimulation import PurificationSimulation
from RandomBondIsing import RandomBondIsing


class RuntimeBenchmarkRunner:

    def __init__(
        self,
        model_configs: list[dict],
        chi_limit: int = 100,
        seeds: list[int] | None = None,
        beta_max=0.5,
        dt=0.05,
        repeats=5,
    ) -> None:

        # Init Parameters
        if seeds is None:
            seeds = [i for i in range(repeats)]
        self.seeds = seeds
        self.chi_limit = chi_limit
        self.seeds = seeds
        self.model_configs = model_configs
        self.beta_max = beta_max
        self.dt = dt
        self.repeats = repeats

    def _get_purification_observable(self, curr_model_config: dict) -> tuple:

        # Init Model and Perform Benchmark
        curr_model = RandomBondIsing(curr_model_config)

        # Run Benchmark on current p
        print(f"Performing Benchmark for {curr_model_config.get("lattice")}")
        benchmark = PurificationSimulation(
            model=curr_model,
            measure_obs="None",
            chi_limit=self.chi_limit,
            track_z= True,
            beta_max= self.beta_max,
            dt = self.dt,
        )
        data = benchmark.run()

        # Get Betas and Observables:
        betas = data.get("beta")
        logZ = data.get("logZ")
        max_chi = data.get("max_chi")

        return (betas, logZ, max_chi)

    def run(self, file_name: str, overwrite: bool = False) -> None:

        if overwrite:
            mode = "w"
        else:
            mode = "a"

        with open(file= file_name, mode = mode, newline="") as file:

            writer = csv.writer(file)

            # Only write header when the CSV rwos are empty aka new csv file
            if file.tell() == 0:
                writer.writerow([
                    "lattice", "Lx", "Ly", "p",
                    "bc_x", "bc_y", "bc_MPS",
                    "runtime_s", "betas", "max_chi", "log_z", "num_run"
                ])

            for curr_params in self.model_configs:

                times = []

                for curr_repeat in range(self.repeats):

                    # Write seed into the model params for reproducibility
                    curr_params["rng"] = np.random.default_rng(self.seeds[curr_repeat])

                    start = timeit.default_timer()

                    betas, log_z, max_chi = self._get_purification_observable(curr_params)

                    runtime = timeit.default_timer() - start
                    times.append(runtime)

                    writer.writerow([
                        curr_params["lattice"],
                        curr_params["Lx"],
                        curr_params["Ly"],
                        curr_params["p"],
                        curr_params["bc_x"],
                        curr_params["bc_y"],
                        curr_params["bc_MPS"],
                        runtime,
                        betas,
                        max_chi,
                        log_z,
                        curr_repeat
                    ])

                    # FLush directly to make output visible directly for debugging
                    # -> Could be removed later but idk how much efficiency this changes
                    file.flush()

                print(
                    f"→ Median runtime: "
                    f"{statistics.median(times):.1f}s\n"
                )
