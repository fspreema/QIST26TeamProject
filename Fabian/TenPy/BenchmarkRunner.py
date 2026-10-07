import csv
import statistics
import timeit

from BenchmarkModel import BenchmarkModel
from RandomBondIsing import RandomBondIsing


class BenchmarkRunner:

    def __init__(
        self,
        model_configs: list[dict],
        seeds: list[int] = 10,
        beta_max=0.4,
        dt=0.05,
        repeats=1,
    ) -> None:

        # Init Parameters
        self.seeds = seeds # Seed currently unused
        self.model_configs = model_configs
        self.beta_max = beta_max
        self.dt = dt
        self.repeats = repeats

    def _get_purification_observable(self, curr_model_config: dict) -> tuple:

        # Init Model and Perform Benchmark
        curr_model = RandomBondIsing(curr_model_config)

        # Run Benchmark on current p
        print(f"Performing Benchmark for {curr_model_config.get("lattice")}, x-Length {curr_model_config.get("Lx")}, y-length {curr_model_config.get("Ly")}")
        benchmark = BenchmarkModel(
            model=curr_model,
            measure_obs="None",
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
                    "runtime_s", "betas", "max_chi", "log_z"
                ])

            for curr_params in self.model_configs:

                times = []

                for _ in range(self.repeats):

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
                        log_z
                    ])

                    # FLush directly to make output visible directly for debugging
                    # -> Could be removed later but idk how much efficiency this changes
                    file.flush()

                print(
                    f"→ Median runtime: "
                    f"{statistics.median(times):.1f}s\n"
                )
