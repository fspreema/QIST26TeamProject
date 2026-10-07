import csv

import numpy as np
from PurificationSimulation import PurificationSimulation
from RandomBondIsing import RandomBondIsing
from scipy.special import logsumexp
from tenpy.models.model import CouplingMPOModel


class AccuracyBenchmarkRunner:

    """
    This class is used to benchmark accuracy of purification procedure in comparison to the exact resullt of log(Z)

    For each Model we perform the following:
        1) Calculate given the RandomBondIsing model the exact energy terms by extracting the current bond terms
        2) Calculate the exact log(Z) by simple brute force
        3) For each chi in the given chi_range run purification procedure and calcualte log(Z)
    """

    def __init__(self,
            model_configs: list[dict],
            chi_range: list[int],
            seeds: list[int] | None = None,
            beta_max=0.5,
            dt=0.05,
            repeats=5,
        ) -> None:

        # Init Parameters
        if seeds is None:
            seeds = [i for i in range(repeats)]
        self.seeds = seeds
        self.model_configs = model_configs
        self.chi_range = chi_range
        self.beta_max = beta_max
        self.dt = dt
        self.repeats = repeats

    def _run_purification(self, current_model: CouplingMPOModel, 
                          curr_model_config: dict, 
                          curr_chi_limit: int
                          ) -> tuple:

        # Run Benchmark on current p
        print(f"Performing Benchmark for {curr_model_config.get("lattice")}")
        benchmark = PurificationSimulation(
            model=current_model,
            measure_obs="None",
            chi_limit= curr_chi_limit,
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

    def _get_exact_logz(self, current_model: CouplingMPOModel) -> float:

        """
        Helper Function for the calculation of the exact log(Z)

        -> Get exact Coupling terms -> calc energy -> get log(Z)
        """

        # Get Num Spins
        n_spins = current_model.lat.N_sites

        if n_spins > 25:
            raise ValueError(
                "This will blow up your computer memory, are you crazy? Please select Lx * Ly <= 25"
            )

        # Get exact Coupling Terms:
        coupling_terms = current_model.all_coupling_terms().to_TermList()

        # Each integer encodes one spin configuration in its binary digits.
        configurations = np.arange(2**n_spins)

        # One energy per configuration.
        energies = np.zeros(configurations.size)

        # Calculate Exact Energy
        for term, coefficient in zip(
            coupling_terms.terms,
            coupling_terms.strength,
        ):
            (_, i), (_, j) = term

            # Convert bit values 0, 1 into Sigmaz eigenvalues -1, +1.
            spin_i = (
                2 * ((configurations >> int(i)) & 1).astype(np.int8) - 1
            )
            spin_j = (
                2 * ((configurations >> int(j)) & 1).astype(np.int8) - 1
            )

            energies += float(coefficient) * spin_i * spin_j

        return float(logsumexp(-self.beta_max * energies))

    def run(self, file_name:str, overwrite:bool = False) -> None:
        
        if overwrite:
            mode = "w"
        else:
            mode = "a"

        with open(file= file_name, mode = mode, newline="") as file:

            writer = csv.writer(file)

            # Only write header when the CSV rwos are empty aka new csv file
            if file.tell() == 0:
                writer.writerow([
                    "lattice", "Lx", "Ly", 
                    "p", "bc_x", "bc_y", 
                    "bc_MPS", "betas", "max_chi", 
                    "log_z_approx", "log_z_exact", "num_run", 
                    "chi_limit", "seed", "dt"
                ])

            for curr_params in self.model_configs:

                for curr_repeat in range(self.repeats):

                    # Write seed into the model params for reproducibility
                    curr_params["rng"] = np.random.default_rng(self.seeds[curr_repeat])

                    # Construct Model with Current Model config & get exact log(Z)
                    curr_model = RandomBondIsing(curr_params)
                    log_z_exact = self._get_exact_logz(curr_model)

                    for curr_chi in self.chi_range:

                        # Run purification procedure and get approx log(Z)
                        betas, log_z_approx, max_chi = self._run_purification(curr_model, curr_params, curr_chi)

                        writer.writerow([
                            curr_params["lattice"],
                            curr_params["Lx"],
                            curr_params["Ly"],
                            curr_params["p"],
                            curr_params["bc_x"],
                            curr_params["bc_y"],
                            curr_params["bc_MPS"],
                            betas,
                            max_chi,
                            log_z_approx,
                            log_z_exact,
                            curr_repeat,
                            curr_chi,
                            self.seeds[curr_repeat],
                            self.dt
                        ])