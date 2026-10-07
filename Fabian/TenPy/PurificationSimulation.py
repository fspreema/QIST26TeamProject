import math

import numpy as np
from tenpy.algorithms.purification import PurificationApplyMPO
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.purification_mps import PurificationMPS


class PurificationSimulation:

    """
    ### Purification Procedure ###

    We perform the following procedure:

    1) Start with compl. disordered physical state (Here each spin uncorrelated  and either in up or down)
    2) Represent non-pure density mtx by purification
        -> Combine with ancilla to entangled state which itself is pure state (Our state is then psi = tr_B(psi_with_ancilla))
    3) Apply imaginary-time evolution to a certain tau which directly correlates to a certain temperatur
        -> By supstituting t with i*tau we get exp(-tau * H) -> Non unitary operation -> Probabilities are not preserved!
        -> Basically Annelaing until a certain time T
    4) Calculate physical observables from the resulting purified state
    """

    def __init__(self, model: CouplingMPOModel,
                 measure_obs: str,
                 chi_limit:int = 100,
                 track_z: bool = False,
                 beta_max=3.0,
                 dt=0.05,
                 order=2,
                 approx="II") -> None:

        # Check measure Observable
        if measure_obs not in {"Sz", "Energy_per_spin", "Correlations", "Mz_squared", "None"}:
            raise ValueError("Please Select Valid Observable")

        # Define Model Params/ Init Model & Set current Beta
        self.model = model
        self.chi_limit = chi_limit
        self.beta_max = beta_max
        self.dt = dt
        self.order = order
        self.approx = approx
        self.observable = measure_obs
        self.track_z = track_z

        # Perform imag evolution setup and reset already computed states
        self._reset()

    def _measure(self):
        """
        Helper function including all supported observables

        -> Tbh this is currently really not needed...
        """

        if self.observable == "Sz":
            return self.psi.expectation_value("Sz")

        elif self.observable == "Energy_per_spin":
            return (
                self.model.H_MPO.expectation_value(self.psi).real
                / self.psi.L
            )

        elif self.observable == "Correlations":
            return self.psi.correlation_function("Sz", "Sz")

        elif self.observable == "Mz_squared":
            correlations = self.psi.correlation_function("Sz", "Sz")
            return np.sqrt(correlations.sum() / self.psi.L**2)

        raise ValueError(f"Unknown observable: {self.observable}")

    def _reset(self) -> None:
        """
        Reset the state, temperature, and measurements.
        """

        self.beta = 0.0
        self.betas = [0.0]
        self.logZ = 0
        self.logZs = []
        self.values = None
        self.max_chis = []
        self._imag_setup_mpo()

    def _imag_setup_mpo(self) -> None:
        """
        Setup purification engine
        """

        # Construct pure state representation by doing the following on every site
        # 1/sqrt(2)*(|Up\rangle |A_up\rangle + |Down\rangle |A_Down\rangle)
        self.psi = PurificationMPS.from_infiniteT(
                        self.model.lat.mps_sites(),
                        bc=self.model.lat.bc_MPS,
                        unit_cell_width=self.model.lat.mps_unit_cell_width,
                    )

        self.logZ = sum(math.log(site.dim) for site in self.psi.sites)
        self.logZs = [self.logZ]
        self.psi.norm = 1.0

        options = {
            "trunc_params": {
                "chi_max": self.chi_limit,
                "svd_min": 1.0e-8,
            }
        }

        # Depending on Order we initilize different time evolution operators in general the scheme is
        # It is worth noting that .make_U approximates this state we therefore have
        # U \approx e^{-\Delta \tau H}
        if self.order == 1:
            self.Us = [self.model.H_MPO.make_U(-self.dt, self.approx)]
        elif self.order == 2:
            self.Us = [
                self.model.H_MPO.make_U(-d * self.dt, self.approx)
                for d in [0.5 + 0.5j, 0.5 - 0.5j]
            ]
        else:
            raise ValueError("order must be 1 or 2")

        # Simply initilize engine and measure the first batch for infinte temperature
        self.eng = PurificationApplyMPO(self.psi, self.Us[0], options)
        if self.observable != "None":
            self.values = [self._measure()]


    def run(self) -> dict:
        """
        Perform evolution run
        """

        while self.beta < self.beta_max - 1e-7:

            # Develop state with imaginary evolution
            # |\Psi_{new}\rangle \propto (e^{-\Delta\tau H_P}\otimes I_A) |\Psi_{old}\rangle
            for U in self.Us:
                self.eng.init_env(U)
                self.eng.run()

                if self.track_z:
                    # Amplitude normalization from this MPO application.
                    r = abs(self.psi.norm)

                    # Z depends on the SQUARED wavefunction norm.
                    self.logZ += 2.0 * math.log(r)

                    # Reset before the next init_env(), which copies the state.
                    self.psi.norm = 1.0

            # Update betas accordingly and observable accordingly
            # Print max chi
            self.beta += 2.0 * self.dt
            self.betas.append(self.beta)
            print(
                f"beta={self.beta:.2f} | "
                f"max chi={max(self.psi.chi)}"
                )
            self.max_chis.append(max(self.psi.chi))

            if self.observable != "None":
                self.values.append(self._measure())
            if self.track_z:
                self.logZs.append(self.logZ)

        if self.track_z:
            return {"beta": self.betas, 
                    self.observable: self.values, 
                    "logZ": self.logZs,
                    "max_chi" : self.max_chis
                    }
        else:
            return {"beta": self.betas, self.observable: self.values}