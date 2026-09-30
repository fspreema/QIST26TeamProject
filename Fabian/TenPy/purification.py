"""Adapted from the TeNPy purification example."""
# Copyright (C) TeNPy Developers, Apache license

from tenpy.algorithms.purification import PurificationApplyMPO
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.purification_mps import PurificationMPS


class BenchmarkModel:

    """
    ** We perform the following procedure: **

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
                 beta_max=3.0,
                 dt=0.05,
                 order=2,
                 approx="II") -> None:

        # Check measure Observable
        if measure_obs not in {"Sz", "Energy_per_spin", "Correlations", "Mz_squared", "all"}:
            raise ValueError("Please Select Valid Observable")

        # Define Model Params/ Init Model & Set current Beta
        self.model = model
        self.beta_max = beta_max
        self.dt = dt
        self.order = order
        self.approx = approx
        self.observable = measure_obs

        # Perform imag evolution setup and reset already computed states
        self._reset()

    def _measure(self):
        if self.observable == "all":
            correlations = self.psi.correlation_function("Sz", "Sz").real

            return {
                "Sz": self.psi.expectation_value("Sz").real,
                "Energy_per_spin": (
                    self.model.H_MPO.expectation_value(self.psi).real
                    / self.psi.L
                ),
                "Correlations": correlations,
                "Mz_squared": correlations.sum() / self.psi.L**2,
            }

        elif self.observable == "Sz":
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
            return correlations.sum().real / self.psi.L**2

        raise ValueError(f"Unknown observable: {self.observable}")

    def _reset(self) -> None:
        """
        Reset the state, temperature, and measurements.
        """

        self.beta = 0.0
        self.betas = [0.0]
        self._imag_setup_mpo()

    def _imag_setup_mpo(self) -> None:

        self.psi = PurificationMPS.from_infiniteT(
                        self.model.lat.mps_sites(),
                        bc=self.model.lat.bc_MPS,
                        unit_cell_width=self.model.lat.mps_unit_cell_width,
                    )

        options = {
            "trunc_params": {
                "chi_max": 100,
                "svd_min": 1.0e-8,
            }
        }

        if self.order == 1:
            self.Us = [self.model.H_MPO.make_U(-self.dt, self.approx)]
        elif self.order == 2:
            self.Us = [
                self.model.H_MPO.make_U(-d * self.dt, self.approx)
                for d in [0.5 + 0.5j, 0.5 - 0.5j]
            ]
        else:
            raise ValueError("order must be 1 or 2")

        self.eng = PurificationApplyMPO(self.psi, self.Us[0], options)

        self.values = [self._measure()]


    def run(self) -> dict:
        """
        Perform evolution run
        """

        while self.beta < self.beta_max:
            for U in self.Us:
                self.eng.init_env(U)
                self.eng.run()

            self.beta += 2.0 * self.dt
            self.betas.append(self.beta)

            # Record the selected observable at this beta.
            self.values.append(self._measure())

        return {"beta": self.betas, self.observable: self.values}