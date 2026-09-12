from time import perf_counter

import numpy as np
import wcnf_matrix as wmc

# Global Variables
IDX: wmc.Index[float] = wmc.Index(2)

P0: wmc.WCNFMatrix[float] = (
    wmc.ket(IDX[0]) * wmc.bra(IDX[0])
)
P1: wmc.WCNFMatrix[float] = (
    wmc.ket(IDX[1]) * wmc.bra(IDX[1])
)
I = wmc.WCNFMatrix.identity(IDX, 1)

class IsingParameters:
    def __init__(self, 
                 length: int,
                 periodic: bool
                 ) -> None:

        self.length = length
        self.periodic = periodic

    def get_1dim_parameters(self, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        -Returns J and h coupling matrices for the 2 dim ising model with
        each nearest-neighbor bond gets an independently sampled coupling J_ij \sim N(0,1).
        -Each spin gets an independently sampled field h_i \sim N(0,1).
        -Non-neighbor couplings and self-couplings are zero.

        We use the layout:

        0 - 1 - 2 - 3 - ...

        """
        self.h_field = np.zeros(self.length)
        self.j_interactions = np.zeros((self.length, self.length))
    
        rng = np.random.default_rng(seed=seed)
        for i in range(self.length):

            # Get magnetic field for each spin
            self.h_field[i] = rng.normal(loc=0.0, scale=1.0)

            for j in range(self.length):
                # Interaction energy constructed out of to the bond connecting the two spins
                # -> Looking at that bond from its other endpoint doesn’t introduce a second interaction.
                # Skip double interaction for the same spin
                if i >= j:
                    continue
                # Only give interaction to nearest neighbors
                if abs(i - j) == 1:
                    self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)
                # If perdioic just take first and last and set them as neighbours
                if self.periodic and abs(i - j) == self.length - 1:
                    self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)
                
        return self.h_field, self.j_interactions

    def get_2dim_parameters(self, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        -Returns J and h coupling matrices for the 2 dim ising model with
        each nearest-neighbor bond gets an independently sampled coupling J_ij \sim N(0,1).
        -Each spin gets an independently sampled field h_i \sim N(0,1).
        -Non-neighbor couplings and self-couplings are zero.

        We use the layout:
            o - o - o 
            |   |   |
            o - o - o
            |   |   |
            o - o - o

        - Interaction energy constructed out of to the bond connecting the two spins
            -> Looking at that bond from its other endpoint doesn’t introduce a second interaction.
        - Check right/ lower neighbour to skip double counting
        """
        self.num_qubits = self.length ** 2
        self.h_field = np.zeros(self.num_qubits)
        self.j_interactions = np.zeros((self.num_qubits, self.num_qubits))

        rng = np.random.default_rng(seed=seed)
        for i in range(self.num_qubits):
            # Get magnetic field for each spin
            self.h_field[i] = rng.normal(loc=0.0, scale=1.0)

        if self.periodic:
            for row in range(self.length):
                for col in range(self.length):
                    i = row * self.length + col

                    # Check normal row condiction and row perdioic boundary condiction
                    if row + 1 <= self.length:
                        j = (i + self.length) % self.num_qubits
                        self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)

                    # Check boundary matrix entries and pair them to first in same Row
                    if col + 1 == self.length:
                        j = i - self.length + 1 
                        self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)

                    # Check normal pair condiction
                    elif col + 1 < self.length:
                        j = i + 1
                        self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)
        else:
            for row in range(self.length):
                for col in range(self.length):
                    i = row * self.length + col

                    # Check lower neighbour
                    if row + 1 < self.length:
                        j = i + self.length
                        self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)

                    # Check right neighbour
                    if col + 1 < self.length:
                        j = i + 1
                        self.j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)
                
        return self.h_field, self.j_interactions

    def random_sampling(self, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        - We set h=0
        - We sample J from a uniformal distribution
        - On average each spin has 3 neighbours:
            -> For every possible pair of spins (i,j), independently create an interaction edge with probability p
            -> p = 3 / (N - 1) since 3 on average neigbours of N-1 total neigbours
        """
        # prob given by
        p = 3 / (self.length - 1)
        rng = np.random.default_rng(seed)

        # Init h and J
        self.h_field = np.zeros(self.length)
        self.j_interactions = np.zeros((self.length, self.length))

        # # J is sampled uniformly from [-1, 1]
        # if I, J skip J, I and i = j -> Upper/lower triangle
        for i in range(self.length):
            for j in range(self.length):
                if rng.random() < p and i<j:
                    self.j_interactions[i, j] = rng.uniform(low=-1.0, high=1.0)

        return self.h_field, self.j_interactions

class IsingModelWMC:

    def __init__(self, 
                 num_spins: int, 
                 j_interactions: np.ndarray, 
                 h_field: np.ndarray, 
                 beta: float, 
                 ) -> None:

        # Initilitze Ising model parameters
        self.num_spins = num_spins
        self.j_interactions = j_interactions
        self.h_field = h_field
        self.beta = beta

        # Check Conditions for interaction matrix
        if self.h_field.shape != (num_spins,) or self.j_interactions.shape != (num_spins, num_spins):
            raise ValueError(
                "J and h must have valid dimensions"
            )

    def _get_tensor_prod(self, A: wmc.WCNFMatrix[float], 
                         i: int, 
                         B: wmc.WCNFMatrix[float] | None = None, 
                         j: int | None= None) -> wmc.WCNFMatrix[float]:
        """
        CURRENTLY LEGACY FUCTION!!!

        Compute the tensor product of two cases
        a) One Matrix given with position i
            -> Calc I tens I ... tens A ... where A is in the i-th pos
        b) Two Matrices given with position i,j respectively
            -> -> Calc B tens I ... tens A ... where A is in the i-th pos and B in the j-th
        """

        if (B is None and j is not None) or (B is not None and j is None):
            raise ValueError(
                "When B is None the corresponding index should also be None and vice verca"
                )

        if i == 0:
            curr_tens_prod = A
        # If both are given check condition
        elif B is not None and j is not None and j == 0:
            curr_tens_prod = B
        else:
            curr_tens_prod = I

        for curr_idx in range(1, self.num_spins):
            if curr_idx == i:
                curr_tens_prod = curr_tens_prod ** A
            # If both are given check condition
            elif B is not None and j is not None and curr_idx == j:
                curr_tens_prod = curr_tens_prod ** B
            else:
                curr_tens_prod = curr_tens_prod ** I

        return curr_tens_prod

    def _get_identity_tensor(self) -> wmc.WCNFMatrix[float]:
        """
        Compute the I tensor for all spins
        """

        return wmc.WCNFMatrix.identity(IDX, self.num_spins)

    def build_partition_formula(self):
        """
        Build the weighted CNF whose model count is the partition function.
        """

        # Initlize Dirac registry
        spins = [wmc.Reg(IDX) for _ in range(self.num_spins)]

        build_start = perf_counter()
        print("  Building WCNF formula...", flush=True)

        # a) First Magnetic Interaction of individual Spins
        exp_magnetic = None

        # If all h = 0 we get the identitiy
        if np.all(self.h_field == 0):
            exp_magnetic = self._get_identity_tensor()

        # If not actually construct the individual contributions
        else:
            for i in range(self.num_spins):

                # if h = 0 we get the identitiy
                if self.h_field[i] == 0:
                    exp_magnetic_i = I

                # Else calculate value of 2-dim matrices
                else:
                    angle = self.beta * self.h_field[i]
                    exp_magnetic_i = np.exp(angle) * P0 + np.exp(-angle) * P1

                # First tensor factor
                if exp_magnetic is None:
                    exp_magnetic = exp_magnetic_i

                # Append further tensor factors
                else:
                    exp_magnetic = exp_magnetic ** exp_magnetic_i

        # b) Second Magnetic Interaction of pairs of spins
        # Full-system identity: its positions correspond to every spin in order.
        exp_pair_interaction = self._get_identity_tensor() | tuple(spins)

        # Def needed constants
        q_plus = P0 ** P0 + P1 ** P1
        q_minus = P0 ** P1 + P1 ** P0

        for i in range(self.num_spins):
            for j in range(self.num_spins):
                
                # if J = 0 just continue since exp(0) = I
                if self.j_interactions[i,j] == 0:
                    continue
                else:
                    # Build local interaction
                    angle = self.beta * self.j_interactions[i,j]
                    local_interactions = (np.exp(angle) * q_plus + np.exp(-angle) * q_minus)

                    # Now construct corresponding tensored interactions
                    # Its first position in tensor prodcut belongs to spin i; its second belongs to spin j.
                    exp_pair_interaction *= local_interactions | (spins[i], spins[j])

        # c) Z = tr(e^(-beta * H)) -> Calcuate by using WMC
        # Since we have registries in joint_exp we first need to call the underlying mtx
        joint_exp = (exp_magnetic | tuple(spins)) * exp_pair_interaction
        cnf, weight_func = joint_exp.mat.trace_formula()

        print(f"  Formula ready after {perf_counter() - build_start:.2f} s; "
              "ready for model counters", flush=True)

        return cnf, weight_func

    def get_partition_function(self, solver_class, *, formula=None):
        """
        Return Z and the solver's own runtime, excluding formula construction.
        Pass this model's prebuilt formula to reuse it across model counters.
        """

        cnf, weight_func = (
            self.build_partition_formula() if formula is None else formula
        )

        result = solver_class().model_count(cnf, weight_func)
        if not result.success:
            raise RuntimeError("Solver failed to execute")
        if not np.isfinite(result.model_count) or result.model_count <= 0:
            raise RuntimeError(
                f"Solver returned an invalid partition function Z={result.model_count}; "
                "possible numerical overflow or underflow"
            )
        if not np.isfinite(result.runtime) or result.runtime < 0:
            raise RuntimeError("Solver did not report a valid runtime")
        return result
