import numpy as np
import wcnf_matrix as wmc
from time import perf_counter

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

    def get_partition_function(self):
        """
        Compute the partition function of the issing model
        """

        build_start = perf_counter()
        print("  Building WCNF formula...", flush=True)

        # a) First Magnetic Interaction of individual Spins
        exp_magnetic = None

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
        exp_pair_interaction = self._get_identity_tensor()

        for i in range(self.num_spins):
            for j in range(self.num_spins):
                
                # if J = 0 just return I since cosh(0) = 1 and sinh(0) = 0
                if self.j_interactions[i,j] == 0:
                    continue
                else:
                    angle = self.beta * self.j_interactions[i,j]
                    q_plus = self._get_tensor_prod(P0,i,P0,j) + self._get_tensor_prod(P1,i,P1,j)
                    q_minus = self._get_tensor_prod(P0,i,P1,j) + self._get_tensor_prod(P1,i,P0,j)
                    exp_pair_interaction *= (np.exp(angle) * q_plus + np.exp(-angle) * q_minus)

        # c) Z = tr(e^(-beta * H)) -> Calcuate by using WMC
        joint_exp = exp_magnetic * exp_pair_interaction
        cnf, weight_func = joint_exp.trace_formula()

        print(f"  Formula ready after {perf_counter() - build_start:.2f} s; "
              "starting model counter...", flush=True)
        return weight_func(cnf)
