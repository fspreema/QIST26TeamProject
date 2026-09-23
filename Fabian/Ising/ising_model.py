from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import wcnf_matrix as wmc

### Global Variables ###

IDX: wmc.Index[float] = wmc.Index(2)
P0: wmc.WCNFMatrix[float] = (
    wmc.ket(IDX[0]) * wmc.bra(IDX[0])
)
P1: wmc.WCNFMatrix[float] = (
    wmc.ket(IDX[1]) * wmc.bra(IDX[1])
)

I = wmc.WCNFMatrix.identity(IDX, 1)
X = wmc.ket(IDX[0]) * wmc.bra(IDX[1]) + wmc.ket(IDX[1]) * wmc.bra(IDX[0])

# It is worth noting here that while scalar multiplication works, devision doesn't
P_plus: wmc.WCNFMatrix[float] = (I + X) * 0.5
P_minus: wmc.WCNFMatrix[float] = (I - X) * 0.5

### GEOMETRY ###

def get_pairs(length: int, dim : int, periodic: bool) -> list[tuple[int, int]]:
    """
    Function to get for any given one dimnesional or 2 dimnesional lattice layout

    a) 1-dim:
        We use the layout:
            o - o - o

        - Check right neigbour to skip double counting 

    b) 2-dim:
        -Non-neighbor couplings and self-couplings are zero.

        We use the layout:
            o - o - o 
            |   |   |
            o - o - o
            |   |   |
            o - o - o

        - Check right/ lower neighbour to skip double counting
    """

    # Init params.
    num_spins = length ** dim
    pairs = []

    # a) 1 dimensional case
    if dim == 1:

        # Get all pairs which is current spin plus next spin
        pairs = [(i, i + 1) for i in range(length - 1)]

        # If peridoic and we have length greater then 2 add the last and first together
        # -> For n<2 this would result in double pairs and therefore double counting
        if periodic and length > 2:
            pairs.append((0, length - 1))

    # b) 2 dimensional case
    elif dim == 2:

        # Check condition
        if length == 1:
            raise ValueError("Length of lattice must be greater than 1 for 2D lattice")
        
        if periodic:
            for row in range(length):
                for col in range(length):
                    i = row * length + col

                    # Check normal row condiction and row perdioic boundary condiction
                    if row + 1 <= length:
                        j = (i + length) % num_spins
                        pairs.append((i,j))

                    # Check boundary matrix entries and pair them to first in same Row
                    if col + 1 == length:
                        j = i - length + 1 
                        pairs.append((i,j))

                    # Check normal pair condiction
                    elif col + 1 < length:
                        j = i + 1
                        pairs.append((i,j))
        else:

            for row in range(length):
                for col in range(length):
                    i = row * length + col

                    # Check lower neighbour
                    if row + 1 < length:
                        j = i + length
                        pairs.append((i,j))

                    # Check right neighbour
                    if col + 1 < length:
                        j = i + 1
                        pairs.append((i,j))

    else:
        raise ValueError("Only 1D and 2D lattices are supported")

    return pairs

### PARAMETER GENERATION ###

def get_lattice_parameters(length: int, 
                           periodic: bool, 
                           dim: int,
                           seed: int = 42
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    a) 1-Dim:
        -Returns J and h coupling matrices for the 2 dim ising model with
        each nearest-neighbor bond gets an independently sampled coupling J_ij \sim N(0,1).
        -Each spin gets an independently sampled field h_i \sim N(0,1).
        -Non-neighbor couplings and self-couplings are zero.

    b) 2-Dim:
        -Returns J and h coupling matrices for the 2 dim ising model with
        each nearest-neighbor bond gets an independently sampled coupling J_ij \sim N(0,1).
        -Each spin gets an independently sampled field h_i \sim N(0,1).
        -Non-neighbor couplings and self-couplings are zero.
    """

    # a) Get pariings for the selected lattice
    pairs = get_pairs(length= length, dim= dim, periodic= periodic)

    # b) Init h and J coupling terms
    num_spins = length ** dim
    rng = np.random.default_rng(seed=seed)
    h_field = rng.normal(size=num_spins)
    j_interactions = np.zeros((num_spins , num_spins))

    # c) Populate h and J coupling terms
    for i,j in pairs:
        j_interactions[i, j] = rng.normal(loc=0.0, scale=1.0)

    return h_field, j_interactions

def get_random_bond_ising_parameters(length: int, 
                           periodic: bool, 
                           dim: int,
                           p_ferro: float, 
                           seed: int = 42
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    - We set h=0
    - We sample J from a bimodal distrubtation given by:
        J_ij = p * delta(J_ij - 1) + (1-p) * delta(J_ij + 1)
    """

    # Init J and rng
    rng = np.random.default_rng(seed)
    num_spins = length ** dim
    j_interactions = np.zeros((num_spins, num_spins))

    # Get pairings for the selectzed lattice
    pairs = get_pairs(length= length, dim= dim, periodic= periodic)

    # J is sampled from a bimodal distribution
    for i,j in pairs:
        if rng.random() < p_ferro:
            j_interactions[i, j] = 1.0
        else:
            j_interactions[i, j] = -1.0

    return j_interactions

def get_random_graph_parameters(length: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    - We set h=0
    - We sample J from a uniformal distribution
    - On average each spin has 3 neighbours:
        -> For every possible pair of spins (i,j), independently create an interaction edge with probability p
        -> p = 3 / (N - 1) since 3 on average neigbours of N-1 total neigbours
    """
    # prob given by
    p = 3 / (length - 1)
    rng = np.random.default_rng(seed)

    # Init h and J
    h_field = np.zeros(length)
    j_interactions = np.zeros((length, length))

    # # J is sampled uniformly from [-1, 1]
    # if I, J skip J, I and i = j -> Upper/lower triangle
    for i in range(length):
        for j in range(length):
            if rng.random() < p and i<j:
                j_interactions[i, j] = rng.uniform(low=-1.0, high=1.0)

    return h_field, j_interactions

class PartitionModelWMC(ABC):

    @abstractmethod
    def build_partition_formula(self):
        pass

    def _get_identity_tensor(self) -> wmc.WCNFMatrix[float]:
        """
        Compute the I tensor for all spins
        """

        return wmc.WCNFMatrix.identity(IDX, self.num_spins)

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

class TransversalIsingModel(PartitionModelWMC):

    def __init__(self, 
                 num_spins: int, 
                 j_constant: float, 
                 g_constant: float, 
                 beta: float,
                 trotter_steps: int, 
                 pairs: list[tuple[int, int]]
                ) -> None:

        # Initilitze Ising model parameters
        self.num_spins = num_spins
        self.j_constant = j_constant
        self.g_constant = g_constant
        self.beta = beta
        self.trotter_steps = trotter_steps
        self.pairs = pairs

        # Check conditions
        if trotter_steps <= 0 or num_spins <= 0:
            raise ValueError("Trotter steps must be positive and num_spins must be greater than 1")

    def build_partition_formula(self):
        """
        Build the weighted CNF whose model count is the partition function.
        """

        # Initlize Dirac registry
        spins = [wmc.Reg(IDX) for _ in range(self.num_spins)]
        build_start = perf_counter()
        print("  Building WCNF formula...", flush=True)

        # Init constant angles
        angle_single = (self.beta * self.j_constant * self.g_constant) / self.trotter_steps
        angle_pairs = (self.beta * self.j_constant) / self.trotter_steps

        # a) First Single Interaction
        exp_magnetic = None

        # If all g = 0 we get the identitiy
        if self.g_constant == 0:
            exp_magnetic = self._get_identity_tensor()

        # If not actually construct the H_X contribution
        else:
            for _ in range(self.num_spins):

                exp_magnetic_i = np.exp(angle_single) * P_plus + np.exp(-angle_single) * P_minus

                # First tensor factor
                if exp_magnetic is None:
                    exp_magnetic = exp_magnetic_i

                # Append further tensor factors
                else:
                    exp_magnetic = exp_magnetic ** exp_magnetic_i

        # b) Pair Interaction
        # Full-system identity: its positions correspond to every spin in order.
        exp_pair_interaction = self._get_identity_tensor() | tuple(spins)

        # Def needed constants
        q_plus = P0 ** P0 + P1 ** P1
        q_minus = P0 ** P1 + P1 ** P0

        if self.j_constant != 0:
            for i, j in self.pairs:

                # Build local interaction
                local_interactions = (np.exp(angle_pairs) * q_plus + np.exp(-angle_pairs) * q_minus)

                # Now construct corresponding tensored interactions
                # Its first position in tensor prodcut belongs to spin i; its second belongs to spin j.
                exp_pair_interaction *= local_interactions | (spins[i], spins[j])

        # c) Z = tr(e^(-beta * H)) -> Calcuate by using WMC
        # Since we have registries in joint_exp we first need to call the underlying mtx
        joint_exp = (exp_magnetic | tuple(spins)) * exp_pair_interaction

        # Implement trotter-step
        joint_exp_trotter = joint_exp
        for _ in range(1,self.trotter_steps):
            joint_exp_trotter = joint_exp_trotter * joint_exp

        cnf, weight_func = joint_exp_trotter.mat.trace_formula()

        print(f"  Formula ready after {perf_counter() - build_start:.2f} s; "
              "ready for model counters", flush=True)

        return cnf, weight_func

class IsingModel(PartitionModelWMC):

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
