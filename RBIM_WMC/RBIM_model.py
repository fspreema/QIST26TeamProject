from wcnf_matrix import *
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce
from scipy.linalg import expm
from typing import Iterable, Literal, Any

SQRT2 = math.sqrt(2)
X_MATRIX = np.matrix([[0, 1], [1, 0]])
Z_MATRIX = np.matrix([[1, 0], [0, -1]])

class RandomBondIsingModel:
    def __init__(self, spin_count: int, interaction: dict[tuple[int, int], float] | None = None):
            """ Constructor """
            self._spin_count = spin_count
            # Interaction strengths
            self._interaction = {} if interaction is None else interaction.copy()

    def __setitem__(self, key: tuple[int, int], value: float):
        """ Set the interaction strength between two spins spin(s). Overrides
            any existing strength """
        self.set_interaction(*key, value)

    def __len__(self) -> int:
        """ Returns the number of spins """
        return self._spin_count

    def set_interaction(self, i: int, j: int, strength: float, *,
    add_to_existing: bool = False):
        """ Set the interaction strength between two spins. If add_to_existing
            is set to true, the strength will be summed with any existing
            strength """
        if i > j:
            i, j = j, i
        existing = 0.0
        if add_to_existing:
            existing = self._interaction.get((i, j), 0.0)
        self._interaction[i, j] =  + strength

    def exp_zz_rotation(self, theta: float, index: Index) -> WCNFMatrix[float]:
        """ Returns a ZZ-rotation matrix with angle theta """
        x, y, r = BoolVar(), BoolVar(), BoolVar()
        cnf = CNF([
            [x, -y, -r],
            [x, y, r],
            [-x, y, -r],
            [-x, -y, r],
        ])
        weight_func = WeightFunction([x, y, r])
        weight_func.fill(1.0)
        weight_func[r, 0] = np.exp(-theta)
        weight_func[r, 1] = np.exp(theta)
        x, y = LogVarRep(2, [x]), LogVarRep(2, [y])
        return WCNFMatrix(index, cnf, weight_func, [x, y], [x, y])

    @staticmethod
    def generate_lattice(shape: tuple[int, int], p_ferro= 0.5, seed: int= 42) -> RandomBondIsingModel:
        """ Generate a rectangular lattice Ising model with interaction strengths.
        The side lengths is given by shape """
        num_spins = shape[0] * shape[1]
        model = RandomBondIsingModel(num_spins)
        rng = np.random.default_rng(seed)
        # Interaction strengths
        for i in range(num_spins):
            if i + shape[0] < num_spins:
                strength = RandomBondIsingModel.get_strength_value(p_ferro= p_ferro, rng= rng)
                model[i, i + shape[0]] = strength
            if i % shape[0] != shape[0] - 1:
                strength = RandomBondIsingModel.get_strength_value(p_ferro= p_ferro, rng= rng)
                model[i, i + 1] = strength
        return model

    @staticmethod
    def generate_square_lattice(size: int, p_ferro= 0.5, seed: int= 42) -> RandomBondIsingModel:
        """ Generate a square lattice Ising model with interaction strengths.
        The side length is given by size """
        return RandomBondIsingModel.generate_lattice((size, size), p_ferro, seed)

    @staticmethod
    def get_strength_value(p_ferro: float, rng: np.random.Generator) -> float:
        """
        - We set h=0
        - We sample J from a bimodal distrubtation given by:
            J_ij = p * delta(J_ij - 1) + (1-p) * delta(J_ij + 1)
        """ 
        # J is sampled from a bimodal distribution
        if rng.random() < p_ferro:
            return 1.0
        # Assign 1.0 (anti-ferro) with prob 1-p
        else:
            return -1.0

    def hamiltonian(self) -> np.matrix:
        """ Determine the Hamiltonian matrix of this model. This method is very
            slow for large models """
        return -np.matrix(self._interaction_hamiltonian())
    
    def interactions(self) -> Iterable[tuple[int, int, float]]:
        """ Get an iterator over all interactions in the model as tuples (i, j,
            strength) """
        for (i, j), strength in self._interaction.items():
            yield (i, j, strength)

    def _interaction_hamiltonian(self) -> np.matrix:
        """ Get the interaction part of the Hamiltonian of this model """
        current = np.zeros((2 ** self._spin_count, 2 ** self._spin_count))
        # Lattice interactions
        for (i, j), strength in self._interaction.items():
            current += strength * reduce(np.kron, [
                np.identity(2 ** i),
                Z_MATRIX,
                np.identity(2 ** (j - i - 1)),
                Z_MATRIX,
                np.identity(2 ** (self._spin_count - j - 1))
            ])
        return np.matrix(current)

    def partition_function(self, beta: float) -> float: # taken from DiracWMC package to compute exact Z
        """ Returns the exact partition function of the model, using
            matrix exponentiation, given the inverse temperature beta. This
            method is very slow for large models """
        # Partition function is Tr(e^(-beta*H))
        return expm(-beta * self.hamiltonian()).trace()

    def get_wcnf_matrix(self, beta: float = 1.0) -> WCNFMatrix:
        """ Convert the given Ising model to a WCNFMatrix, the trace of which is
            equal to the partition function of the Ising model at inverse
            temperature beta """
        index = Index()
        n = len(self)
        regs = [Reg(index) for _ in range(n)]
        interactions = reduce(lambda x, y: x * y, (self.exp_zz_rotation(beta * strength, index) | (regs[i], regs[j]) for i, j, strength in self.interactions()))
        return (interactions).mat