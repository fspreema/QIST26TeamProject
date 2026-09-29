from abc import abstractmethod
from wcnf_matrix import CNF, WeightFunction

class BaseModel:
    # A base class for use in the benchmarking class. This makes the benchmark
    # class reusable for other models we want to test.

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

    @staticmethod
    @abstractmethod
    def generate_lattice(shape: tuple[int, int]) -> BaseModel:
        pass

    @abstractmethod
    def partition_function(self, beta: float) -> float:
        pass

    @abstractmethod
    def get_wcnf(self, beta: float = 1.0) -> tuple[CNF, WeightFunction]:
        pass