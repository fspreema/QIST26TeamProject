from __future__ import annotations

import math
import os
import random
import sys
from itertools import product
from typing import Iterable

_here = os.path.dirname(os.path.abspath(__file__))
_dirac_root = os.path.join(_here, "..", "external", "DiracWMC")
sys.path.insert(0, os.path.join(_dirac_root, "wcnf_matrix"))

from wcnf_matrix import CNF, WeightFunction, BoolVar, DPMC

MAX_EXACT_SPINS = 20


class IsingModel:

    def __init__(self, n_spins: int):
        self.n_spins = n_spins
        self._interactions: dict[tuple[int, int], float] = {}
        self.field: list[float] = [0.0] * n_spins

    def set_interaction(self, i: int, j: int, strength: float):
        if i == j:
            raise ValueError("a spin cannot interact with itself")
        key = (i, j) if i < j else (j, i)
        self._interactions[key] = strength

    def interactions(self) -> Iterable[tuple[int, int, float]]:
        for (i, j), strength in self._interactions.items():
            yield i, j, strength

    def hamiltonian(self, spins: tuple[int, ...]) -> float:
        total = 0.0
        for i, j, strength in self.interactions():
            total -= strength * spins[i] * spins[j]
        for i, h in enumerate(self.field):
            total -= h * spins[i]
        return total

    def partition_function_exact(self, beta: float,
    max_spins: int = MAX_EXACT_SPINS) -> float:
        if self.n_spins > max_spins:
            raise ValueError(
                f"{self.n_spins} spins: 2**{self.n_spins} configurations "
                f"is not tractable by brute force (cap is {max_spins})")
        total = 0.0
        for spins in product((-1, 1), repeat=self.n_spins):
            total += math.exp(-beta * self.hamiltonian(spins))
        return total


def ising_to_wcnf(model: IsingModel, beta: float = 1.0) -> (
tuple[CNF, WeightFunction]):
    spin_vars = [BoolVar() for _ in range(model.n_spins)]
    aux_vars: dict[tuple[int, int], BoolVar] = {}
    cnf = CNF()

    for i, j, _ in model.interactions():
        r = BoolVar()
        aux_vars[i, j] = r
        xi, xj = spin_vars[i], spin_vars[j]
        cnf.add_clause(
            [xi, xj, r], [xi, -xj, -r], [-xi, xj, -r], [-xi, -xj, r],
        )

    all_vars = spin_vars + list(aux_vars.values())
    weight_func = WeightFunction(all_vars)
    weight_func.fill(1.0)

    for i, h in enumerate(model.field):
        weight_func[spin_vars[i], True] = math.exp(beta * h)
        weight_func[spin_vars[i], False] = math.exp(-beta * h)

    for i, j, strength in model.interactions():
        r = aux_vars[i, j]
        weight_func[r, True] = math.exp(beta * strength)
        weight_func[r, False] = math.exp(-beta * strength)

    return cnf, weight_func


def _demo():
    random.seed(0)
    model = IsingModel(4)
    for i in range(3):
        model.set_interaction(i, i + 1, random.uniform(-1.0, 1.0))
    model.field = [random.uniform(-0.5, 0.5) for _ in range(4)]

    beta = 1.0
    exact = model.partition_function_exact(beta)
    cnf, weight_func = ising_to_wcnf(model, beta)
    n_vars = len(list(weight_func))
    print(f"4-spin chain, beta={beta}")
    print(f"  exact Z            = {exact:.6f}")
    print(f"  WMC instance        = {n_vars} variables, {cnf.num_clauses} clauses")

    try:
        result = DPMC().model_count(cnf, weight_func)
        if result.success:
            rel_error = abs(result.model_count / exact - 1)
            print(f"  DPMC model count    = {result.model_count:.6f}  "
            f"(rel. error {rel_error:.2e}, runtime {result.runtime:.4f}s)")
        else:
            print("  DPMC returned success=False -- is Docker running?")
    except Exception as exc:
        print(f"  Skipped DPMC run ({type(exc).__name__}: {exc}) -- Docker "
        f"may not be available here; the exact-Z-vs-instance-size check "
        f"above doesn't need it.")


if __name__ == "__main__":
    _demo()
