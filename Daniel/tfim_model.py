from __future__ import annotations

import math
import os
import sys
from functools import reduce
from typing import Iterable

import numpy as np
from scipy.linalg import expm

_here = os.path.dirname(os.path.abspath(__file__))
_dirac_root = os.path.join(_here, "..", "external", "DiracWMC")
sys.path.insert(0, os.path.join(_dirac_root, "wcnf_matrix"))

from wcnf_matrix import (
    CNF, WeightFunction, BoolVar, WCNFMatrix, Reg, Index, LogVarRep, DPMC,
)

X_MATRIX = np.array([[0, 1], [1, 0]])
Z_MATRIX = np.array([[1, 0], [0, -1]])

MAX_EXACT_SPINS = 12


class QuantumIsingModel:
    def __init__(self, n_spins: int):
        self.n_spins = n_spins
        self._interactions: dict[tuple[int, int], float] = {}
        self.external_field_x = 0.0
        self.external_field_z = 0.0

    def set_interaction(self, i: int, j: int, strength: float):
        if i == j:
            raise ValueError("a spin cannot interact with itself")
        key = (i, j) if i < j else (j, i)
        self._interactions[key] = strength

    def interactions(self) -> Iterable[tuple[int, int, float]]:
        for (i, j), strength in self._interactions.items():
            yield i, j, strength

    def _kron_at(self, single_qubit_matrix: np.ndarray, site: int) -> np.ndarray:
        n = self.n_spins
        return reduce(np.kron, [
            np.identity(2 ** site),
            single_qubit_matrix,
            np.identity(2 ** (n - site - 1)),
        ])

    def hamiltonian(self) -> np.ndarray:
        n = self.n_spins
        h = np.zeros((2 ** n, 2 ** n))
        for i, j, strength in self.interactions():
            zi = self._kron_at(Z_MATRIX, i)
            zj = self._kron_at(Z_MATRIX, j)
            h -= strength * (zi @ zj)
        if self.external_field_x:
            for i in range(n):
                h -= self.external_field_x * self._kron_at(X_MATRIX, i)
        if self.external_field_z:
            for i in range(n):
                h -= self.external_field_z * self._kron_at(Z_MATRIX, i)
        return h

    def partition_function_exact(self, beta: float,
    max_spins: int = MAX_EXACT_SPINS) -> float:
        if self.n_spins > max_spins:
            raise ValueError(
                f"{self.n_spins} spins: a dense 2**{self.n_spins} x "
                f"2**{self.n_spins} matrix exponential is not tractable "
                f"here (cap is {max_spins})")
        return float(np.real(np.trace(expm(-beta * self.hamiltonian()))))


def _identity(index: Index) -> WCNFMatrix:
    x = BoolVar()
    wf = WeightFunction([x])
    wf.fill(1.0)
    xr = LogVarRep(2, [x])
    return WCNFMatrix(index, CNF(), wf, [xr], [xr])


def _exp_z(theta: float, index: Index) -> WCNFMatrix:
    x = BoolVar()
    wf = WeightFunction([x])
    wf[x, 0] = math.exp(theta)
    wf[x, 1] = math.exp(-theta)
    xr = LogVarRep(2, [x])
    return WCNFMatrix(index, CNF(), wf, [xr], [xr])


def _exp_zz(theta: float, index: Index) -> WCNFMatrix:
    x, y, r = BoolVar(), BoolVar(), BoolVar()
    cnf = CNF([
        [-x, -y, r],
        [x, y, r],
        [x, -y, -r],
        [-x, y, -r],
    ])
    wf = WeightFunction([x, y, r])
    wf.fill(1.0)
    wf[r, 0] = math.exp(-theta)
    wf[r, 1] = math.exp(theta)
    xr, yr = LogVarRep(2, [x]), LogVarRep(2, [y])
    return WCNFMatrix(index, cnf, wf, [xr, yr], [xr, yr])


def _hadamard(index: Index) -> WCNFMatrix:
    x, y, r = BoolVar(), BoolVar(), BoolVar()
    cnf = CNF([
        [-r, x],
        [-r, y],
        [r, -x, -y],
    ])
    wf = WeightFunction([x, y, r])
    wf.fill(1.0)
    wf[r, 0] = 1.0 / math.sqrt(2)
    wf[r, 1] = -1.0 / math.sqrt(2)
    xr, yr = LogVarRep(2, [x]), LogVarRep(2, [y])
    return WCNFMatrix(index, cnf, wf, [xr], [yr])


def quantum_ising_to_wcnf(model: QuantumIsingModel, beta: float = 1.0,
trotter_layers: int = 5) -> tuple[CNF, WeightFunction]:
    index = Index()
    n = model.n_spins
    regs = [Reg(index) for _ in range(n)]

    identity_layer = reduce(lambda a, b: a * b,
    (_identity(index) | reg for reg in regs))

    interactions = list(model.interactions())
    if interactions:
        zz_layer = reduce(lambda a, b: a * b,
        (_exp_zz(beta * strength / trotter_layers, index) | (regs[i], regs[j])
        for i, j, strength in interactions))
    else:
        zz_layer = identity_layer

    z_field_layer = reduce(lambda a, b: a * b,
    (_exp_z(beta * model.external_field_z / trotter_layers, index) | reg
    for reg in regs))

    hadamard_layer = reduce(lambda a, b: a * b,
    (_hadamard(index) | reg for reg in regs))

    x_field_layer = reduce(lambda a, b: a * b,
    (_exp_z(beta * model.external_field_x / trotter_layers, index) | reg
    for reg in regs))

    one_trotter_step = (zz_layer * z_field_layer * hadamard_layer *
    x_field_layer * hadamard_layer)

    total = identity_layer
    for _ in range(trotter_layers):
        total = total * one_trotter_step

    return total.mat.trace_formula()


def _demo():
    model = QuantumIsingModel(3)
    model.set_interaction(0, 1, 0.4)
    model.set_interaction(1, 2, 0.4)
    model.external_field_x = 0.6
    model.external_field_z = 0.0

    beta = 1.0
    exact = model.partition_function_exact(beta)
    print(f"3-spin TFIM chain, beta={beta}, exact Z = {exact:.6f}\n")

    for layers in (1, 2, 4, 8):
        cnf, weight_func = quantum_ising_to_wcnf(model, beta, layers)
        n_vars = len(list(weight_func))
        print(f"trotter_layers={layers}: {n_vars} vars, "
        f"{cnf.num_clauses} clauses", end="")
        try:
            result = DPMC().model_count(cnf, weight_func)
            if result.success:
                rel_error = abs(result.model_count / exact - 1)
                print(f"  DPMC={result.model_count:.6f}  "
                f"rel_error={rel_error:.2e}  runtime={result.runtime:.4f}s")
            else:
                print("  DPMC returned success=False -- is Docker running?")
        except Exception as exc:
            print(f"  Skipped DPMC run ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    _demo()
