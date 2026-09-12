"""Bounded integration checks; run with the project Python from any directory."""
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[2] / 'external/DiracWMC/wcnf_matrix')]
import numpy as np
from ising_model import IsingModelWMC, IsingParameters
from local_solvers import CachetExperimental, TensorOrderLocal


def main():
    records = []
    cases = []
    for n, seed in [(4, 0), (4, 1), (6, 0), (6, 1), (8, 0)]:
        h, j = IsingParameters(n, False).random_sampling(seed=seed)
        if seed == 1:
            h = np.random.default_rng(71).uniform(-.5, .5, n)
        cases.append((f'random_n{n}_seed{seed}', h, j))
    cases.append(('isolated_spins', np.zeros(4), np.zeros((4, 4))))
    for name, h, j in cases:
        n = len(h)
        expected = float(sum(np.exp(s @ j @ s + h @ s)
                             for s in map(np.array, itertools.product((-1, 1), repeat=n))))
        cnf, wf = IsingModelWMC(n, j, h, 1.).build_partition_formula()
        start = perf_counter()
        result = CachetExperimental(timeout=5).model_count(cnf, wf)
        record = dict(case=name, expected=expected, success=result.success,
                      Z=result.model_count, elapsed=perf_counter()-start,
                      correct=bool(result.success and np.isclose(result.model_count, expected, rtol=1e-5)))
        records.append(record)
        print(json.dumps(record), flush=True)
        (HERE / 'cachet_validation.json').write_text(json.dumps(records, indent=2))
    h, j = IsingParameters(40, False).random_sampling(seed=0)
    cnf, wf = IsingModelWMC(40, j, h, 1.).build_partition_formula()
    reference = TensorOrderLocal(timeout=10).model_count(cnf, wf)
    start = perf_counter()
    result = CachetExperimental(timeout=15).model_count(cnf, wf)
    record = dict(case='random_n40_seed0', expected=reference.model_count,
                  reference_success=reference.success, success=result.success,
                  Z=result.model_count, elapsed=perf_counter()-start,
                  correct=bool(reference.success and result.success and
                               np.isclose(result.model_count, reference.model_count, rtol=1e-5)))
    records.append(record)
    print(json.dumps(record), flush=True)
    (HERE / 'cachet_validation.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
