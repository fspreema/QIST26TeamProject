"""Isolated reproductions; never edits project code or installed solver images.

Run with the project's Python:
  python Fabian/debug_solver_issues/debug_solvers.py prepare
  python Fabian/debug_solver_issues/debug_solvers.py summarize
Container commands and their raw outputs are stored separately in this folder.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, localcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Fabian"), str(ROOT / "external/DiracWMC/wcnf_matrix")]


def prepare():
    import numpy as np
    from ising_model import IsingParameters, IsingModelWMC
    from wcnf_matrix.cnf.model_counter.formats import format_cachet, format_dpmc

    sources = [*ROOT.glob("Fabian/*.py"), *ROOT.glob("external/DiracWMC/**/*.py")]
    (HERE / "original_hashes.json").write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }, indent=2))
    for n in (40, 50, 60):
        h, j = IsingParameters(n, False).random_sampling(seed=0)
        cnf, wf = IsingModelWMC(n, j, h, 1.0).build_partition_formula()
        normalized = wf.copy()
        with localcontext() as ctx:
            ctx.prec = 80
            factor = Decimal(1)
            logs = []
            for v in wf:
                total = float(wf[v, False] + wf[v, True])
                assert math.isfinite(total) and total > 0
                factor *= Decimal.from_float(total)
                logs.append(math.log(total))
                normalized[v, False] = wf[v, False] / total
                normalized[v, True] = wf[v, True] / total
            factor_string = str(factor)
        text = format_cachet(cnf, normalized)
        # Validate exactly what the solvers will receive.
        lines = text.splitlines()
        header = lines[0].split()
        weights = [line.split() for line in lines if line.startswith("w ")]
        clauses = [line for line in lines if not line.startswith(("p ", "w "))]
        assert len(weights) == int(header[2]) == len(wf)
        assert len(clauses) == int(header[3])
        assert all(math.isfinite(float(row[2])) and 0 <= float(row[2]) <= 1 for row in weights)
        assert all(row.split()[-1] == "0" for row in clauses)
        assert all(1 <= abs(int(lit)) <= len(wf) for row in clauses for lit in row.split()[:-1])
        (HERE / f"n{n}.cnf").write_text(text + "\n")
        dpmc_text = format_dpmc(cnf, wf)
        (HERE / f"n{n}.dpmc").write_text(dpmc_text + "\n")
        mcc_lines = []
        for line in dpmc_text.splitlines():
            if line.startswith("c p weight "):
                mcc_lines.append("w " + line[len("c p weight "):])
            elif not line.startswith("c "):
                mcc_lines.append(line)
        (HERE / f"n{n}.mcc").write_text("\n".join(mcc_lines) + "\n")
        metadata = dict(n=n, seed=0, edges=int(np.count_nonzero(j)), variables=len(wf),
                        clauses=len(clauses), factor_decimal=factor_string,
                        log_factor=math.fsum(logs),
                        log10_upper_bound_Z=float((n*math.log(2)+np.abs(j).sum())/math.log(10)),
                        input_sha256=hashlib.sha256((text + "\n").encode()).hexdigest())
        (HERE / f"n{n}.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)


def summarize():
    for p in sorted(HERE.glob("tensor_n*.json")):
        result = json.loads(p.read_text())
        n = int(re.search(r"n(\d+)", p.stem)[1])
        meta = json.loads((HERE / f"n{n}.json").read_text())
        matches = re.findall(r"^Count:\s*(\S+)", result["stdout"], re.M)
        if matches:
            with localcontext() as ctx:
                ctx.prec = 80
                corrected = Decimal(matches[-1])
                if not p.stem.endswith('_mcc'):
                    corrected *= Decimal(meta["factor_decimal"])
            print(p.name, "raw:", matches[-1], "rescaled:", corrected)
    originals = json.loads((HERE / "original_hashes.json").read_text())
    changed = [name for name, digest in originals.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    print("Changed original Python files:", changed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "summarize"))
    args = parser.parse_args()
    globals()[args.action]()
