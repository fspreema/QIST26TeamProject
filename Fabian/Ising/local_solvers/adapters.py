"""Local adapters using existing TensorOrder and a separate experimental Cachet image."""
import json
import math
import re
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from uuid import uuid4

from wcnf_matrix import ModelCounter, ModelCounterResult

RUNNER = Path(__file__).with_name("container_runner.py").resolve()


def ordered_variables(weight_func):
    return sorted(weight_func.domain)


def weights_for(weight_func, variable):
    positive = float(weight_func[variable, True])
    negative = float(weight_func[variable, False])
    if not all(math.isfinite(w) and w >= 0 for w in (positive, negative)):
        raise ValueError("Literal weights must be finite and nonnegative")
    return positive, negative


def clauses_text(cnf, variables):
    indices = {v: i for i, v in enumerate(variables, 1)}
    return [" ".join(str(indices[lit.var] if lit.value else -indices[lit.var])
                     for lit in clause) + " 0" for clause in cnf.clauses]


def format_mcc(cnf, weight_func):
    """Preserve both original literal weights; do not normalize them."""
    variables = ordered_variables(weight_func)
    lines = [f"p cnf {len(variables)} {cnf.num_clauses}"]
    for i, v in enumerate(variables, 1):
        positive, negative = weights_for(weight_func, v)
        lines.extend((f"w {i} {positive!r}", f"w {-i} {negative!r}"))
    return "\n".join(lines + clauses_text(cnf, variables)) + "\n"


def format_cachet_decimal(cnf, weight_func):
    """Normalize a copy in the output only, retaining the correction in Decimal."""
    variables = ordered_variables(weight_func)
    lines = [f"p cnf {len(variables)} {cnf.num_clauses}"]
    with localcontext() as ctx:
        ctx.prec = 80
        factor = Decimal(1)
        for i, v in enumerate(variables, 1):
            positive, negative = weights_for(weight_func, v)
            total = Decimal.from_float(positive) + Decimal.from_float(negative)
            if total == 0:
                factor = Decimal(0)
                probability = Decimal('0.5')
            else:
                factor *= total
                probability = Decimal.from_float(positive) / total
            # Cachet parses each token into a 64-byte buffer and reads double.
            # Keep the factor in Decimal, but write a short round-trip double.
            lines.append(f"w {i} {float(probability)!r}")
        return "\n".join(lines + clauses_text(cnf, variables)) + "\n", factor


def parse_result(output, solver, elapsed, factor=Decimal(1)):
    if solver == "tensororder":
        count_match = re.search(r"^Count:\s*(\S+)", output, re.M)
        time_match = re.search(r"^Total Time:\s*(\S+)", output, re.M)
        if not count_match or not time_match:
            raise ValueError("TensorOrder output is missing its count or runtime")
        count = float(count_match[1])
        runtime = float(time_match[1])
    else:
        match = re.search(r"^Satisfying probability\s+(\S+)", output, re.M)
        if not match:
            raise ValueError("Cachet output is missing its satisfying probability")
        # Do not convert a tiny raw probability to float before rescaling.
        with localcontext() as ctx:
            ctx.prec = 80
            count = float(Decimal(match[1]) * factor)
        # Match the shared Cachet adapter: wall time of the solver subprocess.
        runtime = elapsed
    if not math.isfinite(count) or count < 0:
        raise ValueError(f"Invalid model count: {count}")
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"Invalid solver runtime: {runtime}")
    return ModelCounterResult(True, runtime, count)


class LocalDockerCounter(ModelCounter):
    image = ""
    solver = ""
    default_timeout = 300

    def __init__(self, *, timeout=None, show_log=False):
        timeout = self.default_timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        super().__init__(timeout=timeout, show_log=show_log)

    @classmethod
    def is_available(cls):
        try:
            result = subprocess.run(["docker", "image", "inspect", cls.image],
                                    capture_output=True, timeout=10)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _stop_container(self, name):
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def model_count(self, cnf, weight_func):
        name = "fabian-ising-" + uuid4().hex
        process = None
        try:
            if self.solver == "tensororder":
                formula, factor = format_mcc(cnf, weight_func), Decimal(1)
            else:
                formula, factor = format_cachet_decimal(cnf, weight_func)
            command = ["docker", "run", "--rm", "--pull=never", "-i", "--platform", "linux/amd64",
                       "--name", name, "--mount",
                       f"type=bind,source={RUNNER},target=/tmp/fabian_runner.py,readonly",
                       "--entrypoint", "python", self.image, "/tmp/fabian_runner.py"]
            request = json.dumps(dict(solver=self.solver, formula=formula, timeout=self._timeout))
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = process.communicate(request.encode(), timeout=self._timeout + 30)
            except BaseException:
                self._stop_container(name)
                process.kill()
                process.communicate()
                raise
            if process.returncode:
                raise RuntimeError(err.decode(errors="replace")[-2000:])
            record = json.loads(out)
            if record["timed_out"]:
                raise RuntimeError(f"Solver exceeded {self._timeout:g} seconds")
            if record["returncode"]:
                raise RuntimeError(f"Solver exit {record['returncode']}: "
                                   + record["stderr"][-1500:] + record["stdout"][-1500:])
            if self._show_log:
                print(record["stdout"], file=sys.stderr)
                print(record["stderr"], file=sys.stderr)
            return parse_result(record["stdout"], self.solver, record["elapsed"], factor)
        except Exception as exc:
            print(f"[{type(self).__name__}] {exc}", file=sys.stderr, flush=True)
            return ModelCounterResult(False)


class TensorOrderLocal(LocalDockerCounter):
    """Original literal weights, using the existing TensorOrder image."""
    image = "tensororder:1.8"
    solver = "tensororder"


class CachetExperimental(LocalDockerCounter):
    """Opt-in candidate memory fixes; larger instances remain unvalidated."""
    image = "fabian-cachet-experimental:1"
    solver = "cachet"
    default_timeout = 300
