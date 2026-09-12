# Local solver fixes — changes to review

The existing shared `external/DiracWMC` files and `ising_model.py` were not changed.
The existing `tensororder:1.8`, `cachet:1.8`, and `dpmc:1.8` images were not replaced.

## What changed

| File | Purpose |
|---|---|
| `../run_ising.py` | Fix the project path after the move to `Fabian/Ising`; select the local TensorOrder adapter; add `--solvers`, all-counter summaries and CSV output; show each solver's actual default timeout. |
| `adapters.py` | TensorOrder original-weight formatting and direct result parsing; separate experimental Cachet adapter with Decimal normalization/rescaling; container cleanup and result validation. |
| `container_runner.py` | Read-only runner mounted into temporary containers. TensorOrder uses `--weights=mcc`. Each subprocess has a timeout. |
| `__init__.py` | Expose the two local adapter classes. |
| `cachet_build/Dockerfile` | Build `fabian-cachet-experimental:1` from the existing Cachet image, without replacing it. |
| `cachet_build/apply_candidate.py` | Apply the two previously investigated memory-safety candidates only inside the experimental image. |
| `cachet_build/candidate.patch` | Human-readable copy of the candidate changes. |
| `test_adapters.py` | Checks input preservation, formatting, numerical rescaling, parsing, failures, and timeout validation. |
| `verification.json` | Results from actual container checks. |
| `runner_changes.diff` | Exact before/after diff of the only existing Python file edited for this change. |

No projector algebra was changed. A model/formula is still constructed once per
lattice size or random instance and reused across solvers.

## Default behavior

The default solvers are now **DPMC, TensorOrder, and CachetExperimental**, as requested.
TensorOrder means the fixed local adapter. Cachet keeps its experimental label and
30-second limit. Every selected counter appears in the final summary and CSV, even
when a batch fails. Failed results are never fabricated as successful timing points.

From the project root, using the project environment:

```sh
.venv/bin/python Fabian/Ising/run_ising.py --experiment random
```

Or from `Fabian/Ising` with the environment activated:

```sh
python run_ising.py --experiment random
```

To specify the same three counters explicitly:

```sh
python run_ising.py --experiment random --solvers DPMC TensorOrder CachetExperimental
```

The new image has been built on this machine. On another machine it must be built
separately, from the project root:

```sh
docker build --platform linux/amd64 -t fabian-cachet-experimental:1 Fabian/Ising/local_solvers/cachet_build
```

The original unmodified Cachet adapter is available through `--solvers Cachet` if
you need a baseline, but it retains the known crash and normalization limitations.

## Numerical changes

TensorOrder receives both original literal weights instead of normalized
probabilities. It no longer multiplies by an enormous correction factor. This
addresses both the demonstrated 50-spin overflow and 60-spin underflow without
changing precision, clauses, or physical parameters.

CachetExperimental writes short double-precision weight tokens to fit Cachet's
legacy 64-byte token buffer. It uses Decimal for the normalization factor and parses its raw
printed probability as Decimal before rescaling. This prevents host float overflow
or premature underflow during rescaling. It does not guarantee that Cachet's own
calculation is accurate on all inputs or repair any additional native defects.

## Validation and limits

- Seven focused adapter tests pass, including the legacy token-length constraint.
- The local TensorOrder adapter returns the exact-enumeration reference for a
  three-spin test and agrees with the previous verified values at 50 and 60 spins.
- The experimental image builds independently. Its small-case result is checked
  within Cachet's limited printed precision, and its timeout handling is tested.
- The earlier diagnostic candidate still timed out at 40 spins within 45 seconds;
  it is **not** established as a fast, fully repaired solver for the large benchmark.
- CachetExperimental defaults to **30 seconds per solver run**. DPMC and TensorOrder
  retain **300 seconds**. Container waits allow a further 30 seconds. This is not
  an overall experiment deadline.
- TensorOrder timing comes from its `Total Time` field; Cachet timing is the solver
  subprocess elapsed time, matching the existing adapter's convention.
- This work does not regenerate existing plots or run a full five-instance sweep.

To run the non-Docker tests from the project root:

```sh
PYTHONPATH="Fabian/Ising:external/DiracWMC/wcnf_matrix" .venv/bin/python -m unittest local_solvers.test_adapters
```

## Reversibility

`runner_changes.diff` records the existing-file edits. The adapter package is
entirely local, and the experimental image has a separate name. The shared package
does not import these adapters, so classmates' scripts are unaffected unless they
explicitly use this local runner or these adapters.

## Additional bounded Cachet verification

Six cases passed against exact enumeration with relative tolerance 1e-5: random
graphs at 4 and 6 spins with two seeds each (including mixed fields), an 8-spin
graph, and four isolated spins. The small output differences are within Cachet's
printed precision. A 40-spin seed-0 case timed out after 15 seconds, with no result.
See `cachet_validation.json`; `verify_cachet.py` reproduces these bounded checks.
This does not establish correctness or acceptable runtime on the larger cases.

The CSV uses FAILED when a complete valid average is unavailable; it is written
even if every solver fails and no logarithmic plot can be generated.
