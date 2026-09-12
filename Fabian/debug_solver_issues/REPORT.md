# Isolated investigation: TensorOrder and Cachet

Date: 12 September 2026.

All edits made for this investigation are in `Fabian/debug_solver_issues/`. Existing
model/runner/package files and installed images were not edited by this investigation.
Temporary containers were removed automatically. Some existing project files changed
elsewhere during the investigation (detected by the starting hashes); the experiments
below use the saved inputs, so they do not depend on subsequent edits to the model.

## Conclusions and decisions

1. **TensorOrder: a tested adapter fix is available.** Send the original literal
   weights using TensorOrder's existing `mcc` format and remove the probability
   normalization/rescaling in its adapter. Keep NumPy float64. This leaves the
   projector encoding and clauses unchanged and does not require rebuilding its image.
2. **Cachet: there are native C++ memory-safety defects, not merely insufficient
   floating-point precision.** A candidate patch is saved for review. It removes the
   observed early failure in bounded tests but has not produced a validated result
   on the 40-spin case within 45 seconds. Do not treat it as a finished repair.
3. **Cachet's adapter also needs stable rescaling** if its native solver is repaired:
   it shares the normalization mechanism that overflowed in TensorOrder. Fixing the
   executable alone is insufficient for the larger instances.

## TensorOrder: two separate numerical failures

The installed NumPy backend selects `numpy.float64`, and the command-line default
is `--entry_type=float64`. This is 64-bit floating point. The host factor is also
computed using ordinary double-precision arithmetic. A float64 value can represent
roughly 1e308 at the upper end; its smallest positive subnormal is roughly 5e-324.

For the saved graphs (seed 0), the adapter normalizes every variable's two weights
to sum to one. It multiplies all these sums into a correction factor. Auxiliary
variables make this factor vastly larger than the physical partition function.

| Spins | Variables / clauses | Correction factor | Raw normalized TensorOrder count |
|---|---|---|---|
| 40 | 1776 / 3776 | 1.4417874789e235 | Original user run succeeds |
| 50 | 2437 / 5182 | 1.1513478625e321 | 2.870488892009924e-302 |
| 60 | 3191 / 6786 | 6.8976647646e419 | 0.0 |

At 50 spins the factor overflows, even though the raw count is finite. Decimal
rescaling of that raw count gives approximately 3.304931250259463e19. At 60 spins
the raw result has already underflowed to zero; changing only the final factor
cannot recover it. Given the recovered Z, the true normalized count is about
2.65e-396, beyond float64's range.

This is not an enormous physical Z: the bounds from the actual graphs are below
1e32 for the 50-spin case and below 1e41 for the 60-spin case.

### Tested alternative

The installed TensorOrder parser already supports `--weights=mcc`, where a line
`w <signed literal> <weight>` gives each literal its original weight. Both positive
and negative literal weights are supplied. The diagnostic `.mcc` inputs were
derived from the exact saved `.dpmc` input files; no clauses or physical parameters
were changed.

| Spins | TensorOrder, original weights | DPMC printed result | Relative difference |
|---|---|---|---|
| 40 | 7.399314796722865e15 | 7.39931e15 | 6.48e-7 |
| 50 | 3.3049312502594515e19 | 3.30493e19 | 3.78e-7 |
| 60 | 1.8293505471602083e24 | 1.82935e24 | 2.99e-7 |

These agree to DPMC's six significant printed digits. A separate three-spin case
with an interaction between spins 0 and 2 and mixed fields matched direct
enumeration: **Z = 8.573208691399596**.

TensorOrder reported approximately 0.75, 1.14, and 1.72 seconds for these three
unnormalized runs. These are individual diagnostic measurements, not five-run
benchmark means or guaranteed speedups.

An experimental extended-precision harness timed out before giving a result;
it is not a validated alternative. Unnormalized input is the better-supported
proposal here. It does not guarantee numerical stability for arbitrarily large
future problems; finite-result checks and numerical reference checks should remain.

### Proposed production changes, NOT applied

- In the TensorOrder adapter, serialize original positive and negative weights
  into MCC format instead of calling `_normalize_problems()`.
- In its container runner, change `--weights=cachet` to `--weights=mcc`.
- Parse the returned count directly, with no correction-factor multiplication.
- Preserve existing timeouts, finite-result checks, and timing conventions.

Both ends must change together. Sending the current probability-only format while
changing only the flag is not the tested fix.

## Cachet: native memory errors

The unmodified binary crashes on the saved `n40.cnf` with exit -11 after about
1.37 seconds. Running it directly excludes the benchmark loop and formula reuse
as the cause. Basic input validation passed: variable IDs, clause count, clause
terminators, and finite probability weights in [0,1].

A copied source build with AddressSanitizer reports **heap-use-after-free**:

- `CSolver::decide_next_branch()`, original `zchaff_solver.cpp:1490`, increments
  a list iterator pointing to deleted storage.
- The list node was freed by `CSolver::percolate_up()`, line 5496, called from
  the cache-hit path at line 1577.

A separate backtrace of the native crash reaches watched-literal handling:
`CSolver::set_var_value_current_dl()` and `CLitPoolElement::direction()`.
The literal-pool reallocation code also stores the pointer displacement in a
32-bit `int` (`zchaff_dbase.cpp`, original line 381). This is a pointer-width issue,
distinct from floating-point precision.

The offset-logging reproduction confirms actual truncation on the saved failing
input: **full displacement = 35184096801792; stored int = -275287040**, followed
by exit -11. See `cachet_offset_trace_n40.json`. This makes pointer truncation a
confirmed defect in this environment, not merely a source-level suspicion.

### Candidate patch and limits

`cachet_candidate.patch` contains two isolated candidate changes:

- Do not increment the component iterator after the last component was processed;
  propagation can already have erased the preceding node used as the iterator.
- Widen the literal-pool displacement to `std::ptrdiff_t`.

Both the sanitizer and optimized candidate builds passed the original early crash
point, but neither completed the 40-spin instance within the 45-second diagnostic
limit. The sanitizer run produced no new diagnostic before timeout. The candidate
passed the small three-spin check within the precision of Cachet's printed count
(approximately 2.43e-6 relative difference after rescaling).

This is evidence of progress, not proof that all memory defects are fixed or that
the candidate will be fast enough for the intended benchmark. A production repair
should review pointer relocation carefully (including pointer lifetime around
`realloc`), run additional memory checks, and validate counts on more small graphs.

Disabling cross-component implications (`-r`) did not prevent the original crash.
A test using Cachet's special `-1` notation for unit-weight variables also timed
out on the candidate build; it is not a demonstrated runtime fix.

Cachet reads weights as `double` and uses `long double` for some accumulated
probabilities; the investigated failure is not evidence of float32 arithmetic.
The wrapper still converts printed probabilities to Python float and uses the
same potentially overflowing correction factor. If Cachet is repaired, Decimal
or log-based rescaling and preserving the raw printed number will need testing too.

## Files and reproduction

- `debug_solvers.py prepare`: snapshots hashes of existing Python sources and
  generates the three saved graph inputs and factor metadata. Re-running this
  overwrites these diagnostic inputs; retain the current files to reproduce this report.
- `container_probe.py`: runs a command with a hard timeout and saves stdout,
  stderr, exit status, and elapsed time as JSON.
- `dpmc_reference.py`: obtains reference results for the identical saved formulas.
- `build_cachet_probe.py`: copies Cachet source from the image into the debug folder
  and builds it there. Default: AddressSanitizer. `--patched`: candidate fixes.
  `--patched --release`: optimized candidate. `--offset-trace --release`: original
  algorithm with pointer-relocation logging.
- `cachet_candidate.patch`: proposed source patch, not applied to installed Cachet.
- `tensor_n*_mcc.json`, `tensor_n50.json`, `tensor_n60.json`: raw numerical evidence.
- `cachet_original_n40.json`, `cachet_asan_n40.json`, `cachet_backtrace_n40.json`,
  `cachet_offset_trace_n40.json`, and `cachet_patched_*.json`: crash and candidate-build evidence.
- `verified_results.json`: numerical comparison table.

Container probes mount only this debug directory at `/debug`. For example, from
the project root:

```sh
docker run --rm --platform linux/amd64 \
  --mount "type=bind,source=$PWD/Fabian/debug_solver_issues,target=/debug" \
  --entrypoint python tensororder:1.8 \
  /debug/container_probe.py --output /debug/tensor_n60_mcc.json \
  --cwd /usr/src/app/TensorOrder --input /debug/n60.mcc -- \
  python ./src/tensororder.py --planner=factor-Flow --weights=mcc --timeout=30
```

The TensorOrder proposal can be accepted independently of further Cachet work.
Keep Cachet failures visible as failures rather than including them in averages.
