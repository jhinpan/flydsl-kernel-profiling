---
name: flydsl-kernel-multishape-benchmark
description: >
  Benchmark a FlyDSL kernel against every available baseline (AITER-compiled,
  AITER-Triton, standalone Triton, PyTorch, CK/hipBLASLt where they exist) across
  a real multi-shape ledger on MI350X/gfx950, judge it on the kernel-only
  (CUDA-graph) metric, classify any gap, and emit a promote/tune/rewrite decision.
  Use when asked to "benchmark kernel X", "is FlyDSL's X faster than aiter",
  "add a multi-shape benchmark for X", or "should we promote X".
---

# FlyDSL multi-shape kernel benchmark

This skill drives `benchmarks/` — a multi-shape benchmark **layer** that lives
beside the diagnostic rocprofv3/ATT trace bundles in `examples/<kernel>/`. It
answers a different question than the ATT bundles do: not "where does this one
kernel stall on one diagnostic shape" but "across the shapes that actually
occur in models and serving, is FlyDSL's kernel faster than the field, and if
not, why."

The two layers are keyed by the same op names but land in different trees:
ATT work writes top-level `examples/<op>/` (`REPORT.md` / `README.md` /
`att_viewer/` / `compute_viewer/` / `source/`); this layer writes
`benchmarks/examples/<op>/` (`shape_ledger.jsonl`, `baseline_matrix.yaml`,
`benchmark_results.{jsonl,csv}`, `coverage_matrix.md`, `benchmark_summary.md`,
and optional tier artifacts). **Never clobber the ATT bundle's `REPORT.md` or
the per-example `README.md`.**

## Purpose

- Compare one FlyDSL kernel (the *candidate*) against every reachable baseline
  on a single set of inputs per shape (apples-to-apples).
- Report the **kernel-only** speedup (the fair metric) plus the eager /
  host-overhead view (the launcher story), per baseline, per stage, per model,
  unweighted + production-weighted.
- Surface every failure, incorrect result, and unsupported/disabled provider —
  never silently drop one.
- Classify each sub-parity hot shape and emit a single promotion decision.

## Non-goals

- Not a full ATT profiler. The benchmark runner measures wall-clock device time;
  it does not capture ISA, stalls, or counters. When a hot shape is sub-parity,
  the profiler gate can run a rocprofv3 `--kernel-trace` diagnosis. Full decoded
  ATT capture remains the ATT layer's job (see the repo's top-level `AGENTS.md`).
- Not a microbenchmark of a single shape. One-shape numbers are diagnostic only.
- Not a correctness test suite. It checks correctness against an fp32 reference
  as a gate on whether a timing number is trustworthy; it is not exhaustive
  numerics coverage.
- Does not claim production perf from synthetic/diagnostic shapes alone.

## The four tiers

A complete benchmark touches four tiers, in increasing cost and decreasing
generality:

1. **Correctness** — every provider's output, upcast to fp32, must match the
   fp32 golden (`benchmarks/ops.py: Op.reference`) within a dtype-aware tolerance
   (`benchmarks/common.py: TOL`). An incorrect provider is still timed but its
   row is labeled `incorrect` and excluded from speedup aggregates. The
   multishape runner records correctness inline; `benchmarks.runners.correctness_runner`
   is the optional no-timing gate that emits `correctness_results.jsonl`.
2. **Diagnostic** — the single shape the existing ATT capture used
   (`source.kind=diagnostic`, `stage=diagnostic`). Lets the wall-clock layer and
   the instruction-level layer talk about the same workload.
3. **Multishape** — the real comparison: the AITER model-config sweep + the
   synthetic boundary probes, across the whole baseline matrix. This is the body
   of the benchmark.
4. **Regression** — re-run a frozen ledger on a new FlyDSL commit and diff the
   per-shape speedups to catch a regression before promotion.
   `benchmarks.runners.regression_runner` emits `regression_summary.md` and
   `regressions.jsonl`.

## Data contracts (authoritative schemas in `benchmarks/schemas/`)

Three required on-disk data contracts. Validate with `benchmarks/validate.py`
(uses `jsonschema` when present, falls back to a required-field + enum check).
Optional runner outputs (`correctness_results.jsonl`, `profiles/...`, and
`regression_summary.md` / `regressions.jsonl`) live beside them under
`benchmarks/examples/<op>/`.

### shape_ledger.jsonl — `schemas/shape_ledger.schema.json`
One JSONL row per benchmarked shape. Identity-defining fields are hashed into a
stable `shape_id` (`common.stable_shape_id` -> `sha1:<16 hex>`), so the same
shape gets the same id across runs and importers.

- Required: `shape_id, op_type, kernel_name, model, stage, arch, gpu, dtype,
  layout, args, source, weight, baselines_available, notes`.
- `stage` enum: `prefill | decode | mixed | synthetic | diagnostic | model_config`.
- `source.kind` enum: `aiter_model_shapes | sglang_trace | atom_workload |
  manual | synthetic | diagnostic | model_config`. Each importer owns one or
  more kinds (see `shape_ledgers/README.md`).
- `weight`: `{occurrences, traffic_weight, baseline_time_weight}` — all null
  until a serving trace populates them; this is what makes the weighted geomean
  meaningful.

### baseline_matrix.yaml — `schemas/baseline_matrix.schema.yaml`
Per-kernel declaration of which providers to run and how to reach each adapter.

- Required: `kernel, op_type, dtype, providers`.
- `reference_provider` (default `pytorch`) supplies the dtype-matched baseline;
  the fp32 golden comes from `ops.py`, not from a provider.
- `layout_contract` documents the layout every provider must honor; an adapter
  that converts must set `includes_layout_conversion=True`.
- Each `providers.<name>`: `{enabled, entrypoint, provider, commit_required?,
  backend_visibility_required?, skip_reason?, notes?}`. `entrypoint` is
  `pkg.module:Attr` resolved by `providers/base.py: load_entrypoint`. A disabled
  provider is recorded as `not_configured` with its `skip_reason` — never
  dropped.

### benchmark_results.jsonl — `schemas/benchmark_result.schema.json`
One JSONL row per `(shape_id, provider)`. The CSV (`benchmark_results.csv`) is a
flattened projection. Failed / unsupported / incorrect / disabled rows are
KEPT.

- Required: `run_id, shape_id, provider, provider_detail, candidate_commit, gpu,
  arch, rocm_version, correct, warmup_iters, repeat_iters, benchmark_status`.
- `provider` enum: `flydsl | aiter | aiter_triton | aiter_ck | aiter_asm | ck |
  triton | gluon | hipblaslt | pytorch`.
- `benchmark_status` enum: `ok | failed | skipped | oom | unsupported |
  incorrect | not_configured`.
- Timing fields carry BOTH views: `median_us` (PRIMARY = kernel-only graph time
  when capture succeeds, else eager), `eager_median_us`, `graph_median_us`,
  `host_overhead_us = eager - graph`, plus `timing_method`, `p10/p90`, `stable`.
- Honesty flags: `includes_allocation`, `includes_layout_conversion`,
  `includes_jit`. `provider_detail` records the backend branch (e.g. aiter's
  `>8192 CK` branch); when the backend is opaque it says so.

## Provider adapter interface (`benchmarks/providers/base.py`)

A *provider* is one implementation; an *adapter* binds it to one `op_type`. The
runner builds inputs ONCE via the reference adapter and hands the SAME `inputs`
dict to every provider. Override the small surface:

```python
class MyAdapter(ProviderAdapter):
    name = "flydsl"                 # must be a provider enum value
    includes_allocation = False     # honesty flags -> result row
    includes_jit = False

    def supports(self, shape) -> tuple[bool, str | None]:
        # (supported, reason). MUST NOT raise. Gate dtype/op/runtime here.
        ...
    def run(self, shape, inputs):
        # launch exactly the kernel under test; no alloc/JIT in the timed region
        # unless the matching includes_* flag is set. Return the raw output.
        ...
    # output(), check_correctness(), benchmark() have working defaults.
```

`load_entrypoint("pkg.mod:Attr", op_type)` accepts a `ProviderAdapter`
subclass, a `get_adapter(op_type)` factory, or an instance. Inputs / fp32
reference / roofline live in `benchmarks/ops.py` (one `Op` subclass per
op_type); providers never build their own inputs.

**rmsnorm providers as wired today** (`benchmarks/examples/rmsnorm/baseline_matrix.yaml`):
`flydsl` (`build_rmsnorm_module`), `pytorch` (`F.rms_norm`, also the fp32
reference source), `aiter` (compiled `module_rmsnorm`, records the Python
`N>8192` CK branch, inner CK/HIP/ASM opaque), `aiter_triton`
(`aiter.ops.triton.normalization.rmsnorm`), `triton` (sglang one-pass).
`aiter_ck` / `aiter_asm` / `ck` / `gluon` / `hipblaslt` are honest
support-detection stubs — disabled in the matrix because they are not separately
selectable from Python, not available on this node, or GEMM-only.

## Environment (REQUIRED) — always launch GPU runners via `benchmarks/env.sh`

Importers and report generators are pure-data and run on a CPU-only box. Any
runner that touches the GPU MUST be launched through `benchmarks/env.sh` (or the
`benchmarks/bench` wrapper, which sources it then `exec`s python). Why:

- `flydsl` is pip-installed editable but the editable tree has NO compiled
  `_mlir` extension; the built `.so` lives in `FLYDSL-lab/build-fly/python_packages`.
- Putting that build tree first on `PYTHONPATH` makes `import flydsl.*` resolve
  to the built copy — which (verified) **also unblocks `import aiter`**, because
  `aiter/__init__` imports `flydsl.expr` transitively. Without the build tree,
  `import aiter` raises `ModuleNotFoundError: flydsl._mlir`.
- The `_mlir` `.so` needs `_mlir_libs` on `LD_LIBRARY_PATH`, which the loader
  reads at exec — so it must be set BEFORE the process starts. `env.sh` does
  this; `common.bootstrap_env()` only covers the `PYTHONPATH` half.

Verified node recipe: MI350X gfx950, ROCm 7.2, torch 2.9.1+rocm, triton 3.6.

## CLI commands (exactly as implemented)

### Build the ledger (importers — CPU-only, idempotent upsert)

```bash
# AITER model-config sweep -> per-op shape_ledger.jsonl  (source.kind=aiter_model_shapes)
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --dtype bf16 --ops rmsnorm

# synthetic boundary probes (FlyDSL fast path: N>=2048 & N%2048==0 & 16-bit)
# + the existing ATT diagnostic shape  (source.kind=synthetic / diagnostic)
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm \
  --out benchmarks/examples --synthetic-boundary --diagnostic 32768,8192,bf16
```

### Run the benchmark (GPU — via the bench wrapper)

```bash
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op rmsnorm \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --warmup-iters 25 --repeat-iters 100
# optional: --stages model_config,synthetic   --limit N   --seed 1234
```

The runner emits `benchmark_results.jsonl` + `benchmark_results.csv`. Each row
carries `correct` + `correctness_error`, so correctness is recorded inline.
For a no-timing correctness gate, run:

```bash
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.correctness_runner \
  --op rmsnorm \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --seed 1234
```

That optional pass emits `correctness_results.jsonl`.

### Generate the reports (CPU-only, from ledger + results)

```bash
# headline benchmark_summary.md (kernel-only vs best, stage/model splits,
# top wins/regressions + classification, eager/host-overhead, decision)
python -m benchmarks.reports.summarize_results \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl \
  --out benchmarks/examples/rmsnorm/benchmark_summary.md --kernel rmsnorm

# per-shape x per-provider status grid
python -m benchmarks.reports.coverage_matrix \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl \
  --out benchmarks/examples/rmsnorm/coverage_matrix.md

# both at once
python -m benchmarks.reports.render_markdown_report \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl \
  --out benchmarks/examples/rmsnorm --kernel rmsnorm

# weighted vs unweighted aggregates (prints n/a until a serving trace adds weights)
python -m benchmarks.reports.weighted_summary \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl
```

### Profiler gate (`benchmarks/runners/profiler_runner.py`)

When a hot shape's kernel-only `speedup_vs_best < 0.90`, run the profiler gate
for the candidate provider or record why it is deferred:

```bash
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.profiler_runner \
  --op rmsnorm --shape-id sha1:... --provider flydsl \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --out benchmarks/examples/rmsnorm/profiles
```

This runner performs a rocprofv3 `--kernel-trace` diagnosis and writes
`profiles/<shape-id-without-sha1>/<provider>/diagnosis.json` plus the repro and
rocprof stdout/stderr. It does not replace a full decoded ATT bundle; if the
diagnosis needs instruction-level stalls/source mapping, follow the repo
top-level `AGENTS.md` ATT workflow.

## Bottleneck classification (`benchmarks/reports/classify_bottleneck.py`)

Classification judges the KERNEL-ONLY speedup vs best baseline (host launch
overhead is a separate eager-only verdict). Categories (exact strings):

- `ok` — kernel-only vs-best >= 0.95 (parity or better).
- `implementation_gap` — a structural kernel issue at this shape class, e.g.
  small `M` (grid=(M,1,1) -> one workgroup/row, GPU under-occupied) or `N`
  misses the fast-vectorized path and runs the generic scalar path.
- `tuning_gap` — aligned large-M shape, sub-parity, no structural cause; fixed
  FlyDSL schedule vs a tuned baseline. Fix: per-shape tuned schedule; confirm
  with rocprofv3.
- `algorithm_gap` — the chosen algorithm cannot reach parity (reserved; use when
  a structurally different algorithm is required).
- `flydsl_codegen_gap` — codegen produced bad/illegal code (e.g. block size >
  AMDGPU `max_flat_workgroup_size`); aligned shape still slow with profiler
  evidence of waits/spills/poor vectorization.
- `launch_or_roofline_limited` — kernel is at the memory/compute roofline, or
  (eager verdict) the `@flyc.jit` launcher's per-call host overhead dominates
  short/decode shapes.
- `measurement_issue` — unstable timing (p90/p10 > 1.2) or no comparable
  baseline measured; re-measure before trusting.
- `baseline_unfair_or_unmatched` — FlyDSL failed correctness (don't trust its
  timing), or the baseline isn't a like-for-like comparison.

## Profiler trigger rule

A shape is a profiler candidate iff it is HOT (in the model/serving set, not
purely synthetic) and its kernel-only `speedup_vs_best < 0.90`. Unstable timing
(`stable == False`, i.e. p90/p10 > 1.2) is re-measured, not profiled. The
profiler gate corroborates launch/grid/kernel-trace facts; escalate to the
top-level ATT workflow before making ISA-level stall/source-mapping claims such
as a confirmed `flydsl_codegen_gap` on an aligned shape.

## Agent implementation contract (ordering)

Do these in order; do not skip a step or reorder.

1. **Identify the op.** Map the kernel to an `op_type` and confirm an `Op` is
   registered in `benchmarks/ops.py` (inputs + fp32 reference + roofline). If
   not, add the `Op` first — providers cannot build their own inputs.
2. **Ensure the shape ledger.** Run the importers so
   `benchmarks/examples/<op>/shape_ledger.jsonl` exists and covers model-config +
   synthetic-boundary + the diagnostic shape (+ serving when available).
   Importers upsert by `source.kind`, so re-running one never clobbers another.
3. **Ensure the baseline matrix.** `benchmarks/examples/<op>/baseline_matrix.yaml` lists
   every provider with an honest `enabled`/`skip_reason`. Disabled providers
   stay listed so coverage shows them as `not_configured`.
4. **Correctness.** Confirm the candidate (and each baseline) passes the fp32
   reference. An `incorrect` candidate means timing is untrustworthy — stop and
   fix before claiming any speedup.
5. **Benchmark.** Run the multishape runner via `benchmarks/bench`. Inspect the
   stderr stream for `failed`/`oom`/`unsupported` rows.
6. **Speedups.** Report `speedup = baseline_median / flydsl_median` (>1 =>
   FlyDSL faster) vs EACH baseline and vs best-available, plus unweighted +
   weighted geomean, per-stage and per-model splits. Use the PRIMARY
   (kernel-only) median.
7. **Coverage.** Render `coverage_matrix.md` — verify nothing silently vanished.
8. **Summary.** Render `benchmark_summary.md`.
9. **Profiler gate.** For every HOT shape with kernel-only `speedup_vs_best <
   0.90`, run `benchmarks.runners.profiler_runner` for a rocprofv3 kernel-trace
   diagnosis (or record why deferred). Escalate to the top-level ATT workflow
   only when instruction-level stalls/source mapping are needed.
10. **Classify.** Assign each sub-parity hot shape a category from the list
    above, with evidence + likely fix.
11. **Decision.** Emit one of `promote | promote_with_guardrails | tune_needed |
    rewrite_needed | blocked` (the implemented `decide()` vocabulary). Tie the
    decision to the data: never claim production perf from synthetic/diagnostic
    shapes alone, never compare vs opaque `aiter` without labeling the backend
    branch.

## See also

- `benchmarks/README.md` — directory tree, env recipe, methodology, worked example.
- `benchmarks/shape_ledgers/README.md` — shape sources + importer CLIs + upsert semantics.
- Repo top-level `AGENTS.md` — the ATT/rocprofv3 diagnostic layer this sits beside.
