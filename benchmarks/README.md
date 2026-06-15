# `benchmarks/` — FlyDSL multi-shape kernel benchmark

A multi-shape benchmark **layer** that lives beside the diagnostic rocprofv3/ATT
trace bundles in `examples/<kernel>/`. The ATT bundles answer "where does this
one kernel stall on one diagnostic shape." This layer answers a different
question: **across the shapes that actually occur in models and serving, is
FlyDSL's kernel faster than the field — and if not, why?**

The two layers are keyed by the same op names but land in different trees. ATT
work owns top-level `examples/<op>/` (`REPORT.md`, per-example `README.md`,
`att_viewer/`, `compute_viewer/`, `source/`). This benchmark layer owns
`benchmarks/examples/<op>/` (`shape_ledger.jsonl`, `baseline_matrix.yaml`,
`benchmark_results.{jsonl,csv}`, `coverage_matrix.md`,
`benchmark_summary.md`, and optional tier artifacts). New benchmark files
**never clobber** the ATT bundle's `REPORT.md` or `README.md`.

## Directory tree

```
benchmarks/
├── README.md                  ← this file
├── env.sh                     ← source before any GPU runner (build-tree PYTHONPATH/LD)
├── bench                      ← wrapper: sources env.sh then `exec python "$@"`
├── common.py                  ← spine: stable_shape_id, read/write_jsonl, provenance,
│                                timing (eager + cudagraph), measure_both, speedup, geomean
├── ops.py                     ← per-op inputs + fp32 reference + roofline (one Op per op_type)
├── validate.py                ← validate rows against schemas/ (jsonschema or fallback)
├── schemas/
│   ├── shape_ledger.schema.json
│   ├── benchmark_result.schema.json
│   └── baseline_matrix.schema.yaml
├── shape_ledgers/             ← importers that build shape_ledger.jsonl (CPU-only)
│   ├── README.md              ← shape sources + per-importer CLI + upsert semantics
│   ├── ledger_io.py           ← idempotent upsert_ledger (dedup by shape_id, replace by kind)
│   ├── aiter_model_shapes_importer.py
│   └── manual_shape_importer.py   (synthetic-boundary + diagnostic + manual)
├── providers/                 ← one adapter per (provider, op_type)
│   ├── base.py                ← ProviderAdapter contract + load_entrypoint
│   ├── flydsl.py / pytorch.py / aiter.py / aiter_triton.py / triton.py
│   │                            ← shared rmsnorm adapters
│   ├── <op>.py                  ← op-specific candidate/baseline adapters
│   └── aiter_ck.py / aiter_asm.py / ck.py / gluon.py / hipblaslt.py
│                                ← backend-specific adapters or honest support stubs
├── runners/
│   ├── correctness_runner.py  ← correctness-only rows, no timing
│   ├── multishape_runner.py   ← orchestrator: inputs once/shape, ref, every provider
│   ├── profiler_runner.py     ← rocprofv3 kernel-trace gate for hot sub-parity shapes
│   └── regression_runner.py   ← current-vs-previous latency + hot-shape guard
├── reports/                   ← CPU-only, from ledger + results
│   ├── analysis.py            ← join + per-shape speedups + best-baseline + aggregates
│   ├── summarize_results.py   ← benchmark_summary.md (headline + splits + decision)
│   ├── coverage_matrix.py     ← coverage_matrix.md (per-shape × per-provider status)
│   ├── render_markdown_report.py  ← render both at once
│   ├── weighted_summary.py    ← weighted vs unweighted aggregate
│   └── classify_bottleneck.py ← rule-based gap classification
└── examples/<op>/             ← benchmark artifacts land here
```

## `env.sh` — the recipe and WHY

```bash
# (env.sh, verbatim intent)
export FLYDSL_LAB=/sgl-workspace/FlyDSL-lab
export PYTHONPATH="$FLYDSL_LAB/build-fly/python_packages:$FLYDSL_LAB:<repo>:$PYTHONPATH"
export LD_LIBRARY_PATH="$FLYDSL_LAB/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH"
export SGLANG_USE_AITER=0
```

**Always launch GPU runners via `benchmarks/env.sh` (or the `benchmarks/bench`
wrapper).** Importers and report generators are pure-data and run on a CPU-only
box without it; anything that imports `flydsl`, `aiter`, `triton`, or `torch`
needs it.

Why it is load-bearing on this node:

- `flydsl` is pip-installed editable, but the editable tree has **no compiled
  `_mlir` extension**. The built `.so` lives in
  `FLYDSL-lab/build-fly/python_packages`. Putting that tree first on
  `PYTHONPATH` makes `import flydsl.*` resolve to the built copy.
- That same path **also unblocks `import aiter`** — `aiter/__init__` imports
  `flydsl.expr` transitively, so without the build tree `import aiter` raises
  `ModuleNotFoundError: flydsl._mlir`. This is the non-obvious part: the FlyDSL
  build tree is what makes the AITER baselines importable at all.
- The `_mlir` `.so` needs its `_mlir_libs` dir on `LD_LIBRARY_PATH`, and the
  dynamic loader reads `LD_LIBRARY_PATH` **at process exec** — so it must be set
  before python starts. `env.sh` does this; `common.bootstrap_env()` only covers
  the `PYTHONPATH` half (so imports resolve), while `common.flydsl_runtime_ok()`
  reports whether the native half actually loaded.
- `SGLANG_USE_AITER=0` lets the standalone sglang Triton kernel import without
  forcing the aiter path.

Verified node recipe: GPU **MI350X gfx950**, **ROCm 7.2**, **torch
2.9.1+rocm**, **triton 3.6**.

## Methodology — kernel-only vs eager

The PRIMARY metric is **kernel-only CUDA-graph time** (`common.benchmark_cudagraph`).
It pays host launch overhead + JIT/autotune + allocation ONCE at capture, then
replays the captured kernel under event timing - leaving pure device time. By
default the graph metric flushes L2 before each replay and records
`cache_state=l2_flushed_graph` with `graph_replay_count=1`, so memory-bound
kernels are not reported as warm-cache graph replays. Older result rows that
lack `cache_state` were produced before this distinction was recorded; use them
for relative historical context, not as cold-cache bandwidth/roofline evidence.

Set `flush_l2=False` in `common.benchmark_cudagraph` only when intentionally
reproducing the old unrolled warm-graph metric (`cache_state=warm_graph_replay`).
The L2-flushed graph metric is the fair default across providers, especially on
short shapes where Python launch overhead would otherwise dominate.

Reported separately:

- **eager event time** (`common.benchmark`, L2-flush + loop amortization), and
- **`host_overhead_us = eager_median - graph_median`** — surfaced as a
  first-class signal. FlyDSL's `@flyc.jit` launcher rebuilds its cache-key every
  call, so it has high per-call host overhead; on short/decode shapes that can
  be tens of µs. This is a **launcher (host-side)** problem, distinct from kernel
  speed, and is reported as a separate eager verdict — it is mitigated when
  serving captures decode in a CUDA/hipgraph (as SGLang does).

`common.measure_both` returns both; the result row carries `median_us` (primary),
`eager_median_us`, `graph_median_us`, `host_overhead_us`, `timing_method`,
`cache_state`, and `graph_replay_count`.

**Speedups.** `speedup = baseline_median / flydsl_median` (>1 => FlyDSL faster).
A shape's headline is its kernel-only speedup vs the **best available** correct
baseline (`speedup_vs_best`). Aggregates: **unweighted geomean** over measured
shapes, **weighted geomean** using `baseline_time_weight` (preferred) or
`traffic_weight` from the ledger, plus **per-baseline**, **per-stage**, and
**per-model** splits. Weighted numbers print `n/a` until a serving trace
populates weights — synthetic/diagnostic shapes alone never carry production
weight.

A measurement is flagged unstable (`stable=False`) when p90/p10 > 1.2; unstable
hot shapes are re-measured, not trusted.

## rmsnorm worked example (end to end)

```bash
# 0. (one-time) inspect the wired matrix
cat benchmarks/examples/rmsnorm/baseline_matrix.yaml

# 1. build the ledger (CPU-only; idempotent upsert by source.kind)
python -m benchmarks.shape_ledgers.aiter_model_shapes_importer \
  --aiter-model-shapes /sgl-workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json \
  --out benchmarks/examples --tp 8 --gpu MI350X --arch gfx950 --dtype bf16 --ops rmsnorm
python -m benchmarks.shape_ledgers.manual_shape_importer --op rmsnorm \
  --out benchmarks/examples --synthetic-boundary --diagnostic 32768,8192,bf16

# 2. run on the GPU — via the bench wrapper so the build-tree PYTHONPATH/LD
#    (which also unblocks `import aiter`) is set BEFORE python starts
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.multishape_runner \
  --op rmsnorm \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --warmup-iters 25 --repeat-iters 100
#   -> benchmark_results.jsonl + benchmark_results.csv

# 3. reports (CPU-only)
python -m benchmarks.reports.render_markdown_report \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl \
  --out benchmarks/examples/rmsnorm --kernel rmsnorm
#   -> coverage_matrix.md + benchmark_summary.md
python -m benchmarks.reports.weighted_summary \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --results benchmarks/examples/rmsnorm/benchmark_results.jsonl
```

Optional tier runners:

```bash
# correctness-only, no timing
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.correctness_runner \
  --op rmsnorm \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --baseline-matrix benchmarks/examples/rmsnorm/baseline_matrix.yaml \
  --out benchmarks/examples/rmsnorm --seed 1234

# profiler gate for a HOT sub-parity shape; writes profiles/<shape-id-without-sha1>/<provider>/
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.profiler_runner \
  --op rmsnorm --shape-id sha1:... --provider flydsl \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --out benchmarks/examples/rmsnorm/profiles

# current-vs-previous guard; exits nonzero when regressions or hot shapes remain
HIP_VISIBLE_DEVICES=7 benchmarks/bench -m benchmarks.runners.regression_runner \
  --shape-ledger benchmarks/examples/rmsnorm/shape_ledger.jsonl \
  --current benchmarks/examples/rmsnorm/benchmark_results.jsonl \
  --previous <previous-benchmark-results.jsonl> \
  --out-dir benchmarks/examples/rmsnorm
```

The runner streams `[i/n] <shape_id> M=..,N=.. <dtype> (<stage>)` per shape and
keeps every provider in the output with an explicit `benchmark_status`
(`ok | failed | oom | unsupported | incorrect | not_configured`) — nothing is
silently dropped. Correctness is recorded inline on each timing row (`correct` +
`correctness_error`). For a fast no-timing gate, run
`benchmarks.runners.correctness_runner`; it emits `correctness_results.jsonl`.

## Where artifacts land

Everything for one benchmarked kernel lands in `benchmarks/examples/<op>/`:

| File | Producer |
|---|---|
| `shape_ledger.jsonl` | importers (`shape_ledgers/*`) |
| `baseline_matrix.yaml` | authored once per kernel |
| `correctness_results.jsonl` | optional correctness-only runner (`runners/correctness_runner.py`) |
| `benchmark_results.jsonl` / `.csv` | `runners/multishape_runner.py` |
| `coverage_matrix.md` | `reports/coverage_matrix.py` |
| `benchmark_summary.md` | `reports/summarize_results.py` |
| `profiles/<shape-id-without-sha1>/<provider>/diagnosis.json` | optional profiler gate (`runners/profiler_runner.py`) |
| `regression_summary.md` / `regressions.jsonl` | optional regression runner (`runners/regression_runner.py`) |

These sit in `benchmarks/examples/<op>/`; the ATT bundle remains in top-level
`examples/<op>/` with `REPORT.md`, `README.md`, `att_viewer/`, `compute_viewer/`,
and `source/`.

## What is wired vs opt-in

- **Wired:** multi-shape ledgers and benchmark summaries for the checked-in
  kernels under `benchmarks/examples/`, the correctness-only runner, the
  multishape timing runner, the rocprofv3 kernel-trace profiler gate, the
  regression runner, and all report generators.
- **Still data-dependent / opt-in:** serving-trace importers
  (`sglang_trace_importer.py`, `atom_workload_importer.py`) populate `weight.*`
  and `source.kind in {sglang_trace, atom_workload}` when serving traces are
  available; otherwise production-weighted geomeans remain `n/a`.

## See also

- `benchmarks/shape_ledgers/README.md` — shape sources + importer CLIs + upsert.
- `.claude/skills/flydsl-kernel-multishape-benchmark/SKILL.md` — the agent contract.
- Repo top-level `AGENTS.md` — the ATT/rocprofv3 diagnostic layer this sits beside.
