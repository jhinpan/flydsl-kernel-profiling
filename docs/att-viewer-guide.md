# ATT Viewer / rocprof-compute-viewer Guide

This repo stores decoded rocprofv3 ATT traces so we can inspect kernels without
re-capturing them. The viewer is powerful, but the tabs answer different
questions. Use this page as the reading order before turning a trace into an
optimization plan.

## Reading Order

Start with the workload shape before interpreting any hotspot:

1. Confirm the captured kernel is the intended dispatch.
2. Confirm the grid is large enough to exercise the GPU. For persistent kernels,
   read the test output for `safe_chunks_per_cta` and `total_ctas`.
3. Open `Compute Unit` to see whether the sampled CU/SIMDs have long empty tails
   or synchronized stalls.
4. Open `Utilization` to see which hardware pipelines are busy while waves are
   resident.
5. Open `Hotspot` and sort by `Latency: Sum all`, then use `hitcount`, `stall`,
   and `idle` to rank real optimization targets.
6. Cross-check with `source/hotspot_analyzer.py`; use its stall taxonomy and
   VGPR/occupancy estimate in `REPORT.md`.

## Compute Unit vs. Utilization

`Compute Unit` is a wave timeline. It shows what waves on the sampled CU/SIMDs
are doing over time: executing, waiting, stalling, sleeping, or leaving gaps.
Use it to answer:

- Are enough waves resident on the CU?
- Do many waves stall at the same time?
- Is there a long tail where only a few waves remain?
- Does the workload end before the CU has enough steady-state work?

`Utilization` is a pipeline timeline. It shows whether hardware units such as
`VALU`, `VMEM`, and `LDS` are busy. Use it to answer:

- Is the kernel feeding vector ALU, memory, or LDS units?
- Are waves resident but pipelines mostly idle?
- Are VMEM/LDS bursts overlapped with VALU/MFMA work, or are they serialized?

The two tabs should be read together: a CU can have waves resident while the
useful pipeline utilization is low because those waves are blocked on
dependencies.

## Instruction Arrows

The colored arrows in the instruction views are dependency/attribution links,
not source-level control flow. Around a stall, they usually show which earlier
instruction produced the value or outstanding memory operation that the current
instruction is waiting for.

This is especially useful near `s_waitcnt`: the link often points back to the
`buffer_load`, `global_load`, `s_load`, or LDS operation whose completion is
blocking progress.

## Why `s_waitcnt` Matters

On AMD GPUs, many memory operations are issued asynchronously. A later
`s_waitcnt` instruction makes the wave wait until selected outstanding
operations have drained far enough.

Common forms:

- `s_waitcnt vmcnt(N)` waits for vector/global-memory operations to drain to at
  most `N` outstanding operations.
- `s_waitcnt lgkmcnt(N)` waits for LDS/scalar-memory/local-memory-related
  operations to drain to at most `N` outstanding operations.
- `s_waitcnt expcnt(N)` waits for export operations.

Treat `s_waitcnt` as a per-wave dependency fence, not a whole-workgroup barrier.
It is still one of the best optimization signals because a large wait usually
means a load was issued too late, the wait was placed too early, or there was
not enough independent work between producer and consumer to hide latency.

## Bubbles, `s_waitcnt`, and `s_nop`

Large blank/bubble regions in the wave timeline often line up with explicit
dependency drains. Start by checking hot `s_waitcnt` instructions and their
dependency arrows, because those waits usually identify the producer/consumer
edge that failed to hide memory or LDS latency.

`s_nop` is different from a barrier: it is compiler-inserted padding or a
scheduling delay, not a synchronization primitive. Still treat hot `s_nop`
clusters as useful evidence. They often mark a spot where the compiler could not
find independent work to cover an outstanding dependency, so the optimization
target is usually the surrounding schedule: issue the producer earlier, move
independent work between the producer and consumer, or reduce the hard wait
boundary.

For ping-pong / double-buffered kernels, the pattern to look for is:

```text
buffer_load ... lds   # issue async DMA-to-LDS for the next tile
s_waitcnt ...         # consume-side drain; hot here means prefetch distance was insufficient
s_barrier             # workgroup visibility before LDS reuse/consume
ds_read_*             # read current tile from LDS
v_mfma_*              # consume the tile
```

If `s_waitcnt` or `s_nop` dominates the top PCs on a steady-state path, the
primary recommendation should usually be to improve overlap or reduce hard
producer/consumer boundaries, not to tune MFMA issue first.

For this repo's first FP4 MQA trace, the production-shape trace showed:

- `LDS/SMEM-wait` from `lgkmcnt`: the largest stall bucket.
- `VMEM-wait` from `vmcnt`: the next important memory-wait bucket.
- VALU latency that is largely dependency latency after MFMA/post-processing,
  not evidence that the kernel is compute-bound.

## Hitcount, Stall, Latency, and Idle

Do not rank hotspots by a single column. Use the columns together:

- `hitcount`: how often the static instruction appears in sampled execution.
  High hitcount makes a hotspot more representative of the steady-state path.
- `latency`: total time attributed to the instruction across sampled waves.
  Sorting by `Latency: Sum all` is a good first pass.
- `stall`: cycles where the instruction could not retire because it was waiting
  on something. High stall is usually more actionable than high latency alone.
- `idle`: issue/pipeline idle time near that instruction. Use it as a hint that
  the scheduler had no ready work, but confirm with dependency arrows and stall
  taxonomy.

A useful optimization target is usually high-hitcount, high-latency, high-stall,
and on a steady-state path. Low-hitcount prologue stalls may still matter for
short workloads, but they should not dominate decisions for production shapes.

## Instruction Classes

Viewer legends group instructions by hardware path:

- `VALU`: vector ALU instructions such as `v_add`, `v_mul`, `v_bfe`, and
  `v_maximum`. High VALU time can be real arithmetic, but can also include
  dependency latency after MFMA results.
- `MFMA` / `MATRIX`: matrix instructions. These are the useful tensor-core-style
  work for GEMM-like kernels.
- `VMEM`: vector/global-memory loads and stores such as `buffer_load` and
  `global_load`.
- `SMEM` / `SALU`: scalar loads and scalar ALU work. Kernel argument loads often
  show up here and retire through `lgkmcnt`.
- `LDS`: local data share, AMD's on-chip workgroup scratchpad, similar in role to
  CUDA shared memory.
- `JUMP`, `NEXT`, `IMMED`, `MSG`: control/metadata categories used by the
  viewer; inspect them only when they become visible hotspots.

## Where VGPR Shows Up

VGPR is register pressure, not a pipeline, so it usually does not appear in the
same way as `VALU`, `VMEM`, or `LDS` in the utilization legend.

Find VGPR pressure from:

- rocprof kernel trace metadata, such as `VGPR_Count` and `Accum_VGPR_Count`.
- the `Occupancy` view.
- `source/hotspot_analyzer.py`, which scans the ISA and reports estimated
  `arch_vgpr`, `accum_vgpr`, current `waves/SIMD`, and the next occupancy step.

For gfx950/CDNA4, arch VGPRs and accumulator VGPRs share a combined pool in the
analyzer's model. A small VGPR reduction can matter if it crosses an occupancy
threshold, for example from 5 to 6 waves/SIMD.

## Workload Sizing Before Capture

Small traces are useful for cold-start/prologue analysis, but they can be
misleading if the grid is too small to fill the GPU. For persistent kernels,
calculate the grid before deciding that a trace is representative.

The FP4 MQA scheduler uses:

```text
chunks_per_batch = ceil(context_len / block_k)
safe_chunks_per_cta = smallest s such that:
    sum(ceil(chunks_per_batch / s)) * next_n <= parallel_unit_num
total_ctas = sum(ceil(chunks_per_batch / safe_chunks_per_cta)) * next_n
```

Then validate the capture:

- `total_ctas` should be close to the target parallelism. With the default
  `parallel_unit_num=512`, prefer a shape that lands near 512 CTAs, not 69 or
  391, when the goal is steady-state analysis.
- `Compute Unit` should not show a long tail where only a few waves remain.
- `Utilization` should not show the kernel ending before pipelines settle into a
  representative pattern.

If `total_ctas` is too low, increase `batch`, increase effective `ctx`, adjust
`parallel_unit_num`, or choose a workload shape with more tiles/chunks. Keep a
small/prologue trace if useful, but add a saturated trace for optimization
ranking.

Always record `block_k`, `safe_chunks_per_cta`, `total_ctas`, and the target
`parallel_unit_num` in the example's `REPORT.md`.
