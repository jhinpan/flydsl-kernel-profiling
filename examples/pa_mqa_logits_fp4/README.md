# FP4 MQA Logits — rocprofv3 trace bundle

Kernel: `pa_mqa_logits_fp4_kernel_0` (FlyDSL commit `9120078`, gfx950 / MI350X).

## What's in here

```
bundle/
├── REPORT.md              ← optimization writeup (read first)
├── README.md              ← this file
├── att_viewer/            ← feed these to AMD ATT Viewer
│   ├── small/             ← batch=4 ctx=8K (1 chunk/CTA, prologue-bound)
│   │   └── ui_output_agent_32435_dispatch_197/
│   └── big/               ← batch=32 ctx=128K (17 chunks/CTA, 507/512 CTAs)
│       └── ui_output_agent_55351_dispatch_426/
├── compute_viewer/        ← feed these to rocprof-compute-viewer
│   ├── small_results.json     (rocprofv3 full event stream)
│   ├── small_agent_info.csv   (GPU/agent metadata)
│   ├── big_results.json
│   ├── big_agent_info.csv
│   └── discover_*.csv         (kernel-level stats from --stats run)
└── source/                ← kernel + harness + rocprofv3 config (for src↔ISA mapping)
    ├── pa_mqa_logits_fp4.py
    ├── test_pa_mqa_logits_fp4.py
    ├── input_trace.yaml         (config that produced att_viewer/small)
    └── input_trace_big.yaml     (config that produced att_viewer/big)
```

## How to open

### ATT Viewer (instruction-level, what you want for ISA inspection)

Each `ui_output_agent_<PID>_dispatch_<N>/` directory is a self-contained trace.
The viewer is served as static HTML, so:

```bash
cd bundle/att_viewer/big
python3 -m http.server 8080
# Open http://localhost:8080/ in a browser → click into a ui_output_agent_* dir
```

Pick `big/` for the primary saturation-validated trace. It uses
`safe_chunks_per_cta=17` and `total_ctas=507` against the default
`parallel_unit_num=512`; the sampled CU timelines do not show an obvious
underfilled tail. Pick `small/` to see the cold-prologue tail.

If the ATT Viewer columns/tabs are unfamiliar, read the repo-level
[`docs/att-viewer-guide.md`](../../docs/att-viewer-guide.md). In particular:
`Compute Unit` is the wave timeline, `Utilization` is the hardware-pipeline
timeline, arrows are dependency/attribution links, and high-hitcount
`s_waitcnt` hotspots are the first places to look for missed overlap.

### Naming convention: `ui_output_agent_<PID>_dispatch_<N>`

`<N>` is rocprofv3's process-wide dispatch counter (covering torch utility
kernels too), not the iteration index inside our test harness. Only one
ATT-captured kernel run is kept per `small/` and `big/`. A `dispatch_N` folder
with no `se*_sm*_*.json` files is an empty placeholder (rocprofv3 reserves the
slot before ATT collection begins) — safe to delete if you re-capture.

Files inside each `ui_output_agent_*/`:
| file | content |
|---|---|
| code.json | ISA disassembly + per-PC (hit, latency, stall, idle) |
| filenames.json | source file paths (currently 0 % mapped — see "source mapping" below) |
| occupancy.json | wave start/stop timeline per (CU, SIMD) |
| realtime.json | gfx_clock ↔ realtime_clock conversion |
| se{N}_sm{M}_sl{L}_wv{W}.json | per-wave timeline (EXEC/STALL/WAIT durations) |
| wstates*.json | extended stall-reason samples |

### rocprof-compute-viewer (aggregate counters / roofline / timeline)

```bash
# Install if not present:
pip install rocprof-compute-viewer   # or: python -m pip install rocm-compute-viewer

# Open one of the results.json files:
rocprof-compute-viewer bundle/compute_viewer/big_results.json
```

`big_results.json` is the full rocprofv3 output for the production-like larger
run (~27 MB);
`small_results.json` is the same for the small run.
The `discover_*.csv` files come from the `rocprofv3 --stats` discovery pass and
include per-kernel call counts / durations across the whole test run, not just
the one ATT-captured iteration.

## Workload sizing note

This example intentionally keeps both a small prologue-oriented trace and a
primary saturation-validated trace. The current `big` shape reports
`safe_chunks_per_cta=17` and `total_ctas=507`, close to the default
`parallel_unit_num=512` target. The kept ATT dispatch has 32 wave JSON files and
512 occupancy rows across the sampled SE/SIMD slots, with no obvious sampled-CU
tail.

## Source mapping note (important)

Our capture set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` to enable DWARF line-tables,
but `code.json` ended up with **0 / 727** instructions source-mapped.
That means the viewer will show ISA but won't jump back to the Python lines.
Likely cause: FlyDSL's JIT pipeline stripped the line table before producing
the hsaco. To fix on a fresh capture, verify on the source machine that:

1. `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is actually exported into the python process
2. The JIT artifact under `/tmp/flydsl_*` (or `KeepBuildTmp`'s output) has a
   non-empty `.debug_line` section: `llvm-dwarfdump --debug-line <hsaco> | head`
3. `filenames.json` lists the kernel source path

With this fixed the ATT Viewer will let you click an ISA line and jump to e.g.
`pa_mqa_logits_fp4.py:545` (the `relu * w` step).

## Reproducing on the source machine

```bash
cd /sgl-workspace/jin/fp4_mqa_probe
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_pa_mqa_logits_fp4.py \
        --batch 32 --ctx 131072 --num_iters 12 --num_warmup 3
```
