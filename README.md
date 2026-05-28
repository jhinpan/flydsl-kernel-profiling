# FlyDSL Kernel Profiling

A library of FlyDSL GPU kernels paired with **decoded rocprofv3 trace bundles**
and **analysis reports**. Each example under [`examples/`](examples) lets anyone
on any machine (no GPU required) load the kernel into the AMD **ATT Viewer**
for instruction-level inspection or **rocprof-compute-viewer** for counter
aggregates — without re-capturing the trace.

## What the bundle gives you

For one specific operator, every example contains all four of:

1. **The kernel source** — the exact `.py` + test harness from the FlyDSL commit
2. **A full rocprofv3 ATT capture** at diagnostic workload shapes, with the
   primary trace validated against grid size and ATT tail checks
3. **The rocprofv3 results.json** with PMC counter samples
4. **A written report** — headline wave-state breakdown, top-N hotspot
   instructions with disassembly context, PC → Python source mapping, and a
   priority-ranked list of optimization candidates

Mission: **everything you need to understand or improve one kernel, in one
self-contained directory**.

## Examples

| folder | kernel | source | headline |
|---|---|---|---|
| [`examples/pa_mqa_logits_fp4`](examples/pa_mqa_logits_fp4) | FP4 MQA Logits | [`ROCm/FlyDSL@9120078`](https://github.com/ROCm/FlyDSL/commit/9120078d35d7d232b3941ded5b76a1ca92329ef0) | 1189.9 TFLOPS at batch=32 ctx=128K with `total_CTAs=507/512`; stall-bound (35 % `vmcnt`, 22 % `lgkmcnt`, only 0.1 % EXEC); 5 waves/SIMD, 11 VGPRs from 6 waves/SIMD |
| [`examples/flash_attn_func`](examples/flash_attn_func) | Flash Attention Func | `FlyDSL-lab@18c5a7e` | 371.7 TFLOPS at B=1 S=2048 H=32 D=128 with `total_CTAs=512/512`; ping-pong exists but consume points still stall (28 % `vmcnt`, 16 % `lgkmcnt`, 58.5 % stall ratio) |

## How to use a captured trace

```bash
git clone https://github.com/jhinpan/flydsl-kernel-profiling
cd flydsl-kernel-profiling/examples/<kernel>

# 1. Read the analysis writeup
$EDITOR REPORT.md

# 2. ATT Viewer (instruction-level): serve the primary trace folder over HTTP
# Read REPORT.md first; examples often use att_viewer/big for the larger trace.
cd att_viewer/big
python3 -m http.server 8080
# open http://<host>:8080/ → click into ui_output_agent_*

# 3. rocprof-compute-viewer (counter aggregates)
pip install rocprof-compute-viewer   # if not installed
rocprof-compute-viewer ../compute_viewer/big_results.json

# 4. Re-run the analysis script on the trace (no GPU needed)
python source/hotspot_analyzer.py att_viewer/big/ui_output_agent_*/ --topk 15 --mode both
```

If the ATT Viewer tabs are unfamiliar, read
[`docs/att-viewer-guide.md`](docs/att-viewer-guide.md). It explains how to use
`Compute Unit`, `Utilization`, dependency arrows, `s_waitcnt`, `hitcount`,
`idle`, `VALU`, `LDS`, and VGPR/occupancy when ranking hotspots.

## Adding a new example

**Read [`AGENTS.md`](AGENTS.md) first.** It codifies the entire workflow:
environment setup, workload/grid sizing, trace capture (`small` / `big` as
diagnostic labels, not automatic proof of saturation), cleanup, analysis with
`hotspot_analyzer.py`, report template, the per-example directory layout, and
the gotchas that cost us time to figure out (empty-shell folders, `dispatch_<N>`
numbering, debug-info plumbing, etc.).

Quick canonical layout:

```
examples/<kernel-name>/
├── README.md
├── REPORT.md
├── att_viewer/{small,big}/ui_output_agent_<PID>_dispatch_<N>/
├── compute_viewer/{small,big}_results.json + agent_info.csv + discover_*.csv
└── source/<kernel>.py + test_<kernel>.py + input_trace*.yaml + hotspot_analyzer.py
```

## Why a separate repo

Capturing an ATT trace needs:
- gfx950 (MI300X / MI350X) hardware
- `rocprofv3` v1.1+ and `librocprof-trace-decoder.so` correctly installed
- the matching FlyDSL build
- 3–5 minutes per kernel for JIT + capture

Shipping the decoded `ui_output_agent_*` folders means anyone can do
instruction-level perf analysis on a laptop.

## Toolchain notes

- **rocprofv3** — bundled with ROCm 6.4+ as `rocprofiler-sdk`. Required: v1.1.0+.
- **rocprof-trace-decoder** — `librocprof-trace-decoder.so` must be in
  `/opt/rocm/lib`. If missing: locate via `find / -name 'librocprof*.so'` and
  copy into place.
- **AMD ATT Viewer** — currently served as static HTML; any HTTP server pointed
  at the `ui_output_agent_*` parent directory works.
- **rocprof-compute-viewer** — `pip install rocprof-compute-viewer` (formerly
  Omniperf).

For the canonical capture recipe see FlyDSL's
[`.claude/skills/capture-kernel-trace/SKILL.md`](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/capture-kernel-trace/SKILL.md);
for the analysis recipe see
[`.claude/skills/kernel-trace-analysis/SKILL.md`](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/kernel-trace-analysis/SKILL.md).
This repo's [`AGENTS.md`](AGENTS.md) layers on top of those: it pins the
output structure, fills in the gotchas the skills don't cover, and specifies
what goes into the per-example REPORT.md.

## License

Kernel sources under `examples/*/source/` derive from FlyDSL (Apache-2.0).
Trace artifacts and analysis are released under the same license.
