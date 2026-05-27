# FlyDSL Kernel Profiling

Instruction-level (ATT) and counter-level traces of FlyDSL GPU kernels on AMD ROCm,
captured with `rocprofv3` on gfx950 / MI350X, ready to load into the AMD **ATT Viewer**
and **rocprof-compute-viewer** without re-running the kernel.

The point: ship the *decoded trace artifacts* — not just analysis text — so anyone
with a viewer can inspect any PC, any wave, any stall reason for themselves.

## Examples

| folder | kernel | source | one-liner |
|---|---|---|---|
| [`examples/pa_mqa_logits_fp4`](examples/pa_mqa_logits_fp4) | FP4 MQA Logits | [`ROCm/FlyDSL@9120078`](https://github.com/ROCm/FlyDSL/commit/9120078d35d7d232b3941ded5b76a1ca92329ef0) | Q-FP4 × KV-FP4 attention logits with pipelined packed-KVS prefetch; 605 TFLOPS at batch=8 ctx=64K; profoundly stall-bound (47 % VALU latency, 34 % `s_waitcnt`, only 0.3 % EXEC) |

## How to use a captured trace

```bash
git clone https://github.com/jhinpan/flydsl-kernel-profiling
cd flydsl-kernel-profiling/examples/<kernel>

# Instruction-level (ATT Viewer)
cd att_viewer/big
python3 -m http.server 8080
# open http://<host>:8080/ → click into ui_output_agent_*

# Counter-level (rocprof-compute-viewer)
rocprof-compute-viewer compute_viewer/big_results.json
```

Each example's `REPORT.md` contains the analysis writeup (hot PCs, source-line
mapping, optimization candidates ranked by expected impact). Each example's
`README.md` documents file layout and reproduction commands.

## Why a separate repo

Capturing an ATT trace needs:
- gfx950 (MI300X/MI350X) hardware
- rocprofv3 v1.1+ and `librocprof-trace-decoder.so` correctly installed
- the matching FlyDSL build with debug info wired through the JIT pipeline
- 3–5 minutes per kernel for compile + capture

Shipping the decoded `ui_output_agent_*` folders means anyone on any machine
(no GPU required for viewing) can inspect ISA-level perf data.

## Adding a new example

```
examples/<kernel-name>/
├── README.md         ← what this kernel does, file layout, repro command
├── REPORT.md         ← analysis: hot PCs, stalls, optimization candidates
├── att_viewer/       ← ui_output_agent_<PID>_dispatch_<N>/ — for ATT Viewer
├── compute_viewer/   ← rocprofv3 results.json + agent_info.csv
└── source/           ← kernel .py + test harness + input_trace*.yaml
```

The yaml in `source/` is the rocprofv3 config that produced the trace — keep
it alongside the output so the capture is reproducible.

## Toolchain notes

- **rocprofv3**: `apt install rocprofiler-sdk` (ROCm 6.4+ ships v1.1+)
- **rocprof-trace-decoder**: `librocprof-trace-decoder.so` must be in
  `/opt/rocm/lib`. If `rocprofv3 -i input.yaml` says "rocprof-trace-decoder
  library path not found", locate it under any rocm install and symlink it in.
- **ATT Viewer**: shipped with rocm-developer-tools / rocprof-trace-decoder-ui;
  alternatively any static HTTP server pointed at the `ui_output_agent_*` parent
  directory will work.
- **rocprof-compute-viewer**: `pip install rocprof-compute-viewer`
  (formerly Omniperf).

## License

Kernel sources under `examples/*/source/` are derived from FlyDSL (Apache-2.0).
Trace artifacts and analysis are released under the same license.
