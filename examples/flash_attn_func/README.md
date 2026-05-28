# Flash Attention Func — rocprofv3 trace bundle

Kernel: `flash_attn_func_kernel_0` (FlyDSL local snapshot
`18c5a7ed79cc5bece508896a95270f1dadb7859b`, gfx950 / MI350X).

## What's in here

```
bundle/
├── REPORT.md              ← ping-pong / stall analysis (read first)
├── README.md              ← this file
├── att_viewer/
│   ├── small/             ← B=1 S=1024 H=16 D=128 bf16 causal, 128 CTAs
│   │   └── ui_output_agent_32235_dispatch_44/
│   └── big/               ← B=1 S=2048 H=32 D=128 bf16 causal, 512 CTAs
│       └── ui_output_agent_38430_dispatch_44/
├── compute_viewer/
│   ├── small_results.json
│   ├── small_agent_info.csv
│   ├── big_results.json
│   ├── big_agent_info.csv
│   └── discover_*.csv
└── source/
    ├── flash_attn_func.py
    ├── kernels_common.py
    ├── test_flash_attn_func.py
    ├── input_trace.yaml
    ├── input_trace_big.yaml
    └── hotspot_analyzer.py
```

## How to open

### ATT Viewer

```bash
cd bundle/att_viewer/big
python3 -m http.server 8080
# Open http://localhost:8080/ in a browser, then click ui_output_agent_38430_dispatch_44/
```

Use `big/` for the primary trace. It has `B=1 S=2048 H=32 D=128`,
`BLOCK_M=128`, 512 CTAs, 32 sampled wave JSON files, 512 occupancy rows across
the sampled shader engines, and 2069 / 2070 ISA rows with Python source
mapping. Use `small/` only as the underfilled diagnostic contrast; it now uses
128 CTAs so the sampled CU reliably has waves while still remaining
underfilled.

`dispatch_<N>` is rocprofv3's process-wide dispatch counter, not the benchmark
iteration index. The cold-debug capture produced two non-empty samples for each
shape; this bundle keeps `dispatch_44` as the later steady-state sample.

### rocprof-compute-viewer

```bash
rocprof-compute-viewer bundle/compute_viewer/big_results.json
```

The discovery CSV reports the primary kernel launch as
`Workgroup_Size_X=256`, `Grid_Size_X=131072`, `VGPR_Count=124`,
`SGPR_Count=112`, and `LDS_Block_Size=49152`. `Grid_Size_X` is in threads, so
the primary workload has `131072 / 256 = 512` CTAs.

## Reproducing on the source machine

```bash
cd /sgl-workspace/jin/flash_attn_func_probe
rm -rf .flydsl_trace_cache_cold_debug prof/att_big
ROCR_VISIBLE_DEVICES=0 FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM=0 \
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 \
FLYDSL_RUNTIME_CACHE_DIR=$PWD/.flydsl_trace_cache_cold_debug \
PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_flash_attn_func.py \
        --batch 1 --seq_len 2048 --num_heads 32 --head_dim 128 \
        --dtype bf16 --causal --warmup 3 --iters 12
```

No-profiler baseline for the primary shape was `PASS`, max error `3.91e-03`,
min cosine `0.99999`, about `92.4 us` and `371.7 TFLOPS`. Ignore the traced
run's printed microbenchmark time; rocprofv3 changes the timing path.

## Source mapping note

This bundle was recaptured from a fresh `FLYDSL_RUNTIME_CACHE_DIR` with
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` before discovery/capture. `code.json` has
2069 / 2070 instructions mapped to Python source, and the FlyDSL code object
contains a non-empty `.debug_line` section. Some hot waits still aggregate to
the kernel decorator or MFMA wrapper line; the report keeps manual source-region
correlation for precise phase labels.
