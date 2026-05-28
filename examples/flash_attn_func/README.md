# Flash Attention Func — rocprofv3 trace bundle

Kernel: `flash_attn_func_kernel_0` (FlyDSL local snapshot
`18c5a7ed79cc5bece508896a95270f1dadb7859b`, gfx950 / MI350X).

## What's in here

```
bundle/
├── REPORT.md              ← ping-pong / stall analysis (read first)
├── README.md              ← this file
├── att_viewer/
│   ├── small/             ← B=1 S=512 H=8 D=128 bf16 causal, 32 CTAs
│   │   └── ui_output_agent_7267_dispatch_42/
│   └── big/               ← B=1 S=2048 H=32 D=128 bf16 causal, 512 CTAs
│       └── ui_output_agent_25171_dispatch_44/
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
# Open http://localhost:8080/ in a browser, then click ui_output_agent_25171_dispatch_44/
```

Use `big/` for the primary trace. It has `B=1 S=2048 H=32 D=128`,
`BLOCK_M=128`, 512 CTAs, 32 sampled wave JSON files, and 512 occupancy rows
across the sampled shader engines. Use `small/` only as the underfilled
diagnostic contrast.

`dispatch_<N>` is rocprofv3's process-wide dispatch counter, not the benchmark
iteration index. This capture also produced a second non-empty scratch
dispatch for the big run; this bundle keeps `dispatch_44` as the later
steady-state sample.

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
ROCR_VISIBLE_DEVICES=0 FLYDSL_FLASH_ATTN_FUNC_USE_CUSTOM_LLVM=0 \
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_flash_attn_func.py \
        --batch 1 --seq_len 2048 --num_heads 32 --head_dim 128 \
        --dtype bf16 --causal --warmup 3 --iters 12
```

No-profiler baseline for the primary shape was `PASS`, max error `3.91e-03`,
min cosine `0.99999`, about `92.4 us` and `371.7 TFLOPS`. Ignore the traced
run's printed microbenchmark time; rocprofv3 changes the timing path.

## Source mapping note

`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` was set, but `code.json` still has 0 / 2070
instructions mapped to Python source. The report therefore maps hot PCs to
source regions manually from the copied source.
