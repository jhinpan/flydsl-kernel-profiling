# FlyDSL Kernel Profiling

A library of FlyDSL GPU kernels paired with **decoded rocprofv3 trace bundles**
and **analysis reports**. Each example under [`examples/`](examples) lets anyone
on any machine (no GPU required) load the kernel into the AMD **ATT Viewer**
for instruction-level inspection or **rocprof-compute-viewer** for counter
aggregates — without re-capturing the trace.

> ### 🛰️ MI350X / gfx950 full sweep — 2026-06-01
> Every major FlyDSL kernel profiled in one pass on **8× AMD Instinct MI350X**
> (FlyDSL `0.1.9.dev594` @ `18c5a7ed`, ROCm 7.2.0, rocprofv3 1.1.0):
> **17 kernels** with 95–100 % source-mapped ATT, **15 matched-shape baselines**
> (AIter / CK / hipBLASLt).
> → **[Interactive dashboard](https://jhinpan.github.io/flydsl-kernel-profiling/)**
> · **[FINDINGS.md](FINDINGS.md)** (the bird's-eye read)
> · Dashboard drawers now show exact **model → shape** coverage for every
> multi-shape benchmarked kernel.
>
> Headline: FlyDSL **wins** Softmax 2.05×, HGEMM-SplitK 1.66×, MoE-GEMM 1.11×;
> clear **headroom** on RoPE 0.17×, TopK-Gating 0.22×, Paged-Attn 0.48×. The
> attention/GEMM losses are register-pressure-capped occupancy (1 wave/SIMD);
> RoPE and TopK-Gating are structural (serialized cross-lane reductions).

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
| [`examples/flash_attn_func`](examples/flash_attn_func) | Flash Attention Func | `FlyDSL-lab@18c5a7e` | 371.7 TFLOPS at B=1 S=2048 H=32 D=128 with `total_CTAs=512/512`; cold-debug capture maps 2069/2070 ISA rows; ping-pong exists but consume points still stall (29 % `vmcnt`, 16 % `lgkmcnt`, 58.8 % stall ratio) |

### MI350X / gfx950 sweep — 2026-06-01 (FlyDSL 0.1.9.dev594 @ 18c5a7ed)

Ordered FlyDSL-wins → parity → headroom. Speedups are FlyDSL vs. the strongest matched-shape baseline. See **[FINDINGS.md](FINDINGS.md)** + the **[dashboard](https://jhinpan.github.io/flydsl-kernel-profiling/)**.

The dashboard now has a **Kernel × model coverage** visualization. Click any row
or kernel card to open the drawer; the **Model × shape coverage** table lists the
exact model/proxy name, stage, dtype, shape arguments, best baseline, speedup,
and production weight when a trace supplied one. The same source of truth is
checked in under `benchmarks/examples/<kernel>/shape_ledger.jsonl` and exported
to `docs/data/kernels.json`.

| kernel | shapes | model groups | models / proxies |
|---|---:|---:|---|
| `blockscale_preshuffle_gemm` | 14 | 3 | deepseek-v3 (10), kimi-k2 (2), qwen3 (2) |
| `flash_attn_func` | 16 | 4 | synthetic (9), DeepSeek-V3 (3), Kimi-K2 (2), Qwen3 (2) |
| `fp8_gemm_rowscale` | 14 | 4 | deepseek-v3 (6), kimi-k2 (3), qwen3-32b (3), synthetic (2) |
| `fused_rope_cache` | 30 | 5 | GPT-OSS 120B (6), Llama3 405B (6), Llama3 70B\|Qwen3-235B-A22B (6), Llama3 8B (6), Llama4 Maverick (6) |
| `hgemm_splitk` | 265 | 8 | DeepSeek-R1 (80), Llama3 405B (40), Llama3 70B (40), Llama3 8B (35), Llama4 Maverick (35), GPT-OSS 120B (15), Qwen3-235B-A22B (15), DeepSeek-R1\|Llama3 8B (5) |
| `layernorm` | 73 | 9 | synthetic (27), DeepSeek-R1 (15), GPT-OSS 120B (5), Llama3 405B (5), Llama3 70B (5), Llama3 8B\|Qwen3-235B-A22B (5), Llama4 Maverick (5), Qwen3-235B-A22B (5), diagnostic (1) |
| `mla_decode` | 14 | 3 | DeepSeek-R1 (6), DeepSeek-V3 (5), Kimi-K2 (3) |
| `moe_blockscale` | 12 | 2 | deepseek-v3 (7), kimi-k2 (5) |
| `moe_gemm` | 29 | 6 | DeepSeek-R1 (10), GPT-OSS 120B (5), Llama4 Maverick (5), Qwen3-235B-A22B (5), synthetic-profiled-att (3), synthetic-smallest-passing (1) |
| `moe_reduce` | 16 | 4 | deepseek-v3 (6), ep-k6 (4), kimi-k2 (3), qwen3-moe (3) |
| `pa` | 12 | 5 | test_pa normal_accuracy (8), DeepSeek/Kimi GQA decode (TP shard -> hkv=1) (1), Qwen-like (hidden 2560, 8 q-heads/shard) long-ctx decode (1), sliding-window model (e.g. Mistral-style) (1), sliding-window model long window (1) |
| `preshuffle_gemm` | 14 | 4 | generic (6), DeepSeek-V3 (3), Qwen (3), Kimi-K2 (2) |
| `quant` | 12 | 6 | DeepSeek-V3 (3), Qwen3 (3), flydsl_test (2), stress (2), Kimi-K2 (1), flydsl_test_default (1) |
| `rmsnorm` | 159 | 10 | DeepSeek-R1 (57), Qwen3-4B (44), synthetic (27), GPT-OSS 120B (5), Llama3 405B (5), Llama3 70B (5), Llama3 8B\|Qwen3-235B-A22B (5), Llama4 Maverick (5), Qwen3-235B-A22B (5), diagnostic (1) |
| `softmax` | 73 | 9 | synthetic (27), DeepSeek-R1 (15), GPT-OSS 120B (5), Llama3 405B (5), Llama3 70B (5), Llama3 8B\|Qwen3-235B-A22B (5), Llama4 Maverick (5), Qwen3-235B-A22B (5), diagnostic (1) |
| `topk_gating_softmax` | 16 | 5 | DeepSeek-R1 (6), Kimi-K2 (3), Mixtral-8x22B-class (3), Llama4-class (2), Mixtral-8x7B (2) |
| `vec_add` | 12 | 4 | micro (7), DeepSeek-V3 (2), Qwen3 (2), Kimi-K2 (1) |

| folder | kernel | source | headline |
|---|---|---|---|
| [`examples/softmax`](examples/softmax) | Softmax | `FlyDSL-lab@18c5a7e` | stall-bound (76% stalled, 46% VMEM-load), occ 5/SIMD; FlyDSL 271.8µs vs AIter triton 558µs (**2.05× win**) at 32768×8192 bf16 — but the fast vectorized path is dead-coded (`False`-gated), so it runs scalar 16-bit loads, leaving the HBM roofline on the table. |
| [`examples/hgemm_splitk`](examples/hgemm_splitk) | HGEMM Split-K | `FlyDSL-lab@18c5a7e` | latency-bound (56% VMEM-wait, MFMA only 2%), occ 2/SIMD @117 VGPR; FlyDSL 7.0µs = 25 TFLOPS, **1.66× vs PyTorch** 11.6µs — only 84 workgroups on 256 CUs, so SPLIT_K=14 starves each block to ~2 K-iters. |
| [`examples/moe_gemm`](examples/moe_gemm) | MoE GEMM (2-stage) | `FlyDSL-lab@18c5a7e` | stall-bound (55% VMEM-wait), occ 1/SIMD @155 VGPR; FlyDSL stage-1 70.8µs = CK 71.1µs (parity), **1.11× on full 2-stage** (stage-2 atomic 1.30×) — load pipeline unpipelined, matrix cores starved. |
| [`examples/layernorm`](examples/layernorm) | LayerNorm | `FlyDSL-lab@18c5a7e` | stall-bound (58% LDS/SMEM-wait), occ 3/SIMD @72 VGPR; FlyDSL 24.1µs ≈ AIter 24.7µs (1.03×) — dependent `shuffle_xor` cross-lane reduction tree is the ceiling, not HBM. |
| [`examples/moe_reduce`](examples/moe_reduce) | MoE Reduction | `FlyDSL-lab@18c5a7e` | bandwidth-bound (91% VMEM load+wait), occ 4/SIMD @56 VGPR; FlyDSL 382.7µs ≈ torch.sum / aiter.moe_sum 382.6µs (1.00×) at the ~5.5 TB/s HBM ceiling — already optimal. |
| [`examples/quant`](examples/quant) | Per-Token Quant | `FlyDSL-lab@18c5a7e` | bandwidth-bound (77.5% total stall), occ 5/SIMD; FlyDSL 16.74µs vs AIter 16.05µs (0.96×) — near the HBM3E roofline; recoverable budget is ~23% barrier+LDS-wait from a two-barrier block reduction. |
| [`examples/mla_decode`](examples/mla_decode) | MLA Decode (fp8) | `FlyDSL-lab@18c5a7e` | stall-bound (83% LDS/SMEM-wait), occ 1/SIMD @ VGPR≈251; FlyDSL 12.40µs vs aiter-HK-CK 11.19µs (0.90×) — exposed LDS→MFMA operand-feed latency on a single-wave decode. |
| [`examples/rmsnorm`](examples/rmsnorm) | RMSNorm | `FlyDSL-lab@18c5a7e` | bandwidth-bound (41% VMEM-wait), occ 4/SIMD @60 VGPR; FlyDSL 25.1µs vs AIter 22.4µs (0.89×) — single `vmcnt(0)` load-drain before the block reduction is the ceiling. |
| [`examples/blockscale_preshuffle_gemm`](examples/blockscale_preshuffle_gemm) | Block-Scale Preshuffle GEMM | `FlyDSL-lab@18c5a7e` | **compute-bound @M=4096**: FlyDSL 869 TFLOPS (156µs) vs AIter **tuned**-CK 1322 TFLOPS (102µs) → **0.66×** — gap widens vs M=16 (0.88×); partly a tuning-parity gap (CK loads per-shape tables, FlyDSL runs a fixed schedule). Needs split-K + tuned tiles. |
| [`examples/preshuffle_gemm`](examples/preshuffle_gemm) | Preshuffle GEMM | `FlyDSL-lab@18c5a7e` | **compute-bound @4096³ fp8**: FlyDSL 1347 TFLOPS (102µs) vs AIter-CK (untuned) 1760 TFLOPS (78µs) → **0.77×** — at saturation the gap is real (M=16 was launch-bound/noise). ATT occ 3/SIMD: deepen the K-loop prefetch to overlap HBM loads with MFMA. |
| [`examples/moe_blockscale`](examples/moe_blockscale) | MoE Block-Scale (2-stage) | `FlyDSL-lab@18c5a7e` | bandwidth-bound (78% of stalls VMEM), occ 2/SIMD @~203 VGPR; FlyDSL 53.8µs vs CK 44.0µs (0.82×) — MFMA-scale starved on an under-prefetched FP8 operand/scale chain. |
| [`examples/pa`](examples/pa) | Paged-Attn Decode (PS) | `FlyDSL-lab@18c5a7e` | stall-bound (65.7%), occ 1/SIMD (VGPR 176, needs ≤128 for 2 waves); FlyDSL 169.5µs vs AIter Gluon 80.6µs (**0.48×**) — single resident wave can't hide its K/V-load + softmax-LDS latency. |
| [`examples/topk_gating_softmax`](examples/topk_gating_softmax) | TopK Gating Softmax | `FlyDSL-lab@18c5a7e` | stall-bound (43% LGKMCNT-wait + 50% "other"), occ 4/SIMD; FlyDSL 30.9µs vs AIter-HIP 6.7µs (**0.22×**) — K=6 serial `shuffle_xor` argmax butterflies on LGKMCNT are the ceiling, not memory. |
| [`examples/fused_rope_cache`](examples/fused_rope_cache) | Fused RoPE + KV-Cache | `FlyDSL-lab@18c5a7e` | stall-bound (80% `lgkmcnt`), occ 8/SIMD but 1 wave/block; FlyDSL 219.6µs vs AIter 37.5µs (**0.17×**) — serialized buffer-descriptor + position + `ds_bpermute` fence chain in a single 64-lane wave. |
| [`examples/vec_add`](examples/vec_add) | Vector Add | `FlyDSL-lab@18c5a7e` | bandwidth-bound (82% VMEM-wait), occ 8/SIMD, 9 VGPR; 6468 GB/s ≈ 81% of HBM3E peak — already-optimal 128-bit streaming triad, no body headroom. |
| [`examples/preshuffle_gemm_v2`](examples/preshuffle_gemm_v2) | Preshuffle GEMM v2 | `FlyDSL-lab@18c5a7e` | **internal v2-vs-v1 @4096×5120×8192 bf16**: v2 layout-API **767 TFLOPS = 1.20× over v1 manual** (638 TFLOPS) — the layout-API refactor is a real FlyDSL-internal win; the external CK gap is tracked under `preshuffle_gemm`. |

## How to use a captured trace

```bash
git clone https://github.com/jhinpan/flydsl-kernel-profiling-lib
cd flydsl-kernel-profiling-lib/examples/<kernel>

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

For ATT source mapping, capture from a **fresh FlyDSL debug cache**. Set
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` and an isolated `FLYDSL_RUNTIME_CACHE_DIR`
before any discovery/capture command that can trigger JIT. If a no-debug HSACO
is already cached, rocprofv3 cannot add Python line tables later; `code.json`
will show `mapped=0/N` even with `AsmDebug: True`.

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
