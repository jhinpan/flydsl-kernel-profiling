# Flash Attention Func — Ping-Pong / ATT Analysis

Source snapshot: `/sgl-workspace/FlyDSL-lab@18c5a7ed79cc5bece508896a95270f1dadb7859b`
Kernel: `flash_attn_func_kernel_0` (bf16 causal, gfx950 / MI350X)
Probe: `/sgl-workspace/jin/flash_attn_func_probe/`

## Workload configurations measured

| config | role | B | S | H | D | BLOCK_M | CTAs | target_CTAs | CU/Utilization tail verdict | wall (us) | TFLOPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| small | underfilled diagnostic | 1 | 1024 | 16 | 128 | 128 | 128 | 512 | underfilled by design, but sampled CU has waves | 38.2 | 112.5 |
| big | primary / saturation-validated | 1 | 2048 | 32 | 128 | 128 | 512 | 512 | validated: 32 wave JSONs, 512 occupancy rows, no long sampled-CU tail | 92.4 | 371.7 |

Grid sizing comes from `grid_x = batch * ceil(seq_len / BLOCK_M) * num_heads`.
For the primary trace that is `1 * 16 * 32 = 512` CTAs. rocprof's discovery CSV
cross-checks this as `Grid_Size_X=131072` threads / `Workgroup_Size_X=256`.
The kept cold-debug traces are `small/ui_output_agent_32235_dispatch_44` and
`big/ui_output_agent_38430_dispatch_44`, both with 2069 / 2070 ISA rows mapped
to Python source.

## 0. Source-location granularity: before → after (#587)

The original traces map 2069/2070 rows but the mapping is useless for navigating
the schedule: **91.9 % of kernel instructions collapse onto `flash_attn_func.py:257`**
(the `@flyc.kernel` decorator) and another 6.3 % onto `:283` (the body of the
one-line `_mfma` helper). Every DMA-to-LDS load, every matmul, every wait lands
on one of two lines. Root cause: FlyDSL emits these ops from nested helper
closures, so `traced_op`'s `_caller_location(depth=1)` resolves to the helper
body, not the user's scheduling line.

The fix (FlyDSL PR #593, branch `jhinpan:fix-issue-587-att-source-loc`, issue
#587) is two layers, both **pure metadata** — debug-off ISA is byte-identical and
kernel perf is unchanged (372 TFLOPS, see config table). Run
`source/att_source_granularity.py <before>/code.json <after>/code.json` to reproduce:

| metric (big, dispatch_44) | before | after |
|---|---:|---:|
| distinct mapped source lines | 7 | **30** |
| top-1 line weight share (Hit+Stall) | 0.76 | 0.53 |
| effective #lines `exp(H)` | 2.0 | **7.1** |
| weight on `:283` (`_mfma`) | 19.9 % | **0 %** |
| GEMM MFMA → call sites | — | 744/745 (GEMM1), 1115/1116 (GEMM2) |
| K/V DMA, waits, V pre-read → kernel lines | — | 658, 696, 738, 693, 981, 1108, 1114 |
| softmax max / sum / rescale → kernel lines | — | 922–961 (`_fmax`/`_fadd`/`_fmul`/`_fsub`) |

1. **Scheduling path** — a `source_loc` scope (honored by untraced ODS builders
   via MLIR's ambient location and by `@traced_op` leaves via a thread-local pin)
   applied to `coop_dma_k` / `coop_dma_v` / `mfma_acc` / `_waitcnt_vm_n` /
   `_read_v_pack`, plus loc-aware `gpu.barrier` / `s_waitcnt` / `sched_*`. K DMA
   double-buffer, GEMM1/GEMM2 MFMA, the V-visibility waits and the GEMM2 V pre-read
   now each resolve to the line that expresses them. Side-effect: 64 of 84
   backend-inserted `s_waitcnt` rows moved onto the MFMA lines they guard.
2. **Softmax compute** — `@source_loc_scope` on the `_fadd`/`_fsub`/`_fmul`/`_fmax`
   arith helpers, so the online-softmax running-max / exp-sum and the o_acc rescale
   (`_fmul(Vec(o_acc), corr_vec)`) attribute to lines 922–961 instead of the
   function line. This is what makes the GEMM↔softmax ping-pong's softmax half
   navigable.

What still sits on the function line (`:258`, ~54 % of weight) is **not**
FlyDSL-source-controllable: 166 backend `s_nop` hazard-pad rows plus a few
very-high-stall compiler-inserted `s_waitcnt` barriers (no source line), and the
`exp2` / `select` ops — these are `ArithValue` methods (framework arith layer,
what FlyDSL commit `9f29c0de` targets), not kernel helpers.

After traces: `big/ui_output_agent_50261_dispatch_44_after_loc_fix`,
`small/ui_output_agent_23205_dispatch_44_after_loc_fix` (recaptured from FlyDSL
branch tip `36e9f0f6`). The `source/` kernel is the patched (post-fix) version;
the pre-fix `dispatch_44` folders keep their own captured
`source_0_flash_attn_func.py` snapshot.

## 1. What this kernel is doing

This is a useful flash-attention trace because the source is explicitly built
around the schedule we wanted to inspect:

| source region | schedule role |
|---|---|
| `flash_attn_func.py:103-142` | builds M128 and M256 variants for `H >= 32`; this shape chooses M128 because `B*S*H < 4096*H` |
| `flash_attn_func.py:643-698` | DMA double-buffer K into LDS; `_k_buf_id` alternates current/next buffers |
| `flash_attn_func.py:709-744` | GEMM1: read K packs from LDS and issue `K @ Q^T` MFMA |
| `flash_attn_func.py:746-953` | online softmax in registers |
| `flash_attn_func.py:970-980` | wait/barrier for V DMA-to-LDS before GEMM2 |
| `flash_attn_func.py:1067-1112` | GEMM2: pre-read V pack, then interleave V reads with `V^T @ P` MFMA |

So the code does implement a K/V ping-pong style schedule: K is double-buffered
across subtiles and V is loaded while GEMM1/softmax work is in flight. The ATT
trace says the structure is correct, but the waits are still visible at the
consume points.

## 2. Headline wave-state breakdown

Timeline state mapping: `1=EXEC`, `2=WAIT`, `3=STALL`, `4=SLEEP`.

| state | small | big | meaning |
|---|---:|---:|---|
| EXEC | 0.4 % | 0.0 % | instruction issue |
| WAIT | 48.4 % | 42.3 % | operand / scheduler wait |
| STALL | 34.8 % | 36.7 % | explicit stall, mostly waitcnt/barrier |
| SLEEP | 16.4 % | 21.0 % | sampled wave inactive windows |

The primary trace is not compute-issue bound. `hotspot_analyzer.py` reports
2.86M total cycles, 1.68M stall cycles, and a 58.8 % stall ratio.

Stall taxonomy from `hotspot_analyzer.py` on `att_viewer/big/...dispatch_44`:

| type | stall cycles | % of total stall |
|---|---:|---:|
| other | 492.9K | 29.4 % |
| VMEM-wait | 490.6K | 29.2 % |
| LDS/SMEM-wait | 274.0K | 16.3 % |
| MFMA/FMA | 133.3K | 7.9 % |
| VMEM-load | 105.9K | 6.3 % |
| VMEM-store | 88.6K | 5.3 % |
| barrier | 66.9K | 4.0 % |
| LDS | 25.4K | 1.5 % |

Per-instruction-class latency/stall breakdown:

| class | count | latency cycles | stall cycles | lat % | stall % |
|---|---:|---:|---:|---:|---:|
| VALU | 1366 | 1.178M | 266.8K | 41.2 % | 15.9 % |
| waitcnt | 84 | 764.5K | 764.5K | 26.8 % | 45.6 % |
| SALU | 407 | 311.2K | 226.1K | 10.9 % | 13.5 % |
| VMEM | 44 | 214.7K | 194.4K | 7.5 % | 11.6 % |
| MFMA | 68 | 199.3K | 133.3K | 7.0 % | 7.9 % |
| LDS | 96 | 120.8K | 25.4K | 4.2 % | 1.5 % |
| barrier | 4 | 66.9K | 66.9K | 2.3 % | 4.0 % |

## 3. Top hotspot PCs

| PC | hit | stall | instruction | interpretation |
|---:|---:|---:|---|---|
| 11448 | 248 | 142.9K | `s_waitcnt vmcnt(0)` | waits after `buffer_load_dwordx4 ... lds`, before V/K LDS reads |
| 15492 | 248 | 111.1K | `s_waitcnt vmcnt(0)` | second repeated DMA-to-LDS drain point |
| 11664 | 248 | 96.9K | `s_waitcnt vmcnt(0)` | consume-side wait after GEMM1 MFMA group |
| 15672 | 248 | 81.2K | `s_waitcnt vmcnt(0)` | same pattern in later unrolled body |
| 7284 | 32 | 43.6K | `s_waitcnt vmcnt(7)` | early global/DMA prefetch drain |
| 5900 | 32 | 41.2K | `s_waitcnt lgkmcnt(0)` | scalar prologue argument load |
| 11264 | 248 | 33.0K | `s_barrier` | barrier after waitcnt before LDS reuse |
| 11320 | 248 | 27.1K | `buffer_load_dwordx4 ... lds` | DMA-to-LDS issue is itself visible |
| 6720 | 32 | 21.7K | `buffer_load_dwordx4 ... lds` | early DMA-to-LDS load |
| 12448 | 248 | 14.4K | `s_nop 10` | compiler-inserted wait before softmax value movement |

The repeated pattern is:

```asm
buffer_load_dwordx4 ... lds
s_waitcnt vmcnt(0)        ; large stall
ds_read_b128 ...
v_mfma_f32_32x32x16_bf16 ...
s_waitcnt lgkmcnt(1/0)    ; smaller but frequent
```

That is the ATT-level evidence that K/V ping-pong is not fully hiding the
DMA-to-LDS latency. The next buffer is issued, but the consumer still catches
up to it.

## 4. Ping-pong verdict

The kernel does have a real ping-pong/circular-buffer schedule:

- K uses `_cur_buf_id` / `1 - _cur_buf_id` and issues `coop_dma_k` for the next
  subtile before consuming the current one.
- V is DMA-loaded before GEMM2 and GEMM2 pre-reads the next V pack while issuing
  the current pair of MFMA instructions.
- Softmax probabilities stay in registers and feed GEMM2 directly, so there is
  no P LDS roundtrip.

The trace also shows why performance is not just "MFMA throughput":

- The largest named bucket is VMEM wait, not MFMA issue.
- Barriers and `lgkmcnt` waits cluster exactly around the K/V LDS reuse points.
- `other` and VALU latency are high because online softmax, mask handling,
  exp2, packing, and accumulator rescaling remain on the critical path between
  the two GEMMs.

## 5. Optimization candidates

Priority ranking:

1. Increase K/V DMA prefetch distance or try the existing 3-buffer path
   (`FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3=1`) for this shape. The top four
   PCs are all `vmcnt(0)` drains after DMA-to-LDS.
2. Split the hard `s_waitcnt(0)` + `s_barrier` boundaries where possible. Some
   softmax/packing work can potentially run before all outstanding DMA drains.
3. Revisit GEMM2 interleaving: pre-reading one V pack is not enough to hide the
   repeated `lgkmcnt(1)` waits around the MFMA group.
4. Reduce register pressure in the softmax/P-pack path. rocprof discovery
   reports `VGPR_Count=124`; the analyzer's occupancy heuristic labels this as
   a limiting factor, though its architecture string incorrectly says gfx942
   for this gfx950 capture.
5. Improve line attribution granularity. Cold-debug recapture fixed source
   mapping (`2069 / 2070` mapped), but many hot waitcnts still aggregate to the
   kernel decorator or MFMA wrapper lines; precise phase labels still require
   correlating the PC pattern with the source regions above.
