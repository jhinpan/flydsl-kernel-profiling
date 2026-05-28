# Flash Attention Func — Ping-Pong / ATT Analysis

Source snapshot: `/sgl-workspace/FlyDSL-lab@18c5a7ed79cc5bece508896a95270f1dadb7859b`
Kernel: `flash_attn_func_kernel_0` (bf16 causal, gfx950 / MI350X)
Probe: `/sgl-workspace/jin/flash_attn_func_probe/`

## Workload configurations measured

| config | role | B | S | H | D | BLOCK_M | CTAs | target_CTAs | CU/Utilization tail verdict | wall (us) | TFLOPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| small | underfilled diagnostic | 1 | 512 | 8 | 128 | 128 | 32 | 512 | underfilled by design | 22.4 | 23.9 |
| big | primary / saturation-validated | 1 | 2048 | 32 | 128 | 128 | 512 | 512 | validated: 32 wave JSONs, 512 occupancy rows, no long sampled-CU tail | 92.4 | 371.7 |

Grid sizing comes from `grid_x = batch * ceil(seq_len / BLOCK_M) * num_heads`.
For the primary trace that is `1 * 16 * 32 = 512` CTAs. rocprof's discovery CSV
cross-checks this as `Grid_Size_X=131072` threads / `Workgroup_Size_X=256`.

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
| EXEC | 1.0 % | 0.0 % | instruction issue |
| WAIT | 43.5 % | 42.7 % | operand / scheduler wait |
| STALL | 35.1 % | 35.9 % | explicit stall, mostly waitcnt/barrier |
| SLEEP | 20.4 % | 21.4 % | sampled wave inactive windows |

The primary trace is not compute-issue bound. `hotspot_analyzer.py` reports
3.17M total cycles, 1.86M stall cycles, and a 58.5 % stall ratio.

Stall taxonomy from `hotspot_analyzer.py` on `att_viewer/big/...dispatch_44`:

| type | stall cycles | % of total stall |
|---|---:|---:|
| other | 575.1K | 31.0 % |
| VMEM-wait | 518.6K | 28.0 % |
| LDS/SMEM-wait | 304.7K | 16.4 % |
| MFMA/FMA | 157.9K | 8.5 % |
| VMEM-load | 106.5K | 5.7 % |
| VMEM-store | 86.4K | 4.7 % |
| barrier | 75.1K | 4.0 % |
| LDS | 30.7K | 1.7 % |

Per-instruction-class latency/stall breakdown:

| class | count | latency cycles | stall cycles | lat % | stall % |
|---|---:|---:|---:|---:|---:|
| VALU | 1366 | 1.342M | 322.2K | 42.3 % | 17.4 % |
| waitcnt | 84 | 823.3K | 823.3K | 26.0 % | 44.4 % |
| SALU | 407 | 343.2K | 252.9K | 10.8 % | 13.6 % |
| MFMA | 68 | 232.4K | 157.9K | 7.3 % | 8.5 % |
| VMEM | 44 | 215.3K | 192.9K | 6.8 % | 10.4 % |
| LDS | 96 | 138.4K | 30.7K | 4.4 % | 1.7 % |
| barrier | 4 | 75.1K | 75.1K | 2.4 % | 4.0 % |

## 3. Top hotspot PCs

| PC | hit | stall | instruction | interpretation |
|---:|---:|---:|---|---|
| 11448 | 280 | 150.6K | `s_waitcnt vmcnt(0)` | waits after `buffer_load_dwordx4 ... lds`, before V/K LDS reads |
| 15492 | 280 | 120.8K | `s_waitcnt vmcnt(0)` | second repeated DMA-to-LDS drain point |
| 11664 | 280 | 101.4K | `s_waitcnt vmcnt(0)` | consume-side wait after GEMM1 MFMA group |
| 15672 | 280 | 85.4K | `s_waitcnt vmcnt(0)` | same pattern in later unrolled body |
| 7284 | 32 | 44.5K | `s_waitcnt vmcnt(7)` | early global/DMA prefetch drain |
| 5900 | 32 | 40.1K | `s_waitcnt lgkmcnt(0)` | scalar prologue argument load |
| 11264 | 280 | 34.2K | `s_barrier` | barrier after waitcnt before LDS reuse |
| 11320 | 280 | 22.0K | `buffer_load_dwordx4 ... lds` | DMA-to-LDS issue is itself visible |
| 6720 | 32 | 20.6K | `buffer_load_dwordx4 ... lds` | early DMA-to-LDS load |
| 12448 | 280 | 16.3K | `s_nop 10` | compiler-inserted wait before softmax value movement |

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
5. Fix source mapping for future captures. `code.json` has 0 / 2070
   instructions mapped despite debug-info mode, so current PC-to-source mapping
   requires manual source-region correlation.
