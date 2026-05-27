# FP4 MQA Logits — Rocprof v3 / ATT Instruction-Level Analysis

Commit: `9120078d35d7d232b3941ded5b76a1ca92329ef0` ("Optimize FP4 MQA logits pipelining")
Kernel: `pa_mqa_logits_fp4_kernel_0` (Q FP4, KV FP4, MFMA(Q_fp4, KV_fp4), gfx950 / MI350X)
Workspace: `/sgl-workspace/jin/fp4_mqa_probe/`

## Workload configurations measured

| config | batch | ctx | block_k | safe_chunks/CTA | total_CTAs | wall (us) | TFLOPS |
|---|---|---|---|---|---|---|---|
| small | 4 | 8192 | 256 | 1 | 69 | 4.50 | 63.7 |
| big | 8 | 65536 | 256 | 3 | 391 | 8.13 | 605.5 |

The "small" config exposes the **prologue / cold-start** overhead; the "big" config exposes the **loop-body** behaviour.

---

## 1. Headline wave-state breakdown

State of one CU's waves during a single steady-state dispatch (ATT v3.0, gfx950, 1 CU × 4 SIMDs × 4 SEs):

| state | small (1 chunk/CTA) | big (3 chunks/CTA) | meaning |
|---|---|---|---|
| **EXEC** | 4.1 % | 0.3 % | instruction issue |
| **STALL** | 56.9 % | 33.8 % | `s_waitcnt` not yet retired |
| **WAIT** | 31.6 % | 47.0 % | operand-dep / VALU pipeline |
| **SLEEP** | 7.3 % | 19.0 % | power-down windows |

**The kernel never spends more than ~4 % of a wave's life actually issuing instructions.** Even in production-shape runs it is overwhelmingly waiting on memory or MFMA-result dependencies — there is no realistic way to make this compute-bound, but the *wait* time is what we can attack.

Per-instruction-class latency breakdown (production-size trace, 28 sampled waves, 727 static instructions):

| class | count | latency cycles | stall cycles | lat % | stall % |
|---|---|---|---|---|---|
| **valu** | 520 | 125 908 | 41 204 | **47.3 %** | 28.5 % |
| **waitcnt** | 35 | 89 960 | 89 960 | 33.8 % | **62.1 %** |
| salu | 73 | 16 312 | 1 860 | 6.1 % | 1.3 % |
| **mfma** | 32 | 13 540 | 3 696 | 5.1 % | 2.6 % |
| **vmem** | 31 | 12 392 | 7 412 | 4.7 % | 5.1 % |
| lds (bperm) | 16 | 6 148 | 620 | 2.3 % | 0.4 % |
| smem | 15 | 1 704 | 24 | 0.6 % | 0.0 % |
| sbranch | 4 | 460 | 0 | 0.2 % | 0.0 % |

Read this as: of the kernel's wall-clock, ~34 % is direct `s_waitcnt` stall, ~47 % is VALU latency *most of which is dependency-stall on MFMA results*, and only ~5 % is the MFMA pipeline itself.

---

## 2. Top instruction-level hotspots

Sorted by total latency (over 28 wave samples) — these are the optimisation targets.

### 2a. Prologue / cold-start (small workload pays this 100 %)

| PC | latency | stall | instruction | what it's waiting on |
|---|---|---|---|---|
| 6736 | 27 148 | 27 148 | `s_waitcnt lgkmcnt(0)` | second batch of kernarg scalar loads (Q/W/strides) |
| 6432 | 19 032 | 19 032 | `s_waitcnt lgkmcnt(0)` | first batch of kernarg scalar loads (cta_info SRD) |
| 6576 | 1 088 (big) / 4 940 (small) | – | `s_waitcnt vmcnt(0)` | the `buffer_load_dwordx4` that fetches the packed cta_info |
| 7164 | 3 004 | 3 004 | `s_waitcnt vmcnt(3)` | drains KV-prefetch to ≤3 outstanding loads |

Context around PC 6432 (the very first stall):
```
s_load_dwordx2 s[8:9], s[0:1], 0x130   ; cta_info_ptr base addr
s_load_dword   s30,    s[0:1], 0x148   ; stride_out_batch
s_lshl_b32     s2,     s2, 4           ; pid * 16 (bytes per cta_info row)
s_mov_b32      s11,    0x27000         ; SRD hi-word (max_size encoding)
s_mov_b32      s10,    -1              ; SRD num_records
>>> s_waitcnt  lgkmcnt(0)              ; ← 6056 cycles in small / 19032 in big
s_and_b32      s9, s9, 0xffff          ; mask hi bits of cta_info base
v_mov_b32_e32  v1, s2                  ; lift pid*16 to VGPR
buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen  ; THE cta_info load
```

Then PC 6576 waits for that `buffer_load_dwordx4` (`s_waitcnt vmcnt(0)`), then PC 6736 waits again on a second wave of `s_load_dword` ops fetching Q/W/KV strides. **Three serial memory-barrier stalls fence the prologue's critical path.**

### 2b. Loop-body — biggest single in-loop stall

| PC | latency | stall | instruction | meaning |
|---|---|---|---|---|
| **7540** | **10 356** | **10 356** | `s_waitcnt vmcnt(3)` | wait for prefetched KV to begin draining |
| 7048 | 4 224 | 4 224 | `s_waitcnt vmcnt(4)` | same, earlier point |
| 7236 | 3 068 | 3 068 | `s_waitcnt vmcnt(0)` | drain all KV before last `nt` MFMA |
| 7216 | 2 516 | 2 516 | `s_waitcnt vmcnt(1)` | etc. |

Context around PC 7540 (the loop's worst stall):
```
v_bfe_u32     v0,  v1, 8,  8           ; extract nt=1 scale byte from packed kvs
v_bfe_u32     v94, v1, 16, 8           ; extract nt=2 scale byte
v_lshrrev_b32 v53, 24, v1              ; extract nt=3 scale byte
>>> s_waitcnt vmcnt(3)                 ; ← 10 356 cycles
v_mfma_scale_f32_16x16x128_f8f6f4 ...  ; nt=1 mfma issue (uses prefetched kv)
v_maximum3_f32 v1, v35, 0, 0           ; relu on prior nt's MFMA acc
```

This is the loop body issuing a fresh `_issue_nt_mfmas(nt=1)` whose operands came from `_prefetch_chunk`. The prefetch is *not* completing before the consume point — **the pipelining isn't deep enough**.

### 2c. Post-process VALU dependency chains

| PC | latency | stall | instruction |
|---|---|---|---|
| 7684 | 3 748 | 3 524 | `v_pk_mul_f32 v[0:1], v[0:1], v[18:19]`  (relu * w) |
| 8908 | 3 500 | 3 276 | `v_pk_mul_f32 v[2:3], v[2:3], v[12:13]` |
| 8064 | 2 628 | 2 404 | `v_pk_mul_f32 v[26:27], v[26:27], v[18:19]` |
| 8480 | 1 676 | 1 452 | `v_pk_mul_f32 v[0:1], v[0:1], v[6:7]` |
| 8704 | 1 372 | 1 148 | `v_pk_mul_f32 v[0:1], v[0:1], v[18:19]` |

Each `v_pk_mul_f32` sits on the chain:
```
acc[mi]  = v_mfma_scale_f32_16x16x128_f8f6f4 ...   ; writes v[34:37] (last in group)
                                                      ↓ ~16 cycle MFMA-result latency
relu_v   = v_maximum3_f32 v[0..3], 0, 0            ; reads v34..v37
                                                      ↓ ~5 cycle VALU latency
prod_v   = v_pk_mul_f32 relu_v, w_per_lane[mi]     ; ← STALL HERE
```

The MFMAs in a group write registers `v22, v26, v30, v34` *in order*, but the relu / post-process consumes `v34` *first* — i.e. the **most recently written register**, which is the **least latency-hidden**.

### 2d. MFMA pipeline bubble at PC 9468 (small workload)

```
v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], ...   ; last MFMA in 4-MFMA group
s_cbranch_scc1 65050                              ; loop back-edge (taken=0 here)
v_mov_b32_e32  v2, s26
>>> s_nop 2                                       ; 1100 cycles in small / 336 in big
v_maximum3_f32 v51, v35, 0, 0                     ; reads v35 (just written by MFMA)
```

The `s_nop 2` is the compiler's hint that the MFMA result `v[34:37]` needs more wait-states before VALU can read it. In production size this is only 336 cycles (3 per iteration) — bearable. In small size where the loop body is entered through the epilogue-only path it inflates to 1100. Re-ordering the post-process to start on the *first*-written MFMA target (`v22`) would push this stall out of the critical path entirely.

---

## 3. Where each stall maps in the Python source

Mapping the hot PCs back to `kernels/pa_mqa_logits_fp4.py`:

| PC range | Python source region | what runs there |
|---|---|---|
| 6400-6580 | `pa_mqa_logits_fp4_kernel:284-298` | prologue: thread_idx + cta_info load |
| 6580-6740 | `pa_mqa_logits_fp4_kernel:300-320` | decode `cta_info_4xi32` + SRD setup |
| 6740-7280 | `pa_mqa_logits_fp4_kernel:328-396` | hoisted Q-load + Q-scale-load + weight-load |
| 7280-7540 | `_load_phys` + `_prefetch_chunk` (lines 407-489) | chunk-0 prefetch (`phys_pre`, `kv_pre`, `kvs_pre`) |
| 7540-7700 | `_extract_kvs_scales` + first `_issue_nt_mfmas` | extract NTPW=4 scales, fire nt=0 MFMAs |
| 7700-9460 | `_compute_chunk` loop-body MFMA + post-process | 4 nt × 4 mi MFMAs and per-nt relu/mul/sum |
| 9460-11020 | epilogue `_compute_chunk(last_c_i32)` (lines 697-708) | last chunk's pipelined-nt processing |
| 11028 | `s_endpgm` | termination |

---

## 4. Optimisation recommendations, in order of expected impact

### A. Re-order MFMA target registers vs. post-process consumption
**Likely impact: medium, easy.**
`_issue_nt_mfmas` (line 505) writes `accs[mi_idx]` in `mi_idx = 0,1,2,3` order, but `_post_process_nt` (line 536) consumes them in the same order. Because of register pressure assignment the compiler ends up with v22, v26, v30, v34 written in order and v34 read *immediately* in the relu (`v_maximum3_f32 v51, v35, 0, 0`). Reversing the `_post_process_nt` loop to `for mi_idx in [m_tiles-1, ..., 0]` (or rotating it so the *oldest* MFMA target is consumed first) would let the MFMA pipeline drain naturally — eliminating most of the `s_nop 2` / 3-5k-cycle `v_pk_mul_f32` stalls in §2c/2d.

### B. Issue next-chunk KV prefetch earlier in the loop body
**Likely impact: large for steady state, medium effort.**
PC 7540's 10 356-cycle `vmcnt(3)` stall says the KV prefetch isn't done in time. The chunk-loop body currently does (lines 670-695):
```python
_compute_chunk(kv_cur_list, kvs_cur_list, ..., nt0_accs_in=nt0_accs_cur)  # consumes carry
kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)            # prefetch
phys_next_next_list = _load_phys(c_next_next_i32)                          # phys for c+2
nt0_accs_next = _issue_nt_mfmas(kv_next, ..., 0)                           # pre-issue
```
The prefetch happens *after* the entire current chunk's compute. Moving it to right after the chunk-0 nt=0 MFMA pre-issue (i.e., interleaved with the relu/mul/sum/store work in `_compute_chunk`) would give the prefetch the entire chunk's post-process time to land.

A cleaner refactor: split `_compute_chunk` so `_prefetch_chunk(c+1)` is called *between* the nt=0 pre-issue and the first `_post_process_nt(nt=0)`. The 700-3500 cycle VALU chain that follows would then perfectly hide a ~1-2 μs cache miss.

### C. Coalesce kernarg `s_load`s in the prologue
**Likely impact: medium for short workloads, low effort.**
PCs 6432 / 6736 sit at the end of *two separate batches* of `s_load_dword{x2}` from kernarg space (`s[0:1]`). Each batch ends with `s_waitcnt lgkmcnt(0)`, serializing the prologue. The compiler emits separate loads because FlyDSL's tensor-pointer ABI lifts each `fx.Tensor` argument's stride / size separately. Two improvements:
1. Pack consecutively-used kernarg fields into 8-dword tuples that the compiler can lower into one `s_load_dwordx8`. The current spread spans offsets `0x30, 0x40, 0x48, 0x68, 0x80, 0x88, 0x100, 0x118, 0x130, 0x148` — most are within an 0x80-byte window and could fold.
2. Issue the second batch *before* the first `s_waitcnt` so both batches' latency overlaps. Today the compiler emits the second batch only after consuming results of the first.

For repeated-kernel-launch workloads (the production case), the SQC kernarg cache helps after the first dispatch, so this is mostly a *cold first dispatch* win — but in the small config it's >40 % of the kernel.

### D. Defer scale extraction
**Likely impact: small, low effort.**
`_extract_kvs_scales` (line 491) extracts all NTPW=4 nt scales up-front from packed kvs i32s — three `v_bfe_u32` + one `v_lshrrev_b32_e32` immediately before the loop-prologue `s_waitcnt vmcnt(3)`. The comment claims this "decouples bfe from the mfma dep chain", but the trace shows those four extractions concentrate right at the critical path. Lazy per-nt extraction (extract inside `_issue_nt_mfmas` for *that* nt only, immediately before the MFMA that consumes it) gives the compiler more freedom to overlap the bfe with whatever VMEM the scheduler chose to issue.

### E. Reduce KV `buffer_load_dwordx4` count via wider vectorisation
**Likely impact: small unless KV bandwidth is the constraint.**
`_prefetch_chunk` issues `N_TILES_PER_WARP * k_tiles = 4 * 1 = 4` separate `buffer_load_dwordx4` for KV per warp per chunk. The 4 loads are at addresses that differ by `_kv_chunk_bytes (16)` per nt — they're not coalescable with a wider vec_width because they target different physical pages. The KVS already does this collapse (1 packed dword for all 4 nts). Doing the same for KV would require a host-side preshuffle that interleaves token bytes across nts — a deeper refactor, but cuts SQ_INSTS_VMEM_RD by 4×.

---

## 5. Files in this analysis

| path | content |
|---|---|
| `/sgl-workspace/jin/fp4_mqa_probe/kernels/pa_mqa_logits_fp4.py` | kernel source (from commit) |
| `/sgl-workspace/jin/fp4_mqa_probe/tests/kernels/test_pa_mqa_logits_fp4.py` | test/bench harness |
| `/sgl-workspace/jin/fp4_mqa_probe/input_trace.yaml` | rocprofv3 config — small workload |
| `/sgl-workspace/jin/fp4_mqa_probe/input_trace_big.yaml` | rocprofv3 config — production workload |
| `/sgl-workspace/jin/fp4_mqa_probe/prof/discover_*` | kernel-discovery CSVs (rocprofv3 --stats) |
| `/sgl-workspace/jin/fp4_mqa_probe/prof/att/ui_output_agent_*` | ATT trace, small workload |
| `/sgl-workspace/jin/fp4_mqa_probe/prof/att_big/ui_output_agent_*` | ATT trace, production workload |

## 6. How to re-run

```bash
cd /sgl-workspace/jin/fp4_mqa_probe

# baseline (no tracing)
PYTHONPATH=build-fly/python_packages:. python tests/kernels/test_pa_mqa_logits_fp4.py \
    --batch 8 --ctx 65536 --num_iters 15 --num_warmup 3

# full ATT trace (production workload)
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_pa_mqa_logits_fp4.py \
        --batch 8 --ctx 65536 --num_iters 12 --num_warmup 3
```
