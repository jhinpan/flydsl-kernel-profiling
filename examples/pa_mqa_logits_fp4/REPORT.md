# FP4 MQA Logits — Rocprof v3 / ATT Instruction-Level Analysis

Commit: `9120078d35d7d232b3941ded5b76a1ca92329ef0` ("Optimize FP4 MQA logits pipelining")
Kernel: `pa_mqa_logits_fp4_kernel_0` (Q FP4, KV FP4, MFMA(Q_fp4, KV_fp4), gfx950 / MI350X)
Workspace: `/sgl-workspace/jin/fp4_mqa_probe/`

## Workload configurations measured

| config | role | batch | ctx | block_k | safe_chunks/CTA | total_CTAs | target_CTAs | CU/Utilization tail verdict | wall (us) | TFLOPS |
|---|---|---|---|---|---|---|---|---|---|---|
| small | prologue/cold-start | 4 | 8192 | 256 | 1 | 69 | 512 | underfilled by design | 4.40 | 65.1 |
| big | primary / saturation-validated | 32 | 131072 | 256 | 17 | 507 | 512 | validated: 32 waves, 512 occupancy rows, no long sampled-CU tail | 29.89 | 1189.9 |

The "small" config exposes the **prologue / cold-start** overhead. The current
`big` config is the primary trace: its persistent-grid schedule lands at
`total_CTAs=507` for a `parallel_unit_num=512` target, and the kept ATT dispatch
has dense sampled wave coverage rather than the previous underfilled
`total_CTAs=391` shape. See [`docs/att-viewer-guide.md`](../../docs/att-viewer-guide.md)
for the grid-sizing rule and viewer-reading checklist.

---

## 1. Headline wave-state breakdown

State of one CU's waves during the saturation-validated dispatch (ATT v3.0,
gfx950, sampled CU 1 across SE mask `0xf`, all 4 SIMDs):

| state | small (1 chunk/CTA) | big (17 chunks/CTA) | meaning |
|---|---|---|---|
| **EXEC** | 0.6 % | 0.1 % | instruction issue |
| **STALL** | 57.3 % | 29.0 % | `s_waitcnt` not yet retired |
| **WAIT** | 36.3 % | 49.7 % | operand-dep / VALU pipeline |
| **SLEEP** | 5.7 % | 21.2 % | power-down windows |

**The kernel spends almost none of a sampled wave's life issuing instructions.**
In the saturated run, the actionable time is split between explicit waitcnt
stalls and operand-dependency wait after MFMA/post-processing. The optimization
work should therefore focus on overlap and dependency ordering, not more raw
compute.

Per-instruction-class latency breakdown (primary trace, 32 sampled waves, 727
static instructions):

| class | count | latency cycles | stall cycles | lat % | stall % |
|---|---|---|---|---|---|
| **valu** | 520 | 782 436 | 259 592 | **52.8 %** | 32.8 % |
| **waitcnt** | 35 | 451 424 | 451 424 | 30.5 % | **57.0 %** |
| **mfma** | 32 | 92 868 | 29 712 | 6.3 % | 3.8 % |
| **vmem** | 31 | 67 108 | 42 380 | 4.5 % | 5.4 % |
| salu | 73 | 42 572 | 4 128 | 2.9 % | 0.5 % |
| lds (bperm) | 16 | 41 504 | 4 540 | 2.8 % | 0.6 % |
| sbranch | 4 | 2 332 | 0 | 0.2 % | 0.0 % |
| smem | 15 | 1 972 | 40 | 0.1 % | 0.0 % |

Read this as: the saturated trace is dominated by direct `s_waitcnt` stalls plus
VALU dependency latency after MFMA/post-processing. MFMA issue itself is still
only a small slice of attributed latency.

### 1a. Cross-check vs. FlyDSL `kernel-trace-analysis` skill

Running `hotspot_analyzer.py` on the same dispatch dir agrees on totals
(1.48 M cycles, 791.8 K stalls, 53.4 % stall ratio) and adds two breakdowns
worth recording:

**Stall by hardware-counter taxonomy** (sharper than my `class=waitcnt` lump):

| stall type | cycles | % of total stall |
|---|---|---|
| **VMEM-wait** (`s_waitcnt vmcnt(N)`) | 278.6 K | **35.2 %** |
| **other** (mostly VALU dependency latency) | 268.3 K | 33.9 % |
| LDS/SMEM-wait (`s_waitcnt lgkmcnt(N)`) | 172.8 K | 21.8 % |
| VMEM-load (load itself blocked at issue) | 39.4 K | 5.0 % |
| MFMA/FMA (RAW on MFMA result) | 29.7 K | 3.8 % |
| VMEM-store | 3.0 K | 0.4 % |
| SMEM | 40 | 0.0 % |

The headline shift after using a saturated shape: **VMEM wait is now the largest
stall bucket**, with VALU/MFMA dependency latency close behind. Kernarg
`lgkmcnt` remains visible but should no longer outrank loop-body overlap work.

**Register pressure / occupancy** (we didn't compute this manually):

```
Architecture:   gfx950 (CDNA4)
arch_vgpr:      ~95 (alloc 96) / 512 combined pool
accum_vgpr:     0 (not used)
occupancy:      5 waves/SIMD
-> 6 waves/SIMD requires total_vgpr ≤ 85
```

So the kernel is **at 5 waves/SIMD** and is **11 VGPRs away from 6 waves/SIMD**
(+20 % occupancy). That keeps VGPR reduction as a medium-priority candidate.

---

## 2. Top instruction-level hotspots

Sorted by total latency/stall over the kept traces — these are the optimisation
targets.

### 2a. Prologue / cold-start (small workload pays this 100 %)

| PC | latency | stall | instruction | what it's waiting on |
|---|---|---|---|---|
| 6736 | 4 088 small / 18 688 big | same | second batch of kernarg scalar loads (Q/W/strides) |
| 6432 | 3 036 small / 48 632 big | same | first batch of kernarg scalar loads (cta_info SRD) |
| 6576 | 368 small / 24 476 big | same | the `buffer_load_dwordx4` that fetches the packed cta_info |
| 7164 | 184 small / 15 048 big | same | drains KV-prefetch to ≤3 outstanding loads |

Context around PC 6432 (the very first stall):
```
s_load_dwordx2 s[8:9], s[0:1], 0x130   ; cta_info_ptr base addr
s_load_dword   s30,    s[0:1], 0x148   ; stride_out_batch
s_lshl_b32     s2,     s2, 4           ; pid * 16 (bytes per cta_info row)
s_mov_b32      s11,    0x27000         ; SRD hi-word (max_size encoding)
s_mov_b32      s10,    -1              ; SRD num_records
>>> s_waitcnt  lgkmcnt(0)              ; ← 3036 cycles in small / 48632 in big
s_and_b32      s9, s9, 0xffff          ; mask hi bits of cta_info base
v_mov_b32_e32  v1, s2                  ; lift pid*16 to VGPR
buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen  ; THE cta_info load
```

Then PC 6576 waits for that `buffer_load_dwordx4` (`s_waitcnt vmcnt(0)`), then PC 6736 waits again on a second wave of `s_load_dword` ops fetching Q/W/KV strides. **Three serial memory-barrier stalls fence the prologue's critical path.**

### 2b. Loop-body — biggest single in-loop stall

| PC | latency | stall | instruction | meaning |
|---|---|---|---|---|
| **9376** | **130 256** | **130 256** | `s_waitcnt vmcnt(5)` | late loop/epilogue drain of outstanding VMEM |
| **9388** | **53 528** | **53 528** | `s_waitcnt vmcnt(4)` | same drain sequence, one step later |
| 7540 | 26 820 | 26 820 | `s_waitcnt vmcnt(3)` | wait for prefetched KV to begin draining |
| 9016 | 26 276 | 26 276 | `s_waitcnt lgkmcnt(0)` | scalar/LDS-side wait in repeated loop body |
| 9000 | 25 288 | 25 288 | `s_waitcnt lgkmcnt(0)` | same family |

Context around PC 7540 (the loop's worst stall):
```
v_bfe_u32     v0,  v1, 8,  8           ; extract nt=1 scale byte from packed kvs
v_bfe_u32     v94, v1, 16, 8           ; extract nt=2 scale byte
v_lshrrev_b32 v53, 24, v1              ; extract nt=3 scale byte
>>> s_waitcnt vmcnt(3)                 ; ← 26 820 cycles in the saturated trace
v_mfma_scale_f32_16x16x128_f8f6f4 ...  ; nt=1 mfma issue (uses prefetched kv)
v_maximum3_f32 v1, v35, 0, 0           ; relu on prior nt's MFMA acc
```

This is the loop body issuing a fresh `_issue_nt_mfmas(nt=1)` whose operands came from `_prefetch_chunk`. The prefetch is *not* completing before the consume point — **the pipelining isn't deep enough**.

### 2c. Post-process VALU dependency chains

| PC | latency | stall | instruction |
|---|---|---|---|
| 7684 | 24 388 | 22 336 | `v_pk_mul_f32 v[0:1], v[0:1], v[18:19]`  (relu * w) |
| 8064 | 20 436 | 18 388 | `v_pk_mul_f32 v[26:27], v[26:27], v[18:19]` |
| 8908 | 16 932 | 14 884 | `v_pk_mul_f32 v[2:3], v[2:3], v[12:13]` |
| 8704 | 9 520 | 7 472 | `v_pk_mul_f32 v[0:1], v[0:1], v[18:19]` |
| 8480 | 4 844 | 2 796 | `v_pk_mul_f32 v[0:1], v[0:1], v[6:7]` |

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
>>> s_nop 2                                       ; 108 cycles in small / 384 in big
v_maximum3_f32 v51, v35, 0, 0                     ; reads v35 (just written by MFMA)
```

The `s_nop 2` is the compiler's hint that the MFMA result `v[34:37]` needs more
wait-states before VALU can read it. In the saturated trace the explicit nop is
small, but the following `v_pk_mul_f32` dependency chain is still large.
Re-ordering the post-process to start on the *first*-written MFMA target (`v22`)
would push this dependency out of the critical path.

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

> **Revised ordering after saturated recapture:** VMEM-wait is now the #1 stall
> bucket (35.2 %), with VALU/MFMA dependency latency close behind (33.9 %).
> Loop-body overlap work should outrank prologue-only fixes.

Priority ranking:
1. **B** — Earlier KV prefetch / VMEM drain overlap (cuts the 130K/53K `vmcnt` stalls; medium effort)
2. **A** — MFMA target re-order (cuts repeated `v_pk_mul_f32` dependency stalls; low effort)
3. **C** — Coalesce kernarg `s_load`s (important for small/prologue, lower for saturated primary)
4. **F** — VGPR reduction for 6 waves/SIMD (medium effort, +20 % occupancy headroom)
5. **D** — Defer scale extraction (small, low effort)
6. **E** — Wider KV vectorisation (small unless KV-BW bound; high effort)

### A. Re-order MFMA target registers vs. post-process consumption
**Likely impact: medium, easy.**
`_issue_nt_mfmas` (line 505) writes `accs[mi_idx]` in `mi_idx = 0,1,2,3` order, but `_post_process_nt` (line 536) consumes them in the same order. Because of register pressure assignment the compiler ends up with v22, v26, v30, v34 written in order and v34 read *immediately* in the relu (`v_maximum3_f32 v51, v35, 0, 0`). Reversing the `_post_process_nt` loop to `for mi_idx in [m_tiles-1, ..., 0]` (or rotating it so the *oldest* MFMA target is consumed first) would let the MFMA pipeline drain naturally — eliminating most of the `s_nop 2` / 3-5k-cycle `v_pk_mul_f32` stalls in §2c/2d.

### B. Issue next-chunk KV prefetch earlier in the loop body
**Likely impact: large for steady state, medium effort.**
The saturated trace's top stalls are `s_waitcnt vmcnt(5)` / `vmcnt(4)` at PCs
9376 / 9388, plus PC 7540's `vmcnt(3)`. They say outstanding KV/VMEM work still
reaches consume/drain points before enough independent work has run. The
chunk-loop body currently does (lines 670-695):
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

For repeated-kernel-launch workloads, the SQC kernarg cache helps after the
first dispatch, so this is mostly a *cold first dispatch* win. In the saturated
trace it is no longer the top bucket, but in the small config it is still the
dominant stall family.

### D. Defer scale extraction
**Likely impact: small, low effort.**
`_extract_kvs_scales` (line 491) extracts all NTPW=4 nt scales up-front from packed kvs i32s — three `v_bfe_u32` + one `v_lshrrev_b32_e32` immediately before the loop-prologue `s_waitcnt vmcnt(3)`. The comment claims this "decouples bfe from the mfma dep chain", but the trace shows those four extractions concentrate right at the critical path. Lazy per-nt extraction (extract inside `_issue_nt_mfmas` for *that* nt only, immediately before the MFMA that consumes it) gives the compiler more freedom to overlap the bfe with whatever VMEM the scheduler chose to issue.

### E. Reduce KV `buffer_load_dwordx4` count via wider vectorisation
**Likely impact: small unless KV bandwidth is the constraint.**
`_prefetch_chunk` issues `N_TILES_PER_WARP * k_tiles = 4 * 1 = 4` separate `buffer_load_dwordx4` for KV per warp per chunk. The 4 loads are at addresses that differ by `_kv_chunk_bytes (16)` per nt — they're not coalescable with a wider vec_width because they target different physical pages. The KVS already does this collapse (1 packed dword for all 4 nts). Doing the same for KV would require a host-side preshuffle that interleaves token bytes across nts — a deeper refactor, but cuts SQ_INSTS_VMEM_RD by 4×.

### F. Reduce VGPR pressure for 6 waves/SIMD
**Likely impact: medium (latent), medium effort.**
The official `hotspot_analyzer.py` reports `arch_vgpr = 96, accum_vgpr = 0, occupancy = 5 waves/SIMD`; **dropping to ≤ 85 VGPRs would put us at 6 waves/SIMD** (a 20 % occupancy gain on gfx950's combined 512-VGPR pool). Sources of the 96-VGPR footprint, in roughly descending size:

1. Hoisted Q (4 mi_idx × 8-dword v8i32) + hoisted weights (4 mi_idx × 4-float). These are loaded once and reused across chunks — moving them off the live set during post-process is hard but the *upper half of each v8i32* is poisoned (cbsz=4 ignores it; see `_pack_lo_i64x2_to_i32x8`), so the allocator could in principle treat them as v4i32 + 4 garbage VGPRs. Worth checking the dumped MIR.
2. The chunk-loop carry: `init_args = kv_pre + kvs_pre + phys_next_pre + nt0_init_scalars` carries `N_KV + N_KVS + NTPW + m_tiles*4 = 16 + 1 + 4 + 16 = 37` loop-carried VGPRs through every iteration. Splitting carry into "hot" (consumed this iter) and "cold" (next-iter prefetch) might let some get spilled to LDS during long VALU stretches; LDS is plentiful on gfx950 (160 KB).
3. Per-nt `accs[mi_idx]` × 4 mi_idx × 4 f32 = 16 f32 live during `_compute_chunk`. Already minimal.

Worth at least dumping the `--print-after-all` allocator output to see which 11+ VGPRs are easiest to retire.

---

## 5. Files in this analysis

| path | content |
|---|---|
| `/sgl-workspace/jin/fp4_mqa_probe/kernels/pa_mqa_logits_fp4.py` | kernel source (from commit) |
| `/sgl-workspace/jin/fp4_mqa_probe/tests/kernels/test_pa_mqa_logits_fp4.py` | test/bench harness |
| `/sgl-workspace/jin/fp4_mqa_probe/input_trace.yaml` | rocprofv3 config — small workload |
| `/sgl-workspace/jin/fp4_mqa_probe/input_trace_big.yaml` | rocprofv3 config — saturation-validated primary workload |
| `/sgl-workspace/jin/fp4_mqa_probe/prof/discover_v2_*` | kernel-discovery CSVs (rocprofv3 --stats) |
| `/sgl-workspace/jin/fp4_mqa_probe/prof/att_small_v2/ui_output_agent_32435_dispatch_197` | ATT trace, small workload |
| `/sgl-workspace/jin/fp4_mqa_probe/prof/att_big_v2/ui_output_agent_55351_dispatch_426` | ATT trace, saturation-validated primary workload |

## 6. How to re-run

```bash
cd /sgl-workspace/jin/fp4_mqa_probe

# baseline (no tracing)
PYTHONPATH=build-fly/python_packages:. python tests/kernels/test_pa_mqa_logits_fp4.py \
    --batch 32 --ctx 131072 --num_iters 12 --num_warmup 3

# ATT trace for the saturation-validated primary workload
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_pa_mqa_logits_fp4.py \
        --batch 32 --ctx 131072 --num_iters 12 --num_warmup 3
```
