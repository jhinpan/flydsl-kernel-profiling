# AGENTS.md — flydsl-kernel-profiling

Guide for any agent (human or LLM) asked to **profile a specific FlyDSL/ROCm
kernel and add it to this repo as a new example**.

## Mission of the repo

This is a hub for **decoded rocprofv3 trace bundles** of FlyDSL GPU kernels.
Each example under `examples/<kernel>/` is a complete, self-contained artifact
that someone on any machine (GPU not required) can `git clone` and:

- load `att_viewer/.../ui_output_agent_*/` into **AMD ATT Viewer** to inspect
  every ISA instruction, every stall, every wave
- load `compute_viewer/*_results.json` into **rocprof-compute-viewer** to inspect
  PMC counters / roofline / occupancy
- read `REPORT.md` to get the analysis writeup

The bundle is the artifact. The repo is not a tool, it's a **library of
inspectable kernels with paired analyses**.

## When you're invoked

Triggers — the user says something like:
- "profile the X kernel"
- "analyze the FP4 GEMM in commit Y"
- "look at why kernel Z is slow"
- "add a new example to this repo for kernel W"

Your output is **one new `examples/<kernel-name>/` directory + one row in the
top-level README's Examples table**. Not analysis text in chat — the artifact
goes into the repo.

## Required input

Ask the user (only if not provided):

| Input | Example | How to ask |
|---|---|---|
| Kernel name OR commit | `pa_mqa_logits_fp4` / `ROCm/FlyDSL@9120078` | "Which kernel? (name or commit URL)" |
| Workload shape | `--batch 8 --ctx 65536` | "Any specific shape, or use the kernel's default sweep?" |
| Target GPU arch | `gfx950` (default) / `gfx942` | usually inferable from `rocminfo` |

Don't ask if you can derive it. Be specific in your question — never just
"what kernel?"

## Output structure (canonical)

```
examples/<kernel-name>/
├── README.md          ← per-example: file layout + repro command + dispatch_<N> naming note
├── REPORT.md          ← per-example analysis writeup (template below)
├── att_viewer/
│   ├── small/ui_output_agent_<PID>_dispatch_<N>/   (diagnostic small/prologue shape)
│   └── big/ui_output_agent_<PID>_dispatch_<N>/     (larger candidate primary shape)
├── compute_viewer/
│   ├── {small,big}_results.json
│   ├── {small,big}_agent_info.csv
│   └── discover_*.csv                              (rocprofv3 --stats discovery)
└── source/
    ├── <kernel>.py                                 (kernel source from FlyDSL commit)
    ├── test_<kernel>.py                            (test harness)
    ├── input_trace.yaml / input_trace_big.yaml     (rocprofv3 config that produced the trace)
    └── hotspot_analyzer.py                         (verbatim copy from FlyDSL kernel-trace-analysis skill)
```

**Why two shapes (`small` vs `big`)?** They are diagnostic labels, not proof
that either workload is the best profiling configuration. Different stalls
dominate at different scales. A small workload (often 1 chunk/CTA) exposes the
cold prologue / kernarg-load chain; a saturation-validated workload exposes the
steady-state loop body. Do not call a trace "primary", "saturated", or
"steady-state" just because it lives under `big/`. The primary profiling config
must be sized from the kernel's tile/chunk/grid schedule and then validated:
`total_ctas` should be close to the target parallelism, and ATT Viewer should
not show an obvious underfilled tail in `Compute Unit` or `Utilization`.

## Workflow

### Step 1 — Verify environment

```bash
# All must succeed:
rocprofv3 --version              # need v1.1+
ls /opt/rocm/lib/librocprof-trace-decoder.so   # if missing, see Gotcha 1
rocminfo | grep -A2 'gfx'        # confirm target arch
# FlyDSL build:
PYTHONPATH=<flydsl_root>/build-fly/python_packages:<flydsl_root> python -c "import flydsl; import flydsl.compiler"
```

### Step 2 — Set up probe workspace

Pick a clean scratch dir (don't pollute the FlyDSL checkout). The probe needs
the kernel + test files importable as `kernels.<kernel>` and `tests.<kernel>`:

```bash
PROBE=/sgl-workspace/jin/<kernel>_probe
mkdir -p $PROBE/{kernels,tests/kernels}
touch $PROBE/{kernels,tests,tests/kernels}/__init__.py

# Pull kernel + test from the FlyDSL commit
git -C <flydsl_clone> show <commit>:kernels/<kernel>.py > $PROBE/kernels/<kernel>.py
git -C <flydsl_clone> show <commit>:tests/kernels/test_<kernel>.py > $PROBE/tests/kernels/test_<kernel>.py

# Symlink the FlyDSL build + shared test utilities
ln -sf <flydsl_root>/build-fly $PROBE/build-fly
ln -sf <flydsl_root>/tests/test_common.py $PROBE/tests/test_common.py
```

### Step 3 — Baseline run (no profiler)

Verify correctness and get a baseline TFLOPS number to compare against the
traced run. Do not let a no-debug baseline contaminate the trace compile cache:
either use the trace debug environment below for this run, or use a separate
baseline cache directory. Never rely on the default `~/.flydsl/cache` for a
trace that must have source mapping.

```bash
cd $PROBE
TRACE_CACHE=$PROBE/.flydsl_trace_cache
rm -rf "$TRACE_CACHE"
export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
export FLYDSL_RUNTIME_CACHE_DIR="$TRACE_CACHE"

PYTHONPATH=build-fly/python_packages:. python tests/kernels/test_<kernel>.py \
    <small-shape-flags> --num_iters 10 --num_warmup 3
# expect: cosine_sim ~1.0, some TFLOPS number
```

If correctness fails, **stop and report**. Don't profile a broken kernel.

### Step 3a — Debug-info cold-cache rule

ATT source mapping only works when FlyDSL compiles the HSACO with DWARF line
tables. `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is compile-time state, not a
post-processing option: rocprofv3 can read `.debug_line`, but it cannot add it
after a kernel was compiled. Therefore every command that can trigger JIT for
the traced kernel must use the same fresh debug cache:

```bash
TRACE_CACHE=$PROBE/.flydsl_trace_cache
rm -rf "$TRACE_CACHE"
export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
export FLYDSL_RUNTIME_CACHE_DIR="$TRACE_CACHE"
export PYTHONPATH=build-fly/python_packages:.
```

This applies to discovery and ATT capture. The existing FlyDSL
`capture-kernel-trace` skill sets `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` for the
capture step, but it does not clear or isolate FlyDSL's JIT cache and its
discovery command can compile the kernel first without debug info. If discovery
or a baseline run already cached a no-debug HSACO, the later capture may show
0 source-mapped instructions even though the rocprofv3 YAML has `AsmDebug:
True`.

Use a fresh `FLYDSL_RUNTIME_CACHE_DIR` for the trace session. Do not delete a
shared `~/.flydsl/cache` unless the user explicitly asks for that.

### Step 4 — Kernel discovery

The JIT-compiled kernel name may differ from the Python function name. Find
it with `--stats`:

```bash
mkdir -p prof
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_CACHE_DIR="$TRACE_CACHE" \
rocprofv3 --stats --kernel-trace -f csv -o prof/discover -- \
    python tests/kernels/test_<kernel>.py <small-shape-flags> --num_iters 8 --num_warmup 2
grep -v "at::native\|rocclr\|rocprim\|Cijk" prof/discover_kernel_stats.csv
# pick the FlyDSL kernel — usually contains the kernel function name
```

Record the exact kernel name (e.g. `pa_mqa_logits_fp4_kernel_0`); you'll
need it as `kernel_include_regex` in the YAML.

### Step 5 — Size the workload grid before capture

Before capturing the larger / candidate primary trace, calculate whether the
workload actually fills the target parallelism. Do not assume that `small`,
`big`, a larger `ctx`, or a production-looking shape is enough. For persistent
kernels like `pa_mqa_logits_fp4`, the host schedule is derived from
tiles/chunks:

```text
chunks_per_batch = ceil(context_len / block_k)
safe_chunks_per_cta = smallest s such that:
    sum(ceil(chunks_per_batch / s)) * next_n <= parallel_unit_num
total_ctas = sum(ceil(chunks_per_batch / safe_chunks_per_cta)) * next_n
```

The test harness prints:

```text
schedule: parallel_unit=<target> num_warps=<N> safe_chunks_per_cta=<S> total_ctas=<G>
```

Use this before trace capture:

- For a prologue trace, a small underfilled grid is fine; it is intentionally
  cold-start/prologue-biased.
- For the primary steady-state trace, `total_ctas` should be close to the target
  parallelism. With the default `parallel_unit_num=512`, prefer a shape that
  lands near 512 CTAs, not 69 or 391.
- If `total_ctas` is too low, increase `batch`, increase effective `ctx`, choose
  a shape with more chunks/tiles, or adjust `parallel_unit_num`.
- After capture, confirm in ATT Viewer that `Compute Unit` and `Utilization` do
  not show an obvious underfilled tail where only a few waves remain.

Record `block_k`, `safe_chunks_per_cta`, `total_ctas`, and
`parallel_unit_num` in `REPORT.md`, plus a short verdict for `Compute Unit` and
`Utilization` tail checks. If a trace is intentionally underfilled, label it as
prologue/cold-start or underfilled loop-body coverage and do not use it as the
main optimization ranking signal. If both captured shapes are underfilled,
recapture a saturation-validated primary trace before writing the final ranking.

### Step 6 — Capture two ATT traces

Write `input_trace.yaml` (small/prologue shape) and `input_trace_big.yaml`
(larger candidate primary shape; only call it saturated after Step 5 passes).
Template:

```yaml
GlobalParameters:
  KeepBuildTmp: True
  AsmDebug: True
jobs:
    -
        kernel_include_regex: "<EXACT_KERNEL_NAME>"
        kernel_iteration_range: "[<N_skip>, [<M_trace>-<M_trace>]]"
        output_file: out
        output_directory: prof/att          # or prof/att_big
        output_format: [json, csv]
        truncate_kernels: true
        sys_trace: false
        advanced_thread_trace: true
        att_target_cu: 1
        att_shader_engine_mask: "0xf"
        att_simd_select: "0xf"
        att_buffer_size: "0x6000000"        # 96 MB; bump to 0xC000000 if truncated
pmc:
  - SQ_INSTS_VALU
  - SQ_INSTS_MFMA
  - SQ_INSTS_VMEM
  - SQ_WAVES
  - SQ_WAIT_INST_LDS
  - SQ_LDS_BANK_CONFLICT
  - GRBM_GUI_ACTIVE
```

Pick `kernel_iteration_range` so `N_skip` skips warmup launches and `M_trace`
hits steady-state. With test default `num_warmup=3, num_iters=15`, the launch
sequence is: 1 (correctness) + 3 (warmup) + 15 (perftest) → 19 total. A safe
choice: `"[6, [8-8]]"` (skip 0-5, capture iter 8).

Run both:

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_CACHE_DIR="$TRACE_CACHE" \
PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace.yaml -- python tests/kernels/test_<kernel>.py <small-flags> ...

FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_CACHE_DIR="$TRACE_CACHE" \
PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_<kernel>.py <primary-shape-flags> ...
```

### Step 7 — Cleanup

Each capture produces `prof/att*/ui_output_agent_<PID>_dispatch_<N>/` folders.
Some are empty shells; some are duplicates. Filter:

```bash
for d in prof/att*/ui_output_agent_*/; do
    nwaves=$(ls $d/se*_sm*_*.json 2>/dev/null | wc -l)
    ins=$(python -c "import json; print(len(json.load(open('$d/code.json'))['code'] or []))")
    echo "$d  waves=$nwaves  ins=$ins"
done
# Keep one folder per workload with waves>0 and ins>0. Delete the rest.
```

(See Gotcha 3 for why empty shells exist.)

### Step 7a — Verify source mapping

Source mapping is required unless the report explicitly says why it is
unavailable. Check both `code.json` and the captured code object:

```bash
python3 - <<'PY'
import glob, json
for p in glob.glob("prof/att*/ui_output_agent_*/code.json"):
    code = json.load(open(p))["code"]
    mapped = sum(1 for row in code if len(row) > 3 and row[3])
    print(f"{p}: mapped={mapped}/{len(code)}")
PY

/opt/rocm/llvm/bin/llvm-dwarfdump --debug-line prof/att_big/out_gfx*_code_object_id_*.out | head -80
```

If `mapped=0` for the FlyDSL kernel and the relevant code object's
`.debug_line` is empty, recapture from a fresh `FLYDSL_RUNTIME_CACHE_DIR`.
Do not publish the trace as source-mapped.

### Step 8 — Analyze with `hotspot_analyzer.py` (DO NOT roll your own)

Copy the analyzer from FlyDSL's `kernel-trace-analysis` skill:

```bash
cp <flydsl_root>/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py .
python hotspot_analyzer.py prof/att_big/ui_output_agent_*/ --topk 15 --mode both
python hotspot_analyzer.py prof/att_big/ui_output_agent_*/ --topk 5  --mode src --detail --context 4
```

Capture from its output:

1. **Stall taxonomy** — bucketed as LDS/SMEM-wait (`lgkmcnt`), VMEM-wait
   (`vmcnt`), VMEM-load, MFMA/FMA, barrier. Note which bucket is #1; this
   shapes the ranking in §4 of the report.
2. **Top-15 hotspot PCs** — each with `(stall_cycles, stall_type, ASM, source_loc)`.
   Treat hot `s_waitcnt` PCs as consume-side dependency drains. Treat hot
   `s_nop` clusters as compiler-inserted scheduling bubbles, not barriers, but
   still record them because they often mark missing overlap around the same
   producer/consumer boundary.
3. **Register pressure & occupancy** — `arch_vgpr`, `accum_vgpr`,
   `waves/SIMD`, and "next occupancy step requires VGPR ≤ X". The skill
   knows the right formula per arch (gfx942 split / gfx950 combined pool).

If `source_loc` is `<unknown>` everywhere, see Gotcha 4 — debug info wasn't
plumbed through. The report can still go ahead, but mark the PC→source
mapping in §3 as manually derived from PC ranges.

### Step 9 — Write REPORT.md (template)

Required sections, in order:

```
# <Kernel name> — Rocprof v3 / ATT Instruction-Level Analysis

Commit: <link>
Kernel: <JIT name> (<arch>)
Workspace: <local probe path>

## Workload configurations measured
[Table: shape | role | TFLOPS | duration | block_k | safe_chunks_per_CTA |
 total_CTAs | target_CTAs | CU/Utilization tail verdict]

## 1. Headline wave-state breakdown
[Table: EXEC/STALL/WAIT/SLEEP %]
[Per-instruction-class latency table]

### 1a. Cross-check vs. hotspot_analyzer.py
[Stall taxonomy bucketed by hardware counter]
[Register pressure / occupancy]

## 2. Top instruction-level hotspots
### 2a. Prologue / cold-start
### 2b. Loop-body
### 2c. Post-process dependency chains
[For each: PC, latency, instruction, source-disassembly context]

## 3. PC → Python source mapping
[Table: PC range | source region | what runs there]

## 4. Optimisation recommendations
[Numbered, ranked by expected impact, each with:
 - effort level
 - root cause from §2
 - concrete code change
 - estimated gain (if measurable)]
[Top of section: priority ranking summary]

## 5. Files in this analysis
[Paths to source/yaml/trace dirs]

## 6. How to re-run
[Exact commands]
```

Rank order in §4 should follow the §1a stall taxonomy: whichever bucket is
biggest should get the top-priority recommendation. Don't just transcribe;
**use the data to drive the priority**.

### Step 10 — Bundle into repo

```bash
EX=examples/<kernel-name>
mkdir -p $EX/att_viewer/{small,big} $EX/compute_viewer $EX/source
cp -r prof/att/ui_output_agent_*    $EX/att_viewer/small/
cp -r prof/att_big/ui_output_agent_* $EX/att_viewer/big/
cp prof/att/out_results.json     $EX/compute_viewer/small_results.json
cp prof/att/out_agent_info.csv   $EX/compute_viewer/small_agent_info.csv
cp prof/att_big/out_results.json $EX/compute_viewer/big_results.json
cp prof/att_big/out_agent_info.csv $EX/compute_viewer/big_agent_info.csv
cp prof/discover_*.csv           $EX/compute_viewer/
cp kernels/<kernel>.py           $EX/source/
cp tests/kernels/test_<kernel>.py $EX/source/
cp input_trace*.yaml             $EX/source/
cp hotspot_analyzer.py           $EX/source/
# Then write README.md and REPORT.md into $EX/
```

### Step 11 — Update top-level README

Add a row to the **Examples** table in the top-level `README.md` linking
to your new example. Format: `| folder | kernel | source | one-liner |`.
Lead the one-liner with the headline finding (e.g. "profoundly stall-bound:
47 % VALU latency, 34 % `s_waitcnt`, only 0.3 % EXEC").

### Step 12 — Commit

One commit per example, with a message body that includes the headline
finding so `git log` is useful:

```
First example: <kernel> ATT + counter trace

<2-3 sentences describing what was captured and the top finding>
```

## Gotchas — the things we learned the hard way

### 1. `librocprof-trace-decoder.so` may be missing from `/opt/rocm/lib`

Symptom: `rocprofv3 -i input.yaml` says "rocprof-trace-decoder library path
not found".

Fix: locate it in any rocm install (often inside a docker overlay) and
symlink:
```bash
find / -name 'librocprof-trace-decoder.so' 2>/dev/null
cp /path/to/found/librocprof-trace-decoder.so /opt/rocm/lib/
```
Or use the official installer (see FlyDSL `kernel-trace-analysis/SKILL.md`
Step 3).

### 2. `dispatch_<N>` is the process-wide HSA dispatch counter

It includes torch utility kernels (`vectorized_elementwise_kernel`,
`reduce_kernel`, etc.) that don't match `kernel_include_regex`. So if your
kernel ran 12 times you may see ATT folders for `dispatch_234, dispatch_236,
...` with non-matching dispatches occupying the odd numbers. Not a bug.

### 3. Empty-shell dispatch folders

Some `ui_output_agent_<PID>_dispatch_<N>/` folders contain only
`code.json/filenames.json/occupancy.json/realtime.json` with zero waves and
empty code. These are placeholder slots rocprofv3 reserves before ATT
collection actually starts. Detect via `waves=0` and delete.

### 4. `kernel_iteration_range "[N, [M-K]]"` inner range is a lower bound

v1.1 captures iteration M and **also** the next matching dispatch if the ATT
buffer has slack. With `[8-8]` you reliably get TWO captures. Both are full
and valid; pick either. Two captures function as a noise-floor sanity check
(should be within a few percent of each other).

### 5. Source mapping can silently reuse a no-debug HSACO

Symptom: every `code.json` entry has `source_loc = ""`; `snapshots.json` and
`source_0_*.py` files don't exist; `hotspot_analyzer.py` shows `<unknown>`
for every line.

First suspect cache contamination: discovery or a no-profiler baseline may have
compiled and cached the kernel before `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` was
set. Recapture with a fresh `FLYDSL_RUNTIME_CACHE_DIR` and the debug env set
for every JIT-triggering command.

Verify on the source machine:
```bash
# Check the captured code object for .debug_line.
/opt/rocm/llvm/bin/llvm-dwarfdump --debug-line prof/att_big/out_gfx*_code_object_id_*.out | head -80
```

If `.debug_line` is still empty after a confirmed cold debug-cache capture,
then the FlyDSL compile pipeline is not honoring the env var for that kernel.
Report that in the example's README under "Source mapping note" and proceed
without claiming source-mapped ATT.

## Anti-patterns — don't do these

- **Don't re-implement `hotspot_analyzer.py` in Python by hand.** It's 395
  lines in the FlyDSL skill, it auto-detects arch, it has the right stall
  taxonomy. Copy and use.
- **Don't capture only one workload shape.** Small and big expose different
  bottlenecks; one-shape analyses miss whichever stall the shape doesn't
  exercise.
- **Don't call an underfilled grid "steady-state."** `small` and `big` are trace
  labels, not validation results. If `total_ctas` is far below the target
  parallelism, or `Compute Unit` / `Utilization` shows a tail where only a few
  waves remain, the trace may be useful for prologue or loop-body diagnosis, but
  it should not drive the main optimization ranking. Resize the workload and
  recapture.
- **Don't paste full ATT trace into the report.** REPORT.md should be the
  *analysis*; raw trace lives under `att_viewer/`.
- **Don't bury the headline.** The §1 wave-state breakdown should
  immediately tell the reader whether this kernel is EXEC-bound,
  STALL-bound, or WAIT-bound.
- **Don't skip §4 priority ranking.** A list of optimization candidates
  without ranking is just observations.

## Tools you'll use

- [`rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/) v1.1+ — capture
- `hotspot_analyzer.py` — analyze (from FlyDSL `.claude/skills/kernel-trace-analysis/`)
- AMD ATT Viewer — for the human / user to inspect after upload
- `rocprof-compute-viewer` — for counter aggregates

The agent's job stops at producing the bundle; the user opens the viewers.

## See also

- This repo's `README.md` — what the artifacts are and how to open them
- This repo's `docs/att-viewer-guide.md` — how to read ATT Viewer /
  rocprof-compute-viewer tabs, columns, and workload-sizing signals
- FlyDSL `.claude/skills/capture-kernel-trace/SKILL.md` — verbatim capture recipe
- FlyDSL `.claude/skills/kernel-trace-analysis/SKILL.md` — analysis patterns +
  MFMA latency table + register pressure formulas
- Existing `examples/pa_mqa_logits_fp4/` — reference for layout, naming, and
  report structure
