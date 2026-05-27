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
│   ├── small/ui_output_agent_<PID>_dispatch_<N>/   (1 chunk/CTA shape — prologue-bound)
│   └── big/ui_output_agent_<PID>_dispatch_<N>/     (≥3 chunks/CTA — steady-state)
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

**Why two shapes (`small` vs `big`)?** Different stalls dominate at different
scales. A small workload (1 chunk/CTA) exposes the cold prologue / kernarg-load
chain; a large workload (≥3 chunks/CTA) exposes the steady-state loop body.
Both are valuable; capture both.

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
traced run:

```bash
cd $PROBE
PYTHONPATH=build-fly/python_packages:. python tests/kernels/test_<kernel>.py \
    <small-shape-flags> --num_iters 10 --num_warmup 3
# expect: cosine_sim ~1.0, some TFLOPS number
```

If correctness fails, **stop and report**. Don't profile a broken kernel.

### Step 4 — Kernel discovery

The JIT-compiled kernel name may differ from the Python function name. Find
it with `--stats`:

```bash
mkdir -p prof
rocprofv3 --stats --kernel-trace -f csv -o prof/discover -- \
    python tests/kernels/test_<kernel>.py <small-shape-flags> --num_iters 8 --num_warmup 2
grep -v "at::native\|rocclr\|rocprim\|Cijk" prof/discover_kernel_stats.csv
# pick the FlyDSL kernel — usually contains the kernel function name
```

Record the exact kernel name (e.g. `pa_mqa_logits_fp4_kernel_0`); you'll
need it as `kernel_include_regex` in the YAML.

### Step 5 — Capture two ATT traces

Write `input_trace.yaml` (small shape) and `input_trace_big.yaml` (production
shape). Template:

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
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace.yaml -- python tests/kernels/test_<kernel>.py <small-flags> ...

FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=build-fly/python_packages:. \
    rocprofv3 -i input_trace_big.yaml -- python tests/kernels/test_<kernel>.py <big-flags> ...
```

### Step 6 — Cleanup

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

### Step 7 — Analyze with `hotspot_analyzer.py` (DO NOT roll your own)

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
3. **Register pressure & occupancy** — `arch_vgpr`, `accum_vgpr`,
   `waves/SIMD`, and "next occupancy step requires VGPR ≤ X". The skill
   knows the right formula per arch (gfx942 split / gfx950 combined pool).

If `source_loc` is `<unknown>` everywhere, see Gotcha 4 — debug info wasn't
plumbed through. The report can still go ahead, but mark the PC→source
mapping in §3 as manually derived from PC ranges.

### Step 8 — Write REPORT.md (template)

Required sections, in order:

```
# <Kernel name> — Rocprof v3 / ATT Instruction-Level Analysis

Commit: <link>
Kernel: <JIT name> (<arch>)
Workspace: <local probe path>

## Workload configurations measured
[Table: shape | TFLOPS | duration | chunks_per_CTA]

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

### Step 9 — Bundle into repo

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

### Step 10 — Update top-level README

Add a row to the **Examples** table in the top-level `README.md` linking
to your new example. Format: `| folder | kernel | source | one-liner |`.
Lead the one-liner with the headline finding (e.g. "profoundly stall-bound:
47 % VALU latency, 34 % `s_waitcnt`, only 0.3 % EXEC").

### Step 11 — Commit

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

### 5. `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` doesn't always plumb through

Symptom: every `code.json` entry has `source_loc = ""`; `snapshots.json` and
`source_0_*.py` files don't exist; `hotspot_analyzer.py` shows `<unknown>`
for every line.

The env var should make the FlyDSL JIT pass DWARF line-table info into the
hsaco. If it didn't work, verify on the source machine:
```bash
# Find the JIT artifact
ls /tmp/flydsl_* /tmp/kernel_trace_output/*build_tmp* 2>/dev/null
# Check for .debug_line section
llvm-dwarfdump --debug-line <kernel>.hsaco | head -30
```

If `.debug_line` is empty, the FlyDSL compile pipeline isn't honouring the
env var — that's a FlyDSL-side bug, not a rocprofv3 issue. Report in the
example's README under "Source mapping note" and proceed.

## Anti-patterns — don't do these

- **Don't re-implement `hotspot_analyzer.py` in Python by hand.** It's 395
  lines in the FlyDSL skill, it auto-detects arch, it has the right stall
  taxonomy. Copy and use.
- **Don't capture only one workload shape.** Small and big expose different
  bottlenecks; one-shape analyses miss whichever stall the shape doesn't
  exercise.
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
- FlyDSL `.claude/skills/capture-kernel-trace/SKILL.md` — verbatim capture recipe
- FlyDSL `.claude/skills/kernel-trace-analysis/SKILL.md` — analysis patterns +
  MFMA latency table + register pressure formulas
- Existing `examples/pa_mqa_logits_fp4/` — reference for layout, naming, and
  report structure
