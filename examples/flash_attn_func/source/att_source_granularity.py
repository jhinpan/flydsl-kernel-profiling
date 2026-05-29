#!/usr/bin/env python3
"""ATT code.json source-granularity metric (GPU-free). Lower coarse-sink share = finer."""
import json
import math
import os
import sys
from collections import defaultdict


def analyze(path, label):
    code = json.load(open(path))["code"]
    total = len(code)
    perline_w = defaultdict(float)  # (file,line) -> Hit+Stall
    perline_rows = defaultdict(int)
    mapped = 0
    fa_rows = 0
    for row in code:
        src = row[3]
        if not src or ":" not in src:
            continue
        mapped += 1
        fname, _, line = src.rpartition(":")
        base = os.path.basename(fname)
        if "flash_attn_func" in base:
            fa_rows += 1
        hit = row[6] if len(row) > 6 and isinstance(row[6], (int, float)) else 0
        stall = row[8] if len(row) > 8 and isinstance(row[8], (int, float)) else 0
        key = (base, int(line))
        perline_w[key] += hit + stall
        perline_rows[key] += 1

    total_w = sum(perline_w.values()) or 1.0
    items = sorted(perline_w.items(), key=lambda kv: kv[1], reverse=True)
    distinct = len(perline_w)
    top1 = items[0][1] / total_w if items else 0.0
    top2 = sum(w for _, w in items[:2]) / total_w if items else 0.0
    ent = -sum((w / total_w) * math.log(w / total_w) for _, w in items if w > 0)
    eff_lines = math.exp(ent)

    print(f"\n================  {label}  ================")
    print(f"  file: {path}")
    print(f"  total ISA rows           : {total}")
    print(f"  mapped rows              : {mapped}   flash_attn_func rows: {fa_rows}")
    print(f"  DISTINCT (file,line)     : {distinct}")
    print(f"  top-1 line weight share  : {top1:.4f}   (lower = finer)")
    print(f"  top-2 line weight share  : {top2:.4f}")
    print(f"  effective #lines exp(H)  : {eff_lines:.2f}   (higher = finer)")
    print(f"  --- top 12 lines by weight (Hit+Stall) ---")
    print(f"     {'file:line':<28} {'rows':>5} {'weight':>10} {'w%':>7}")
    for (f, ln), w in items[:12]:
        print(f"     {f+':'+str(ln):<28} {perline_rows[(f,ln)]:>5} {int(w):>10} {100*w/total_w:>6.1f}%")
    return dict(distinct=distinct, top1=top1, eff_lines=eff_lines, total_w=total_w)


if __name__ == "__main__":
    before, after = sys.argv[1], sys.argv[2]
    b = analyze(before, "BEFORE (unpatched)")
    a = analyze(after, "AFTER (source_loc fix)")
    print("\n================  SUMMARY  ================")
    print(f"  distinct mapped lines : {b['distinct']:>6}  ->  {a['distinct']:<6}  ({a['distinct']/max(b['distinct'],1):.1f}x)")
    print(f"  top-1 line weight     : {b['top1']:.4f}  ->  {a['top1']:.4f}")
    print(f"  effective #lines      : {b['eff_lines']:>6.2f}  ->  {a['eff_lines']:<6.2f}  ({a['eff_lines']/max(b['eff_lines'],1e-9):.1f}x)")
