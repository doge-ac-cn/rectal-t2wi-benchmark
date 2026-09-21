#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-file consistency anchors for the segmentation metrics (Table S8b).

Run from the repository root:  python code/nnunet/verify_hd95_consistency.py

Asserts that the shipped HD95 artifact (hd95_1000ep__*) describes the SAME mask
chain as the shipped per-case Dice (dice_per_patient_1000ep.csv), and that the
counts quoted in the manuscript (16 internal empty contours; 2 external
zero-Dice mislocalized cases) are recomputable from the shipped files.
"""
import csv, json, os, sys

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


hd = list(csv.DictReader(open(f"{PKG}/results/model_performance/hd95_1000ep__hd95_per_patient.csv", encoding="utf-8")))
dc = list(csv.DictReader(open(f"{PKG}/results/nnunet_segmentation/dice_per_patient_1000ep.csv", encoding="utf-8")))
sm = json.load(open(f"{PKG}/results/model_performance/hd95_1000ep__hd95_summary.json", encoding="utf-8"))
ds = json.load(open(f"{PKG}/results/nnunet_segmentation/dice_summary_1000ep.json", encoding="utf-8"))

check("row counts 1050/1050", len(hd) == 1050 and len(dc) == 1050, f"hd95={len(hd)} dice={len(dc)}")

# 1. same-mask-chain proof: per-case Dice identical at 4 dp (hd95 side stores 4dp)
dmap = {r["case_key"]: r["dice"] for r in dc}
ndiff, maxd = 0, 0.0
for r in hd:
    diff = abs(float(r["dice"]) - round(float(dmap[r["case_key"]]), 4))
    maxd = max(maxd, diff)
    ndiff += int(diff > 1e-9)
check("per-case Dice identical (4 dp)", ndiff == 0, f"max|diff|={maxd:.2e} ndiff={ndiff}")

# 2. summary anchors = manuscript Table S8b
check("RC_A n=952, dice 0.7233, HD95 5.002, ASSD 1.433",
      sm["RC_A_val"]["n"] == 952 and abs(sm["RC_A_val"]["dice_mean"] - 0.7233) < 1e-9
      and abs(sm["RC_A_val"]["hd95_median"] - 5.002) < 1e-9 and abs(sm["RC_A_val"]["assd_median"] - 1.433) < 1e-9)
check("RC_B n=82, dice 0.7238, HD95 5.188, ASSD 1.514",
      sm["RC_B"]["n"] == 82 and abs(sm["RC_B"]["dice_mean"] - 0.7238) < 1e-9
      and abs(sm["RC_B"]["hd95_median"] - 5.188) < 1e-9 and abs(sm["RC_B"]["assd_median"] - 1.514) < 1e-9)

# 3. empty-contour counts: 16 internal empty, 0 external empty, 2 external zero-Dice
ea = sum(1 for r in dc if r["dataset"] == "RC_A_val" and abs(float(r["rel_vol_err"]) + 1.0) < 1e-6)
eb = sum(1 for r in dc if r["dataset"] == "RC_B" and abs(float(r["rel_vol_err"]) + 1.0) < 1e-6)
zb = sum(1 for r in dc if r["dataset"] == "RC_B" and float(r["dice"]) == 0.0)
check("is_empty: 16 internal, 0 external", ea == 16 and eb == 0, f"int={ea} ext={eb}")
check("external zero-Dice (mislocalized, non-empty) = 2", zb == 2, f"n={zb}")

# 4. Dice summary agreement (all-cases internal mean incl. empty = 0.7113)
ga = [g for g in ds["summary"] if g["group"] == "RC_A_val"][0]
check("dice_summary RC_A mean 0.7113 (n=968)", abs(ga["dice_mean"] - 0.7113051226983427) < 1e-9 and ga["n"] == 968)

# 5. crosscheck field resolved (not a silent null)
check("summary carries explicit crosscheck result",
      sm.get("_crosscheck_vs_dice1000ep", {}).get("maxabsdice") == 0.0,
      str(sm.get("_crosscheck_vs_dice1000ep")))

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
