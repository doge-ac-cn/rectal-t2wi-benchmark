# Summary regeneration log (package v1.3)

An independent reproducibility audit (2026-09-13) found that a few `summary.json` files
were truncated by a later overwrite in the internal analysis repo (fold records lost),
while the per-fold external predictions and training histories remained complete.
Affected summaries were regenerated directly from those primary artifacts:

- external AUC mean/SD = per-fold external AUCs, population SD (ddof=0),
  identical to the training scripts' own summary convention;
- internal validation AUC mean/SD = best-epoch validation AUC per fold from the
  shipped per-epoch histories.

The regeneration procedure was validated bit-exact against four intact arms
(resnet50_d25_roi, resnet50_d2_roi, resnet50_ziso_d25_roi, d3nat_m0_s42) before use. Manuscript values were unchanged.

Regenerated: `resnet50_d3_m0_48`, `resnet50d25nat_roi`, `resnet50d3_pt_std`, `resnet50d3_scratch_std`, `resnet50d3nat_pt`

---

# v1.3.1 — artifact chain correction (2026-09-21)

An external audit found that the shipped HD95/ASSD artifacts and the `stats_bundle__*` files
dated from the **250-epoch** auto-contour chain, while the manuscript segmentation numbers
(Table S8b: Dice 0.711/0.724, HD95 5.0/5.2 mm, ASSD 1.4/1.5 mm) and the automation statistics
(Table S13-a/b) come from the **1,000-epoch** chain. The matching 1,000-epoch HD95 run existed
internally but had not been packaged. Corrections:

- Added `hd95_1000ep__hd95_per_patient.csv` and `hd95_1000ep__hd95_summary.json`
  (source: the internal 1,000-epoch surface-distance run; per-case Dice verified against
  `dice_per_patient_1000ep.csv` — 1,050/1,050 cases identical at 4 decimal places).
- Added `nnunet_segmentation/surface_distances_hd95_1000ep.csv` (same source).
- Added `stats_bundle_1000ep__tost_dual_margin.json` and `stats_bundle_1000ep__mdd.json`;
  values are bit-identical to the manuscript Table S13-a/b source artifacts
  (the automation-chain bootstrap outputs `paper_numbers.json` / `mdd_power_1000.json`).
- `dice_eval_1000ep.py` outputs and the shipped per-case Dice files/JSON now carry an
  `is_empty` column (`rel_vol_err = −1.000` ⇔ empty prediction): 16 internal cases,
  0 external (2 external cases have Dice = 0 through complete tumour miss, not emptiness).
- The superseded 250-epoch artifacts were moved unmodified to `legacy/`
  (`hd95_250ep/`, `stats_bundle_250ep/`) for provenance.
- `hd95.py` no longer silently writes a null crosscheck: failures now record an explicit
  `_crosscheck_error` reason, and `--dice-csv` allows chain-matched crosschecking.
- `INDEX.md` manuscript-table mapping was re-anchored to the final 14-section supplement
  numbering (e.g. segmentation metrics → Table S8b, perturbation → Table S7,
  vendor → Tables S3a–c, T2vT3 → Tables S4a–c, cost → Table S5c).

No manuscript value changed.
