# nnU-Net automatic contour masks (1,000-epoch configuration) - NOT included

- The automatic tumor masks themselves are **not distributed in this repository**: they are
  binary masks derived from private imaging, i.e. patient-derived data, and are therefore
  withheld from public release (see the manuscript's Data availability statement for the
  access policy and request route).
- What is shipped instead, and what fully determines every segmentation number in the paper:
  `dice_summary_1000ep.json` (cohort-level), `dice_per_patient_1000ep.csv` and
  `surface_distances_hd95_1000ep.csv` (per-case Dice / HD95, keyed by `case_key`), and
  `oof_prediction_manifest.csv` (which fold produced each internal mask/prediction).
- Provenance for the withheld masks: produced strictly out-of-fold on the internal cohort
  (fold-k model scores only its own held-out validation fold; identical splits to every
  classification experiment) and by the five-model ensemble on the external cohort; the
  generating pipeline is `code/nnunet/` (train: `nnunet_1000ep.sh`, inference:
  `mask_predict.sh`, evaluation: `dice_eval.py`, `hd95.py`).
- Every table in the paper is reproducible from the per-case metrics and the training code
  alone; the masks are inputs to the downstream automation experiments, not to any
  reported statistic.
