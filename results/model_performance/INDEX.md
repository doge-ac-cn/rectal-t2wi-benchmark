# Model performance results index

Every directory here is one training arm: `summary*.json` = per-fold internal-validation and
external AUC (+ ensemble, timing), `ext_preds*.npz` = per-fold external predictions with
`ext_labels` (external cohort, n=82; subset arms carry their own subset labels), `fold*_hist.json`
= per-epoch training history. File names map to the manuscript as follows (arm__<source-file>):

| Directory pattern | Manuscript location |
|---|---|
| `resnet50_d2_roi`, `resnet50_d25_roi`, `resnet50_d5_roi`, `resnet50_d7_roi`, `resnet50_d2_rect`, `resnet50_d25_rect` | Table 1 (shallow grid); Table S1 (2D/2.5D rows) |
| `m{0,5,10}_{32,48,64}std` | Table S1 (3D rows of the 15-variant grid); `m10_64std` is also the Table 1 "3D 64-cube, 10 mm margin" row (0.756) |
| `resnet50d3_pt_std`, `resnet50d3_scratch_std` | Table 1 3D tumor-ROI 48-cube rows (ImageNet-inflated 0.785 / from-scratch 0.761); Table S5b |
| `resnet50d3nat_pt`, `resnet50d25nat_roi` | Table S2 (native-slice arms: 3D 0.815, 2.5D 0.858); Table S10c (three-seed) |
| `resnet50_ziso_*`, `resnet50_z3_*`, `resnet50_z6_*`, `resnet50_ip48_*`, `resnet50d3ziso*` | Table 2 (anisotropy intervention); Table S2 |
| `resnet18/34/50/101/152_d25_roi`, `efficientnet_*`, `convnext_*`, `vit_*` | Table S5 (backbone sweep); per-fold detail in Section S9 |
| `convnext_base_mtl*`, `resnet50_mtl*`, `unet3d_*` | Table S6 (multi-task ablation) |
| `*_s43`, `*_s44`, `d3nat_m0_s4*`, `ziso_m0_48_s4*`, `seed_sensitivity` | Table S10b/S10c (three-seed replication) |
| `delong_matrix` | Table S10a (21-pair Holm family) |
| `stats_bundle_1000ep` | Table S13-a/S13-b (TOST dual margin, MDD; 1,000-epoch chain — matches the manuscript exactly) |
| `*`, `radiomics_ensemble` | Tables 3-4, S13-d (automation, fusion, auto1000 chains) |
| `vendor`, `vendor_1000`, `resnet50_d25roi_vend_*` | Tables S3a–S3c |
| `t2vt3` | Tables S4a–S4c |
| `contour_perturbation*`, `perturb_ensemble_1000*` | Table S7; Figures S1–S2 (Section S12). Locked-model numbers use `perturb_ensemble_1000*` |
| `hd95_1000ep`, `nnunet_1000ep` | Table S8b (segmentation metrics beyond Dice) |
| `cost` | Table S5c |
| `imputation_sensitivity` | Section S7 |
| `dose_response` | Table S1 (stacking-depth dose-response) |

Notes:
- `results/REBUILD_LOG.md` lists the few arm summaries regenerated from per-fold predictions
  during packaging (originals were truncated by a later overwrite in the internal repo);
  the regeneration method was validated bit-exact on intact arms and no manuscript value changed.
  It also documents the v1.3.1 artifact-chain correction (HD95 / stats bundle moved from the
  250-epoch to the 1,000-epoch chain; superseded files kept under `legacy/`).
- `resnet50_d3_m0_48` is the 4x-wide (737.6M parameter) capacity-control run cited in
  Supplementary S5b (external 0.783±0.016); its checkpoints predate a layout fix that was
  verified against the reference implementation, and every mainline 3D row in the paper uses
  the standard-layout re-runs (`resnet50d3_pt_std`, `resnet50d3_scratch_std`).
  The remaining earlier-generation directories (`resnet50_d3_m0_48_fs34`, `_pretrained`,
  `resnet50d3_m10_64`) are superseded intermediates not referenced by any manuscript number
  and are still excluded.
- `mdd_repr` is the bootstrap MDD for the representation-family ensemble contrasts
  (Supplementary Table S13-b rows "2.5D vs best 3D"); protocol identical to the automation-family
  MDD (B=10,000, seed 42).
- The pooled three-seed analysis script lives in the internal analysis repo; its per-seed
  inputs (the seed-suffixed arms here) fully determine Table S10c.
