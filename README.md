# rectal-t2wi-benchmark

Code and process artifacts for: *"Impact of automated contouring and acquisition geometry on
radiomics and deep learning for preoperative T-staging of rectal cancer on T2WI: a fully
automated multicenter imaging-informatics benchmark."*

The repository contains (1) the full training/evaluation pipeline - crop construction, nnU-Net
segmentation, 2D/2.5D/3D deep learning, radiomics with nested-CV ensembles, and the paired
statistics - and (2) the process artifacts needed to audit every number in the paper: fixed
splits, per-fold and three-seed results for every arm, per-case segmentation metrics, and the
out-of-fold prediction manifest.

## De-identification

Case identifiers are replaced everywhere by `case_key = sha256("rcbench_v1|<pseudonym id>")[:12]`.
The salt is fixed and published here so that users with the original (pseudonymized) cohort can
reproduce the join deterministically; no identifier, DICOM header, or image pixel data is
distributed. The imaging and the derived nnU-Net masks are patient-derived data and are not
redistributable here; they are described in the manuscript's Data availability statement, and
every paper table is reproducible from the shipped per-case metrics and training code alone.

## Repository layout

```
code/
  prep/      crop construction from the resampled volumes (15-variant grid, geometry audit)
  nnunet/    nnU-Net training, strict out-of-fold inference, Dice/HD95 evaluation
  dl/        2D/2.5D and 3D training loops (ResNet/EfficientNet/ConvNeXt/ViT; MONAI ResNet50 3D)
  dl/dltrain/  shared data/split utilities imported by the training scripts
  dltrain/   the same utilities, importable as <repo>/code/dltrain (used by code/stats and
             code/radiomics; a duplicated copy so both import styles work unchanged)
  radiomics/ PyRadiomics extraction, L1 screening, nested-CV five-family ensemble
  stats/     verified DeLong (Sun-Xu fast algorithm), 21-pair Holm family, TOST/MDD bundle,
             representation-family bootstrap MDD (mdd_repr.py)
  figures/   manuscript table generation
data/
  case_index.csv            hashed case list with label / split / pT
  vendor_map.csv            hashed acquisition metadata (vendor, model, spacing, thickness bin)
  splits_fixed_5fold.json   the five folds used by EVERY experiment (968 internal cases)
results/
  nnunet_segmentation/      per-case Dice, surface distances, OOF manifest (masks are
                            patient-derived data and are NOT distributed; see README_masks.md)
  model_performance/        every arm: summary.json (per-fold + ensemble AUC), ext_preds.npz,
                            training histories; see INDEX.md for the paper-table mapping
  REBUILD_LOG.md            provenance of the few regenerated arm summaries (v1.1)
```

## Environment

Python 3.11, PyTorch 2.x (CUDA 12.x), single RTX 3090 (24 GB). See `requirements.txt`.
nnU-Net uses the official `nnunetv2` package; the classifier pipeline is plain PyTorch +
timm (+ MONAI for the 3D ResNet50).

## Reproduction (paper tables)

1. **Crops** - `code/prep/prepare_datasets.py` builds the 15-variant grid
   (`d2_*, d25_*, d3_m*_3[268]`) on the 0.7188 x 0.7188 x 1.5 mm grid.
2. **Segmentation** - train nnU-Net (`code/nnunet/nnunet_1000ep.sh`, official 1,000-epoch
   config, five folds), then predict strictly out-of-fold on the internal cohort
   (`mask_predict.sh` protocol) and by five-model ensemble externally; Dice/HD95:
   `dice_eval_1000ep.py`, `hd95.py` -> Table S8b.
3. **Deep learning** - `code/dl/train_2d25.py --variant <v> --model <m>` and
   `train_3d.py --variant d3_m0_48 --model <scratch|imagenet|medicalnet|swinunetr>`;
   fixed five-fold splits, seed 42 (+43/44 for the replicated arms). Table 1, Table 2, S1, S2,
   S2b, S9, S10b-c.
4. **Radiomics** - `code/radiomics/radiomics_extract.py` (1,316 features, original+LoG+wavelet),
   `radiomics_ensemble.py` (nested-CV five-family ensemble; identical protocol per contour
   source). Tables 3-4, S13-d.
5. **Statistics** - `code/stats/` (DeLong 21-pair Holm family -> Table S10a; TOST at 0.05/0.10
   margins and minimum detectable differences -> Table S13-a/b; `mdd_repr.py`, bootstrap
   MDD for the representation-family ensemble contrasts -> Table S13-b rows "2.5D vs best 3D").
6. **Automation** - repeat 3-4 with automatic contours (auto masks from step 2; 16 empty internal
   contours imputed with training-set medians; automated DL evaluated on the 193 gradable cases).

Absolute paths in the scripts are placeholders (`/PATH/TO/...`); point them at your local copies
of the data and outputs. Each results directory in `results/model_performance/` carries the
`summary.json` needed to check every aggregate against the paper without retraining; every
aggregate was verified against the manuscript during packaging (Supplementary Table S1 in full,
plus the Table 1/2 anchors - see `results/REBUILD_LOG.md` and `results/model_performance/INDEX.md`).

## Version history

- **v1.3** (current) - aligns the package with the final manuscript (v3.6): adds the 4x-wide
  737.6M capacity-control arm `resnet50_d3_m0_48` (Supplementary S5b, external 0.783±0.016),
  the representation-family bootstrap MDD `mdd_repr` (Supplementary Table S13-b), and
  `code/stats/mdd_repr.py`; anchor set extended accordingly (S5b wide row, S13-b MDD
  rows, and the S9 fold-ensemble 0.806 recomputed from the shipped predictions).
- **v1.2** - masks excluded (patient-derived data); everything reproducible from per-case
  metrics and per-fold predictions.
- **v1.1** - ResNet family + 3D std grid arms added; broken summaries regenerated
  (results/REBUILD_LOG.md); dltrain import paths fixed.
- **v1.0** - initial release.

## License

MIT (see `LICENSE`). If you use the code, please cite the manuscript (citation to be added upon
publication).
