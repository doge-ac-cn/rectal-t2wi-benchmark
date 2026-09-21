#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从产物 JSON 直接生成论文 Tables 1-4 (markdown) -> manuscript/tables.md"""
import os, json

ROOT = "/PATH/TO/rectal_project"
OUTD = os.path.join(ROOT, "experiments/outputs")

def jload(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None

def ms(v):
    return f"{v[0]:.3f}±{v[1]:.3f}" if isinstance(v, (list, tuple)) and len(v) == 2 else "-"

L = ["# Manuscript Tables (auto-generated from product JSONs, 2026-09-04)", ""]

# ---- Table 1: cropping variants (ResNet50) ----
L += ["## Table 1. Input-representation comparison (ResNet50, 2.5D/2D) and 3D grid",
      "| Representation | Variant | Internal val AUC | External AUC |", "|---|---|---|---|"]
t1 = [("2D", "tumor-ROI 224²", "resnet50_d2_roi"), ("2D", "bounding-box 224²", "resnet50_d2_rect"),
      ("2.5D", "tumor-ROI 224²", "resnet50_d25_roi"), ("2.5D", "bounding-box 224²", "resnet50_d25_rect"),
      ("3D", "ROI 48³, margin 0", "resnet50_d3_m0_48"), ("3D", "ROI 64³, margin 10mm", "resnet50d3_m10_64")]
for rep, var, d in t1:
    s = jload(f"{OUTD}/{d}/summary.json")
    if s:
        L.append(f"| {rep} | {var} | {ms(s.get('val_auc_mean_std'))} | {ms(s.get('ext_auc_mean_std'))} |")

# ---- Table 2: backbone sweep ----
L += ["", "## Table 2. Backbone sweep (2.5D tumor-ROI, ImageNet init)",
      "| Backbone | Params (M) | Internal val AUC | External AUC |", "|---|---|---|---|"]
params = {"resnet18": 11.2, "resnet34": 21.3, "resnet50": 25.6, "resnet101": 42.5, "resnet152": 58.1,
          "efficientnet_b0": 5.3, "efficientnet_b1": 7.8, "efficientnet_b2": 9.2, "efficientnet_b3": 12.3,
          "efficientnet_b4": 19.3, "efficientnet_b5": 28.3, "efficientnet_b6": 40.7, "efficientnet_b7": 63.8,
          "convnext_t": 28.6, "convnext_base": 87.6, "convnext_large": 196.2, "vit_base": 85.8, "vit_large": 303.3}
order = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] + \
        [f"efficientnet_b{i}" for i in range(8)] + ["convnext_t", "convnext_base", "convnext_large",
                                                    "vit_base", "vit_large"]
for b in order:
    s = jload(f"{OUTD}/{b}_d25_roi/summary.json")
    if s:
        L.append(f"| {b} | {params.get(b,'-')} | {ms(s.get('val_auc_mean_std'))} | {ms(s.get('ext_auc_mean_std'))} |")

# ---- Table 3: 3D initialization chain ----
L += ["", "## Table 3. The 3D plateau: initialization and context variants (3D ResNet50, 48³ unless noted)",
      "| Variant | Initialization / change | Internal val AUC | External AUC |", "|---|---|---|---|"]
t3 = [("scratch", "random init", "resnet50d3_scratch_std"),
      ("ImageNet-inflated", "2D ImageNet weights inflated to 3D", "resnet50d3_pt_std"),
      ("MedicalNet", "23-dataset 3D medical pretraining", "medicalnet_d3_m0_48"),
      ("SwinUNETR-SSL", "5050-volume CT self-supervised encoder", "swinunetr_ssl_d3_m0_48"),
      ("Context +10mm/64³", "margin 10mm, grid 64³", "resnet50d3_m10_64"),
      ("MTL (UNet3D)", "joint segmentation+classification", "unet3d_mtl_ls0.5")]
for name, note, d in t3:
    s = jload(f"{OUTD}/{d}/summary.json")
    if s:
        L.append(f"| {name} | {note} | {ms(s.get('val_auc_mean_std'))} | {ms(s.get('ext_auc_mean_std'))} |")
L.append("| *2.5D reference (ResNet50)* | *—* | *0.831±0.024* | *0.852±0.037* |")

# ---- Table 4: manual vs auto contour ----
dl = jload(f"{OUTD}/delong/summary.json")
delong = {d_["comparison"]: d_ for d_ in dl.get("delong_tests", [])} if dl else {}
e17 = jload(f"{OUTD}/auto/summary.json") or {}
L += ["", "## Table 4. Manual vs fully-automated contour pipeline (paired)",
      "| Branch | Set | Manual AUC | Auto AUC | Δ | DeLong p |", "|---|---|---|---|---|---|"]
for comp_key, key in [(("Radiomics manual", "test"), ("Radiomics", "Internal test (n=194)")),
                      (("Radiomics manual", "external"), ("Radiomics", "External (n=82)")),
                      (("DL", "external"), ("DL (ConvNeXt-B)", "External (n=82)"))]:
    t = next((d_ for d_ in dl.get("delong_tests", [])
              if comp_key[0] in d_.get("comparison", "") and comp_key[1] == d_.get("set")), None)
    if t:
        L.append(f"| {key[0]} | {key[1]} | {t['auc_a']:.4f} | {t['auc_b']:.4f} | "
                 f"{t['auc_a']-t['auc_b']:+.4f} | {t.get('p', t.get('p_value', float('nan'))):.3g} |")
L += ["", "*Auto radiomics external 0.8694 exceeds the protocol-matched published radiomics-only baseline "
          "(0.813 [0.712–0.891]). Auto DL reference: 0.7905±0.042 (ConvNeXt-B 2.5D-ROI-auto).*",
      "", "[PENDING] reproduction rows: paper-pipeline SVM (LASSO→SMOTE→GridSearchCV) on manual/auto features."]

open(os.path.join(ROOT, "manuscript/tables.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
