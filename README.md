# har-project

---

# Vision-Based & Sensor-Based Human Activity Recognition (HAR)

Multimodal HAR using **video** (Vision Transformer) and **wearable sensors** (accelerometer + Random Forest), with late **probability-level fusion** and **explainability** (Grad-CAM, Attention Rollout, LRP). Evaluated for **in-domain accuracy** and **cross-subject generalization** (MMAct 18–20).&#x20;

> **Course:** LK489 – M.Eng. in Computer Vision & AI, University of Limerick
> **Author:** Siddartha Sandeep Peddada (24192929)
> **Supervisors:** Dr. Patrick Denny
> **Date:** August 22, 2025.&#x20;

---

## Table of contents

* [Overview & objectives](#overview--objectives)
* [Data](#data)
* [Methodology](#methodology)
* [Evaluation protocol & metrics](#evaluation-protocol--metrics)
* [Results (headline numbers)](#results-headline-numbers)
* [Explainability](#explainability)
* [Repository structure](#repository-structure)
* [Quickstart](#quickstart)
* [Reproducibility & environment](#reproducibility--environment)
* [Limitations & future work](#limitations--future-work)
* [Acknowledgements & citation](#acknowledgements--citation)

---

## Overview & objectives

Many HAR systems excel **within** their training domain but stumble when subjects/environments change. We investigate a **multimodal** approach that blends a **Vision Transformer (ViT)** trained on clips with a **Random Forest (RF)** trained on compact accelerometer statistics, then **fuse** their predicted probabilities with **weights from per-model F1**. Objectives: robust recognition of **Sitting, Standing, Walking**, strong **cross-subject** generalization, and **transparent** decisions via XAI.&#x20;

---

## Data

* **Video (Kaggle – Human Activity Recognition Video Dataset):** seven activities originally; we **subset** to **Sitting, Standing, Walking** to align with the sensor data. Diversity in backgrounds/lighting helps robustness but adds noise.&#x20;
* **Sensors (MMAct – smartphone accelerometer):** triaxial (X/Y/Z) segments; **subjects 18–20 held out** entirely for cross-subject testing.&#x20;

Both modalities show **class imbalance** (Walking more frequent), motivating class-aware training and per-class evaluation.&#x20;

---

## Methodology

**Video (ViT):**

* Preprocess: 8 frames/clip (**T=8**), resize **224×224**, channel-wise normalize (mean=std=0.5).&#x20;
* Backbone: **`vit_base_patch16_224`**, temporal **average pooling** across the 8 frame embeddings → linear head (3 classes). Train with **Adam (lr=1e-4)**, **batch=4**, **class-weighted cross-entropy**.&#x20;

**Sensors (RF):**

* Features per segment: **mean, std, min, max** for **X/Y/Z** → **12-D** vector; **standardize** features.&#x20;
* Classifier: **Random Forest (100 trees)** with **class\_weight="balanced"**; CV & learning/validation curves used for sanity checks.&#x20;

**Fusion (late / probability-level):**

* Weighted average of class-probabilities:
  `p_c = w_RF * p_c^RF + w_ViT * p_c^ViT`, with **w** normalized from each model’s **validation F1**.&#x20;

---

## Evaluation protocol & metrics

* **Strict splits**: hold out test sets per modality; for sensors, **MMAct subjects 18–20** are never seen in training and used for generalization tests.&#x20;
* **Metrics**: Accuracy, **per-class** Precision/Recall/**F1**, **AUPRC**, **ROC-AUC**, **confusion matrices**. Imbalance handled during training **and** reporting.&#x20;

---

## Results (headline numbers)

**On original (in-domain) test sets**

* **ViT (video):** **99% accuracy**, near-perfect PR/ROC; tiny confusion between Sitting↔Standing.&#x20;
* **RF (accelerometer):** **98.96% accuracy**; AUC=**1.00** for all classes; 1 misclass (Standing→Walking).&#x20;

**Cross-subject generalization (MMAct 18–20)**

* **ViT:** **66% accuracy**; **Standing** entirely misclassified (F1=0.00).&#x20;
* **RF:** **99% accuracy**; Standing F1≈**0.98** (0.96 recall).&#x20;
* **Fusion:** fixes much of ViT’s Standing failures; **more balanced** across all three classes (see fusion confusion matrix).&#x20;

**Takeaways**

* **Video** = highly discriminative **in-domain** but **brittle** under shift;
* **Sensors** = **robust** to new subjects via simple statistics + ensemble;
* **Fusion** = best of both: **Standing** improves markedly while retaining Sitting/Walking strength.&#x20;

---

## Explainability

Applied to ViT on representative clips:

* **Grad-CAM:** highlights lower limbs for Walking; sometimes background spill for Sitting.&#x20;
* **Attention Rollout:** most **coherent** focus on relevant body parts; sharp maps for dynamic classes.&#x20;
* **LRP:** confirms legs/torso relevance; exposes weaker, diffuse cues for **Standing** → mirrors performance gap.&#x20;

Net: predictions are **semantically grounded**, but **Standing** cues are less stable—exactly where fusion helps.&#x20;

---

## Repository structure

```
har-project/
├─ README.md
├─ .gitignore
├─ notebooks/
│  ├─ 01-video-vit.ipynb            # ViT training/eval
│  ├─ 02-accel-rf.ipynb             # RF features + model + CV
│  ├─ 03-fusion-weighted-avg.ipynb  # late fusion experiments
│  └─ 04-explainability.ipynb       # Grad-CAM, Attention Rollout, LRP
├─ models/
│  ├─ normal.onnx
│  ├─ pruned.onnx
│  └─ quantized_static.onnx
├─ docs/
│  ├─ report.pdf
│  ├─ presentation.pdf
│  └─ figures/
│     ├─ arch.png
│     ├─ confusion_matrix_vit.png
│     ├─ confusion_matrix_rf.png
│     └─ confusion_matrix_fusion.png
└─ requirements.txt
```

> The **ONNX models** (normal/pruned/quantized-static) are included for deployment/testing; the report’s core experiments use the **ViT**, **RF**, and **fusion** notebooks above.

---

## Quickstart

```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) Open notebooks
jupyter notebook  # run 01 → 02 → 03 → 04 in order
```

**requirements.txt (suggested)**

```
torch
timm
scikit-learn
numpy
pandas
opencv-python
matplotlib
seaborn
```

---

## Reproducibility & environment

* **Training**: PyTorch + `timm` for ViT (`vit_base_patch16_224`); scikit-learn for RF; OpenCV for video I/O; Matplotlib/Seaborn for plots. **Adam 1e-4**, **batch 4**, **class-weighted CE**; RF **n\_estimators=100**, **balanced** weights.&#x20;
* **Hardware**: Cloud GPU (**2× NVIDIA T4**) for ViT; CPU sufficient for RF/features.
* **Seeds** fixed across PyTorch/NumPy/sklearn; modular notebooks per stage.&#x20;

---

## Limitations & future work

* **Domain shift** remains hard for vision-only models; fusion alleviates but does not fully solve it.&#x20;
* Future directions (from the report): expand modalities/classes, stronger domain adaptation, pretraining or data augmentation targeted at **Standing**, and on-device optimization.&#x20;

---

## Acknowledgements & citation

Thanks to **University of Limerick**, supervisor **Dr. Patrick Denny**, and dataset authors (Kaggle HAR Video; MMAct).&#x20;

If you use this work, please cite the project report:

> Peddada, S. S. (2025). *Vision Based Human Activity Recognition*. LK489 – University of Limerick. (Report & supplementary materials in `docs/`).&#x20;

---

### Notes for reviewers

* Confusion matrices and per-class reports for **ViT (in-domain 99%)**, **RF (test 98.96%)**, **ViT on MMAct (66%)**, and **RF on MMAct (99%)** are reproduced in the report and figures. The **fusion** confusion matrix demonstrates recovery of **Standing** while preserving other classes.

---
