# Benchmarking Instance Segmentation Architectures: Failure Modes and Improvements

HLCV course project evaluating six instance segmentation models on COCO val2017, with
per-category and per-size failure analysis and two implemented improvements.

---

## Results Summary

Evaluated on a balanced 200-image subset of COCO val2017 (2,754 instances total).  
Metric: **mIoU** — mean IoU over all ground-truth instances (unmatched GT = IoU 0).

| Model | mIoU | Success rate | Failure rate | Small | Medium | Large |
|---|---|---|---|---|---|---|
| SAM (box, GT-guided) | **0.751** | **0.905** | 0.026 | 0.707 | 0.786 | 0.833 |
| SAM (box\_point) | 0.734 | 0.875 | 0.044 | 0.687 | 0.770 | 0.825 |
| SAM (box\_pos\_neg) | 0.702 | 0.820 | 0.058 | 0.615 | 0.786 | 0.836 |
| SAM (multi\_point) | 0.635 | 0.731 | 0.166 | 0.529 | 0.752 | 0.771 |
| **Mask R-CNN** | **0.501** | 0.592 | 0.350 | 0.337 | 0.648 | 0.785 |
| SAM (center\_point) | 0.494 | 0.540 | 0.319 | 0.512 | 0.493 | 0.427 |
| **Cascade** (MaskRCNN→SAM) | **0.492** | 0.574 | 0.359 | 0.343 | 0.639 | 0.716 |
| **SAM-Auto** (no GT) | **0.470** | 0.516 | 0.398 | 0.347 | 0.607 | 0.622 |
| **DETR** | **0.446** | 0.517 | 0.399 | 0.275 | 0.597 | 0.747 |
| Mask2Former† | 0.039 | 0.008 | 0.970 | — | — | — |

† **Mask2Former excluded from main analysis.** Despite its reported 51.1 AP on COCO, the HuggingFace checkpoint `facebook/mask2former-swin-base-coco-instance` produces mIoU=0.039 in our environment (transformers 4.46.3 + Python 3.8). Extensive diagnostics confirmed the issue is in the model's class-score calibration — class scores and mask quality are completely decoupled — and is a checkpoint/library compatibility issue, not a preprocessing error. See [report/report.md §10](report/report.md) for the full diagnostic.

**Key findings:**
- The **prompting gap** between oracle SAM (`box`, mIoU=0.751) and realistic deployment (`SAM-Auto`, mIoU=0.470) is **0.281 mIoU** — deployment SAM merely matches Mask R-CNN.
- **Cascade** (Mask R-CNN boxes → SAM masks) helps small objects (+0.006) but hurts large (−0.069); best applied selectively, not globally.
- All models fail disproportionately on **small objects** — Mask R-CNN drops 57% from large (0.785) to small (0.337).
- **Food items** (sandwich mIoU=0.076, donut=0.117) and **thin utensils** (spoon=0.141, fork=0.181) are the worst-performing categories.

---

## Overview

| Model | Architecture Family | Notes |
|---|---|---|
| Mask R-CNN | CNN + region proposals (FPN + RoIAlign) | Baseline; dense proposals |
| Mask2Former | Transformer (masked-attention) | HF checkpoint; excluded — see above |
| DETR | Transformer (bipartite matching, no NMS) | Panoptic model; things only |
| SAM | Vision transformer + prompt encoder | 5 GT-guided prompt strategies |
| SAM-Auto | SAM with automatic mask generation | No GT — real deployment scenario |
| Cascade (MaskRCNN→SAM) | Two-stage: detection + SAM refinement | **Implemented improvement** |

The two improvements address the reviewer feedback to "propose improvements that challenge
common failure cases":

1. **SAM-Auto** — quantifies the "prompting gap" between oracle GT-box prompting and
   realistic deployment (no GT information at all).
2. **Cascade** — Mask R-CNN predicts bounding boxes; SAM refines each box into a
   full-resolution mask, replacing Mask R-CNN's low-resolution 28×28 mask head. Directly
   addresses Mask R-CNN's main failure mode: imprecise boundaries on thin/articulated objects.

---

## Requirements

- Windows 10/11 with a CUDA-capable GPU (8 GB VRAM recommended)
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Git

---

## Setup

Open **Anaconda Prompt** and run:

```bat
:: 1. Clone the repo
git clone <your-repo-url>
cd HLCV_16_SMD

:: 2. Create conda environment
conda create -n hlcv python=3.10 -y
conda activate hlcv

:: 3. Install PyTorch with CUDA (adjust cu118/cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 4. Install all other dependencies + SAM
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -e .
```

Check your CUDA version with `nvidia-smi` and pick the right wheel:
| CUDA version | PyTorch index suffix |
|---|---|
| 11.8 | `cu118` |
| 12.1 | `cu121` |
| 12.4 | `cu124` |

---

## Download Data and Checkpoints

Run the PowerShell download script (downloads COCO val2017 + SAM checkpoint in parallel):

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Or download manually:

| File | URL | Destination |
|------|-----|-------------|
| COCO val2017 images | http://images.cocodataset.org/zips/val2017.zip | `data\coco\` |
| COCO annotations | http://images.cocodataset.org/annotations/annotations_trainval2017.zip | `data\coco\` |
| SAM ViT-B checkpoint | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth | `checkpoints\` |

After extracting, the structure should look like:
```
data\coco\
    val2017\               ← 5,000 images
    annotations\
        instances_val2017.json
checkpoints\
    sam_vit_b_01ec64.pth
```

---

## Running the Full Pipeline

All commands are run from the repo root with the `hlcv` environment active.

### Step 1 — Create evaluation subset (balanced 200-image sample)

```bat
python src\dataset\coco_loader.py --n-images 200 --output data\subsets\subset_200.json
```

### Step 2 — Run baseline model evaluations (requires CUDA)

```bat
python src\experiments\run_maskrcnn.py ^
    --subset data\subsets\subset_200.json ^
    --output results\maskrcnn_results.json --device cuda

python src\experiments\run_mask2former.py ^
    --subset data\subsets\subset_200.json ^
    --output results\mask2former_results.json ^
    --score-threshold 0.1 --device cuda

python src\experiments\run_detr.py ^
    --subset data\subsets\subset_200.json ^
    --output results\detr_results.json --device cuda

python src\experiments\run_sam.py ^
    --subset data\subsets\subset_200.json ^
    --checkpoint checkpoints\sam_vit_b_01ec64.pth ^
    --output results\sam_results.json --device cuda
```

### Step 2b — Run improvement evaluations (same data, same checkpoint)

```bat
python src\experiments\run_sam_auto.py ^
    --subset data\subsets\subset_200.json ^
    --checkpoint checkpoints\sam_vit_b_01ec64.pth ^
    --output results\sam_auto_results.json --device cuda

python src\experiments\run_cascade.py ^
    --subset data\subsets\subset_200.json ^
    --checkpoint checkpoints\sam_vit_b_01ec64.pth ^
    --output results\cascade_results.json --device cuda
```

### Step 3 — Generate all figures

```bat
python src\visualization\plot.py ^
    --sam         results\sam_results.json ^
    --maskrcnn    results\maskrcnn_results.json ^
    --mask2former results\mask2former_results.json ^
    --detr        results\detr_results.json ^
    --sam-auto    results\sam_auto_results.json ^
    --cascade     results\cascade_results.json ^
    --output      figures\
```

Any `--*` argument can be omitted if that result file is not yet available — the plotter
skips missing models gracefully.

---

## Output Figures

All 11 figures are saved to `figures/` after running Step 3.

| Figure | Description |
|---|---|
| `overall_miou.png` | mIoU for all models and SAM strategies |
| `miou_by_size.png` | mIoU split by small / medium / large objects |
| `failure_rate_by_size.png` | Failure rate (IoU < 0.3) by object size |
| `worst_cats_maskrcnn.png` | 10 worst categories for Mask R-CNN |
| `worst_cats_mask2former.png` | 10 worst categories for Mask2Former |
| `worst_cats_detr.png` | 10 worst categories for DETR |
| `worst_cats_cascade.png` | 10 worst categories for the Cascade model |
| `sam_prompt_comparison.png` | mIoU across the 5 SAM prompt strategies |
| `thin_vs_nonthin.png` | mIoU on thin vs non-thin objects per model |
| `failure_modes.png` | Stacked bar: success / partial / hard-failure per model |
| `improvement_comparison.png` | MaskRCNN → Cascade → SAM-Auto → SAM(GT) by size |

---

## Report

The full written analysis is at [report/report.md](report/report.md), covering:
- Overall and size-stratified mIoU tables
- Prompt-quality ablation and the 0.281 prompting gap
- Thin-object and worst-category analysis
- Improvement implementation and results
- Mask2Former diagnostic and exclusion rationale
- Conclusions and failure mode taxonomy

---

## Project Structure

```
HLCV_16_SMD/
├── src/
│   ├── dataset/
│   │   ├── instance.py              # InstanceRecord dataclass, size grouping
│   │   └── coco_loader.py           # COCO loader, balanced subset creation
│   ├── models/
│   │   ├── sam_predictor.py         # SAM with 5 GT-guided prompt strategies
│   │   ├── maskrcnn_predictor.py    # torchvision Mask R-CNN wrapper
│   │   ├── mask2former_predictor.py # HuggingFace Mask2Former wrapper
│   │   ├── detr_predictor.py        # HuggingFace DETR panoptic wrapper
│   │   └── cascade_predictor.py     # MaskRCNN detect → SAM segment (improvement)
│   ├── evaluation/
│   │   ├── metrics.py               # IoU, mIoU, success/failure rate
│   │   ├── matcher.py               # Greedy GT↔prediction IoU matching
│   │   └── failure_grouper.py       # Group failures by size/category/type
│   ├── experiments/
│   │   ├── run_sam.py
│   │   ├── run_maskrcnn.py
│   │   ├── run_mask2former.py
│   │   ├── run_detr.py
│   │   ├── run_sam_auto.py          # SAM automatic (no GT) — improvement 1
│   │   └── run_cascade.py           # Cascade pipeline — improvement 2
│   └── visualization/
│       └── plot.py                  # All 11 report figures
├── data/coco/                       # COCO val2017 (not in repo — download separately)
├── checkpoints/                     # SAM ViT-B weights (not in repo — download separately)
├── results/                         # JSON outputs written by experiment runners
├── figures/                         # PNG figures written by plot.py
├── report/
│   └── report.md                    # Full written analysis
├── setup.ps1                        # Windows download script
└── requirements.txt
```

---

## Models

### Baselines

| Model | HuggingFace / source | Notes |
|---|---|---|
| Mask R-CNN ResNet50-FPN | `torchvision.models.detection` (COCO\_V1) | Dense proposals, FPN, RoIAlign |
| Mask2Former Swin-B | `facebook/mask2former-swin-base-coco-instance` | Excluded — see Results note |
| DETR ResNet-50 | `facebook/detr-resnet-50-panoptic` | Thing segments only; score threshold 0.5 |
| SAM ViT-B | `sam_vit_b_01ec64.pth` (local) | GT-prompted; see strategies below |

### Improvements

| Model | Description |
|---|---|
| SAM-Auto | `SamAutomaticMaskGenerator` with 32×32 point grid; no GT information |
| Cascade | Mask R-CNN boxes → `SamPredictor`; SAM image encoded once per image; no GT |

---

## SAM Prompt Strategies

| Strategy | Input | Description |
|---|---|---|
| `box` | GT bounding box | Tightest possible spatial constraint |
| `center_point` | Single point at mask centroid | Minimal point prompt |
| `multi_point` | 3–5 random points inside GT mask | More coverage, some noise |
| `box_point` | GT box + centroid point | Box with extra guidance |
| `box_pos_neg` | GT box + interior positive + boundary negative | Richest prompt |

---

## Evaluation Protocol

- **Subset**: 200 images balanced across COCO val2017 categories (2,754 instances)
- **Matching**: Greedy per-GT IoU matching (unmatched GT = IoU 0)
- **Metrics**: mIoU, success rate (IoU ≥ 0.5), failure rate (IoU < 0.3)
- **Groupings**: small (<1024 px²) / medium / large; thin vs non-thin; per-category
- **Thin categories**: bicycle, chair, tie, fork, spoon, wine glass, skateboard, umbrella, scissors

---

## References

- Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023
- Cheng et al., [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527), CVPR 2022
- He et al., [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017
- Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020
- Lin et al., [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312), ECCV 2014
