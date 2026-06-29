# Failure Modes and Prompt-based Improvements for Modern Object Segmentation Models

HLCV course project comparing SAM, Mask2Former, and Mask R-CNN on COCO val2017.  
**Recommended platform: Windows with CUDA GPU** (Mac with 8 GB RAM is too slow for these models).

---

## Requirements

- Windows 10/11 with a CUDA-capable GPU (8 GB VRAM recommended)
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Git

---

## Setup (Windows + CUDA)

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
- CUDA 11.8 → `cu118`
- CUDA 12.1 → `cu121`
- CUDA 12.4 → `cu124`

---

## Download Data and Checkpoints (Windows)

Run the PowerShell download script (downloads COCO + SAM in parallel):

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Or download manually:

| File | URL | Save to |
|------|-----|---------|
| COCO val2017 images | http://images.cocodataset.org/zips/val2017.zip | `data\coco\` |
| COCO annotations | http://images.cocodataset.org/annotations/annotations_trainval2017.zip | `data\coco\` |
| SAM ViT-B checkpoint | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth | `checkpoints\` |

Extract both zips into `data\coco\` so the structure looks like:
```
data\coco\val2017\          ← 5000 images
data\coco\annotations\
    instances_val2017.json
```

---

## Running the Experiments

All commands below are run from the repo root in Anaconda Prompt with the `hlcv` environment active.

### 1. Build evaluation subset
```bat
python src\dataset\coco_loader.py --n-images 200 --output data\subsets\subset_200.json
```

### 2. Run Mask R-CNN
```bat
python src\experiments\run_maskrcnn.py --subset data\subsets\subset_200.json --output results\maskrcnn_results.json --device cuda
```

### 3. Run Mask2Former
```bat
python src\experiments\run_mask2former.py --subset data\subsets\subset_200.json --output results\mask2former_results.json --device cuda
```

### 4. Run SAM (all 5 prompt strategies)
```bat
python src\experiments\run_sam.py --subset data\subsets\subset_200.json --checkpoint checkpoints\sam_vit_b_01ec64.pth --output results\sam_results.json --device cuda
```

### 5. Generate plots
```bat
python src\visualization\plot.py --sam results\sam_results.json --maskrcnn results\maskrcnn_results.json --mask2former results\mask2former_results.json --output figures\
```

---

## Project Structure

```
HLCV_16_SMD/
├── src/
│   ├── dataset/
│   │   ├── instance.py          # InstanceRecord dataclass, size grouping
│   │   └── coco_loader.py       # COCO loader, balanced subset creation
│   ├── models/
│   │   ├── sam_predictor.py     # SAM with 5 prompt strategies
│   │   ├── maskrcnn_predictor.py
│   │   └── mask2former_predictor.py
│   ├── evaluation/
│   │   ├── metrics.py           # IoU, mIoU, success/failure rate
│   │   ├── matcher.py           # GT↔prediction matching
│   │   └── failure_grouper.py   # Group by size/category/failure mode
│   ├── experiments/
│   │   ├── run_sam.py
│   │   ├── run_maskrcnn.py
│   │   └── run_mask2former.py
│   └── visualization/
│       └── plot.py              # All report figures
├── data/coco/                   # COCO val2017 (download separately)
├── checkpoints/                 # SAM weights (download separately)
├── results/                     # JSON outputs from experiment runs
├── figures/                     # Generated plots
├── setup.sh                     # Linux/Mac one-shot setup
├── setup.ps1                    # Windows one-shot setup
└── requirements.txt
```

---

## Models Used

| Model | Type | Source |
|-------|------|--------|
| SAM ViT-B | Promptable foundation model | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| Mask2Former | Transformer (DETR-style) | `facebook/mask2former-swin-base-coco-instance` via HuggingFace |
| Mask R-CNN ResNet50-FPN | CNN baseline | `torchvision.models.detection` (COCO_V1 weights) |

---

## SAM Prompt Strategies

Five strategies are compared in the prompt-refinement experiment:

| Strategy | Description |
|----------|-------------|
| `box` | GT bounding box only |
| `center_point` | Single point at mask centroid |
| `multi_point` | 3–5 points sampled inside GT mask |
| `box_point` | GT box + center point |
| `box_pos_neg` | GT box + positive points inside + negative points near boundary |

---

## References

- Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023
- Cheng et al., [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527), CVPR 2022
- He et al., [Mask R-CNN](https://arxiv.org/abs/1703.06870), ICCV 2017
- Lin et al., [Microsoft COCO](https://arxiv.org/abs/1405.0312), ECCV 2014
- Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV 2020
