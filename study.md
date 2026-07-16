# Study Guide: Instance Segmentation — From Zero to This Project

This guide teaches you everything about this project from scratch.  
Read it top-to-bottom. Every concept builds on the one before it.

---

## Table of Contents

1. [What Is Instance Segmentation?](#1-what-is-instance-segmentation)
2. [Core Concepts You Must Know](#2-core-concepts-you-must-know)
3. [The COCO Dataset](#3-the-coco-dataset)
4. [Evaluation Metrics Used in This Project](#4-evaluation-metrics-used-in-this-project)
5. [Model 1 — Mask R-CNN (the CNN baseline)](#5-model-1--mask-r-cnn)
6. [Model 2 — DETR (transformer with bipartite matching)](#6-model-2--detr)
7. [Model 3 — SAM (Segment Anything Model)](#7-model-3--sam)
8. [Model 4 — SAM-Auto (Improvement 1)](#8-model-4--sam-auto-improvement-1)
9. [Model 5 — Cascade MaskRCNN→SAM (Improvement 2)](#9-model-5--cascade-maskrcnn--sam-improvement-2)
10. [Model 6 — Mask2Former (and why it failed)](#10-model-6--mask2former-and-why-it-failed)
11. [How the Evaluation Pipeline Works](#11-how-the-evaluation-pipeline-works)
12. [Results Deep Dive](#12-results-deep-dive)
13. [Failure Modes Explained](#13-failure-modes-explained)
14. [Likely Exam / Viva Questions and Answers](#14-likely-exam--viva-questions-and-answers)

---

## 1. What Is Instance Segmentation?

### The three levels of vision understanding

```
Image
  │
  ├── Image Classification   → "There is a cat in this image"
  │                             (one label for the whole image)
  │
  ├── Object Detection       → "There is a cat at [x=120, y=80, w=200, h=150]"
  │                             (bounding box per object)
  │
  ├── Semantic Segmentation  → every pixel labelled: cat/dog/background
  │                             (no distinction between two cats)
  │
  └── Instance Segmentation  → pixel mask for EACH object separately
                                (cat #1 mask + cat #2 mask + dog mask)
```

**Instance segmentation = detection + pixel-level mask per object.**

Two cats standing next to each other:
- Semantic segmentation paints ALL cat pixels the same colour.
- Instance segmentation gives cat A a blue mask and cat B a red mask.

### Why is it hard?

1. You need to find every object (detection is hard).
2. You need to draw the exact pixel boundary (harder than a box).
3. Two objects of the same class can overlap — you still need separate masks.
4. Tiny objects have very few pixels — any pixel error is a large percentage mistake.

---

## 2. Core Concepts You Must Know

### 2.1 Bounding Box

A rectangle `[x1, y1, x2, y2]` (top-left corner and bottom-right corner) that encloses one object.  
Sometimes written as `[x, y, width, height]` — COCO uses this format.

```
(x1,y1)──────────────┐
   │                  │
   │    object        │
   │                  │
   └──────────────(x2,y2)
```

### 2.2 Binary Mask

A 2D array of 0s and 1s the same size as the image.  
`1` = this pixel belongs to the object. `0` = background.

```
0 0 0 0 0 0
0 0 1 1 0 0
0 1 1 1 1 0    ← mask for one object
0 0 1 1 0 0
0 0 0 0 0 0
```

### 2.3 IoU — Intersection over Union

The primary metric for "how well does your mask match the ground truth?"

```
IoU = |Predicted ∩ Ground Truth|
      ─────────────────────────
      |Predicted ∪ Ground Truth|
```

```
Ground Truth:   ████████
                ████████
Prediction:        █████████
                   █████████

Intersection:      █████
Union:          ████████████

IoU = 5/12 = 0.42
```

- IoU = 1.0 → perfect match
- IoU = 0.0 → no overlap at all
- IoU ≥ 0.5 → generally considered a "good" detection
- IoU < 0.3 → generally considered a "failure"

### 2.4 RLE — Run-Length Encoding

COCO stores masks as RLE to save space instead of storing the full binary array.  
"RLE" counts consecutive 0s and 1s: `[5, 3, 2, 4]` means "5 zeros, 3 ones, 2 zeros, 4 ones".  
The code uses `pycocotools.mask.decode(rle)` to convert back to a numpy array.

### 2.5 Feature Map

When a convolutional network processes an image, intermediate results are called feature maps.  
If the input is 640×480 and the network downsamples by 4×, the feature map is 160×120.  
Each location in the feature map represents a region of the original image.

### 2.6 Anchor Boxes

In older detection models (including parts of Mask R-CNN), the model pre-defines a set of boxes at every location in the feature map in different sizes and aspect ratios. The model then adjusts ("regresses") these pre-defined boxes to fit actual objects. This is called anchor-based detection.

### 2.7 Softmax and Sigmoid

- **Softmax** — converts a vector of scores into probabilities that sum to 1. Used for mutually exclusive classes: "is this cat OR dog OR background?"
- **Sigmoid** — converts a single score to a probability between 0 and 1, independently. Used for binary decisions: "is this pixel foreground?" Each pixel is decided independently.

---

## 3. The COCO Dataset

**COCO = Common Objects in Context.**  
The standard benchmark for instance segmentation since 2014.

### What's in it

- 80 object categories (person, car, dog, sandwich, spoon, etc.)
- 118,000 training images, 5,000 validation images
- Each image has multiple objects annotated with: bounding box + polygon mask + category label
- We use **val2017** (the 5,000 validation images) for evaluation

### Our subset

We don't use all 5,000 images — we use a **balanced 200-image subset** with **2,754 annotated instances**.

Why balanced? COCO is biased — "person" appears in almost every image. If we sampled randomly, we'd evaluate mostly on persons. Balanced sampling ensures all 80 categories are represented.

### Category IDs

COCO category IDs are NOT 0–79. They are non-contiguous: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14... (12 is skipped). This trips up many implementations. Our code maps from model label indices to COCO category IDs.

### Object size classes (COCO convention)

| Class | Area range | Count in our subset |
|---|---|---|
| Small | < 32² = 1,024 px² | 1,470 (53%) |
| Medium | 1,024 – 9,216 px² | 900 (33%) |
| Large | > 9,216 px² | 384 (14%) |

Small objects dominate — over half of all instances. This is why all models struggle; the evaluation is dominated by the hardest case.

---

## 4. Evaluation Metrics Used in This Project

### 4.1 mIoU (mean IoU) — our primary metric

```
mIoU = average IoU over ALL ground-truth instances
```

For each GT instance, we find the best-matching prediction (by IoU).  
If no prediction overlaps, IoU = 0.  
Average all these values → mIoU.

**Why not use COCO AP?** COCO AP averages over IoU thresholds (0.5, 0.55, ..., 0.95) and over categories. Our mIoU is simpler and more interpretable: it directly tells you how well masks overlap on average.

### 4.2 Success Rate

Fraction of instances where IoU ≥ 0.5.  
Mask R-CNN success = 0.592 → 59.2% of 2,754 instances are well-segmented.

### 4.3 Failure Rate

Fraction of instances where IoU < 0.3.  
Mask R-CNN failure = 0.350 → 35% of instances are essentially missed.

### 4.4 The matching rule

**Greedy one-to-one matching:**
1. For each GT instance, find the prediction with the highest IoU.
2. Each prediction can only match one GT (once matched, it's removed from the pool).
3. Unmatched GT instances get IoU = 0 — they count as complete failures.

This is important: **if a model predicts nothing for an image, all its GT instances score IoU = 0.**

---

## 5. Model 1 — Mask R-CNN

### The big idea

Mask R-CNN is a **two-stage detector**:
1. **Stage 1**: Find regions that might contain objects (region proposals).
2. **Stage 2**: For each promising region, classify it and predict a mask.

It was introduced by Facebook AI (He et al., ICCV 2017) and is the gold-standard CNN baseline.

### Architecture — step by step

```
Input Image (e.g. 640×480 RGB)
        │
        ▼
 ┌─────────────────┐
 │  ResNet-50      │  Backbone: extracts rich features from the image.
 │  (backbone)     │  ResNet has 50 layers of convolutions.
 └────────┬────────┘
          │  Multiple feature maps at different scales:
          │  P2 (160×120), P3 (80×60), P4 (40×30), P5 (20×15)
          ▼
 ┌─────────────────┐
 │  FPN            │  Feature Pyramid Network: combines all scales into
 │  (neck)         │  a multi-scale feature representation.
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  RPN            │  Region Proposal Network: slides a small network
 │  (stage 1)      │  over every feature map location and asks:
 └────────┬────────┘  "Is there an object here?" → ~2000 candidate boxes
          │
          ▼
 ┌─────────────────┐
 │  RoI Align      │  Crops the feature map at each proposed region
 │                 │  and resizes to a fixed 7×7 grid (exactly, no rounding).
 └────────┬────────┘
          │
          ├──────────────┬──────────────────────┐
          ▼              ▼                      ▼
   Box regression    Classification         Mask head
   (refine the box)  (what class is it?)    (28×28 binary mask)
```

### FPN — Feature Pyramid Network

Different objects have different sizes:
- A small bird might be 20×20 pixels → best detected in high-resolution features.
- A large truck might be 400×300 pixels → best detected in low-resolution features.

FPN creates a pyramid where:
- P2 = largest, most detailed features (for small objects)
- P5 = smallest, most abstract features (for large objects)

And it connects them all together so each level has both detail AND context.

### RoI Align

After getting a proposed region (e.g. "there's something at x=120, y=80, w=60, h=40"), we need to extract features for just that region.

**RoI Pool** (old approach): just round to nearest pixel → loses precision.  
**RoI Align** (Mask R-CNN innovation): use bilinear interpolation to sample sub-pixel locations. This is critical for masks — without it, the mask is spatially misaligned.

### The Mask Head

Outputs a 28×28 binary mask for each detected object.  
This is then upsampled (bilinear interpolation) to the object's actual bounding-box size.

**Key weakness:** 28×28 = 784 pixels to represent the mask. For a large 400×400 object, you're trying to represent 160,000 pixels with only 784 values. Boundaries will be blurry and imprecise. This is why Mask R-CNN's main failure mode is **imprecise boundaries**.

### Our results

| Metric | Value |
|---|---|
| mIoU | 0.501 |
| Success rate (IoU ≥ 0.5) | 0.592 |
| Failure rate (IoU < 0.3) | 0.350 |
| Small objects | 0.337 |
| Medium objects | 0.648 |
| Large objects | 0.785 |
| Thin objects | 0.364 |
| Non-thin objects | 0.532 |

### Why does it fail on small objects?

A small object (< 1024 px²) has maybe 32×32 = 1024 pixels in the original image. After ResNet's downsampling (÷32 at the deepest level), it's a single 1×1 feature. Even FPN's highest resolution (P2, ÷4) gives 8×8 features. The 28×28 mask head is too coarse relative to the actual object size.

### Why does it fail on thin objects?

A fork is 3 pixels wide and 60 pixels tall. Aspect ratio ≈ 1:20. The 28×28 mask head is designed for roughly square regions. A 28×28 grid representing a 3×60 object gets only ~2 pixels of width to represent the fork tines. Rounding causes them to disappear.

---

## 6. Model 2 — DETR

### The big idea

DETR (Detection Transformer, Carion et al., ECCV 2020) replaces the complex region proposal + NMS pipeline with a **single forward pass through a transformer**, followed by **Hungarian bipartite matching** to assign predictions to ground-truth objects.

"No NMS" is the key innovation: instead of generating 2,000 proposals and filtering duplicates, DETR generates exactly N predictions (N=100) and optimises them to cover all objects with one-to-one matching.

### Architecture — step by step

```
Input Image
        │
        ▼
 ┌─────────────────┐
 │  ResNet-50      │  Extract feature map (H/32 × W/32 × 2048)
 │  (backbone)     │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  1×1 Conv       │  Reduce channels to 256 for transformer
 │  + flatten      │  Shape: (H/32 × W/32) × 256 → a sequence of tokens
 └────────┬────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Transformer Encoder                    │
 │  Self-attention: every image patch      │
 │  attends to every other patch           │
 │  → global context built into features  │
 └──────────────────┬──────────────────────┘
                    │
          ┌─────────┘    +
          │         100 object queries (learnable embeddings,
          │         one per "slot" — the model learns what to look for)
          ▼
 ┌─────────────────────────────────────────┐
 │  Transformer Decoder                    │
 │  Each query attends to:                 │
 │   1. All other queries (self-attention) │
 │   2. The encoded image (cross-attention)│
 │  → each query produces one prediction  │
 └──────────────────┬──────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Class prediction        Box prediction
   (80 classes +           (4 numbers: cx, cy, w, h)
    "no object")
```

### Bipartite Matching (Hungarian Algorithm)

After the decoder outputs 100 predictions, we need to know which prediction should match which ground-truth object.

**The problem:** 100 predictions, maybe 5 GT objects. Which prediction matches which GT?

**Hungarian matching** finds the assignment that minimises total cost, where cost = class loss + box loss.

Result: each GT is matched to exactly one prediction; the remaining 95 predictions are matched to "no object". During training, predictions matched to GT must learn to output the right class and box; predictions matched to "no object" must output the no-object class.

**This eliminates the need for NMS** (Non-Maximum Suppression). Traditional methods generate many overlapping predictions and filter duplicates. DETR forces each object to be predicted exactly once.

### Our model: DETR for Panoptic → Instance

We use `facebook/detr-resnet-50-panoptic`, which predicts a panoptic segmentation map (labels every pixel as either a "thing" — countable objects — or "stuff" — background regions like sky/grass).

For instance segmentation evaluation, we extract only the "thing" segments (those with `is_thing=True`).

### Our results

| Metric | Value |
|---|---|
| mIoU | 0.446 |
| Success rate | 0.517 |
| Failure rate | 0.399 |
| Small | 0.275 |
| Medium | 0.597 |
| Large | 0.747 |
| Thin | 0.350 |

### Why DETR scores lower than Mask R-CNN

1. **No FPN.** DETR's backbone doesn't use a feature pyramid, so small objects with few features are harder to detect. Mask R-CNN's FPN specifically targets multi-scale detection.

2. **Panoptic model for instance task.** We're extracting instance masks from a panoptic model. The panoptic head uses pixel-assignment (each pixel → one segment), which can merge adjacent instances of the same class.

3. **Stacked books problem.** DETR often merges stacked identical instances (e.g. a row of books) into one segment. Book mIoU = 0.062 (lowest across all categories for DETR).

4. **Training objective mismatch.** DETR panoptic is optimised for panoptic quality (PQ), not per-instance IoU.

### DETR's advantage over Mask R-CNN

DETR handles **long-range dependencies** better. Every image patch attends to every other patch in the encoder. This helps with large objects where context from far away matters (e.g. recognising a "bus" requires seeing the whole vehicle). DETR large-object mIoU = 0.747 vs Mask R-CNN = 0.785 (close).

---

## 7. Model 3 — SAM (Segment Anything Model)

### The big idea

SAM (Kirillov et al., ICCV 2023) is a **foundation model** trained on 11 billion masks from 1 billion images. It takes an image AND a prompt (a point, a box, text, or free-form) and outputs a mask for whatever the prompt is pointing at.

It does NOT classify what it segments — it just segments. "Segment anything" means "I'll segment whatever you point at, I won't tell you what it is."

### Architecture — three components

```
Input Image
        │
        ▼
 ┌─────────────────┐
 │  Image Encoder  │  A ViT (Vision Transformer) — specifically ViT-B (Base).
 │  (ViT-B)        │  Processes the image ONCE and outputs an image embedding.
 │                 │  This is the expensive step (~80% of compute time).
 └────────┬────────┘
          │  Image embedding (stored — can be reused for multiple prompts)
          ▼
 ┌─────────────────┐
 │  Prompt Encoder │  Converts your prompt into embeddings:
 │                 │  - Points → positional encodings
 │                 │  - Boxes → two corner points
 │                 │  - Text → CLIP text embeddings (not used here)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────────────────────────────┐
 │  Mask Decoder (lightweight transformer) │
 │  Attends to: image embedding + prompts │
 │  Outputs: 3 candidate masks + scores   │
 └─────────────────────────────────────────┘
```

### ViT — Vision Transformer

Traditional CNNs process images with local filters (convolutional kernels).  
ViT divides the image into 16×16 patches, treats each patch as a "token", and uses self-attention — exactly like a language model processes words.

Each patch can attend to every other patch → global context from the start.  
ViT-B has 12 attention layers and 86 million parameters.

**Advantage:** massive pre-training on billions of masks makes it generalise to any object.  
**Disadvantage:** slow — the ViT-B encoder alone is the computational bottleneck.

### SAM's multi-mask output

When given an ambiguous prompt (e.g. a point that could be the wheel or the whole car), SAM outputs 3 masks at different levels of granularity:
- Small mask (just the wheel)
- Medium mask (the car)
- Large mask (the car + background region)

With `multimask_output=False`, we take only the top-scoring single mask.

### The five prompt strategies in this project

All five strategies use **ground-truth information at test time** — they are oracle evaluations, not realistic deployment.

#### Strategy 1: `box`
- Input: the GT bounding box `[x1, y1, x2, y2]`
- SAM gets the exact tight box around the GT object
- **Best strategy** (mIoU = 0.751) — the box perfectly constrains the search space
- SAM converts the box into 2 corner points with special "box corner" labels

#### Strategy 2: `center_point`
- Input: single point at the centroid of the GT mask
- mIoU = 0.494 — barely better than Mask R-CNN
- Fails for elongated objects: the centroid of a fork may be IN THE MIDDLE OF EMPTY SPACE
- Ambiguous: a centroid point inside a crowd scene could refer to any of 10 people

#### Strategy 3: `multi_point`
- Input: 3–5 random positive points sampled from inside the GT mask
- mIoU = 0.635 — much better than single point
- Multiple points reduce ambiguity — harder to be confused about which object is intended
- Points are sampled with `np.random.choice` over pixels where GT mask = 1

#### Strategy 4: `box_point`
- Input: GT box + GT centroid point
- mIoU = 0.734 — very close to box-only
- The extra point helps slightly for ambiguous boxes but adds little when box is tight

#### Strategy 5: `box_pos_neg`
- Input: GT box + positive points (inside mask) + negative points (near boundary, outside mask)
- mIoU = 0.702 — LOWER than plain `box`
- Why worse? Negative boundary points sometimes fall in ambiguous regions. A negative point at "not this pixel" occasionally forces the decoder to exclude valid mask area.

### The key insight from SAM prompt ablation

```
box (0.751)  >  box_point (0.734)  >  box_pos_neg (0.702)  >  multi_point (0.635)  >  center_point (0.494)
```

**More information ≠ always better.** A tight box is the strongest constraint. Adding extra points can confuse the model when those points fall in ambiguous locations.

### SAM size performance

| Size | mIoU (box strategy) |
|---|---|
| Small | 0.707 |
| Medium | 0.786 |
| Large | 0.833 |

SAM is uniquely good across all sizes because its full-resolution mask decoder doesn't suffer from the 28×28 bottleneck. Even small objects get a proper full-resolution mask.

---

## 8. Model 4 — SAM-Auto (Improvement 1)

### The big idea

All five SAM strategies above are **cheating** — they use ground-truth boxes or points at test time. Real deployment has no ground truth.

SAM-Auto answers: **what can SAM do with ZERO prior knowledge?**

### How it works

`SamAutomaticMaskGenerator` runs SAM on a 32×32 grid of points covering the entire image — **1,024 points total**. At each point it asks SAM: "what's here?" and collects the resulting mask.

```
Image (e.g. 640×480)
        │
        ▼
Generate 32×32 = 1024 grid points
        │
        ▼
For each point → run SAM → get mask + quality score
        │
        ▼
Filter by: predicted_iou > 0.86
           stability_score > 0.92
           area > 64 pixels
        │
        ▼
Remove duplicates (NMS on masks)
        │
        ▼
~20–80 masks per image (typical)
```

### The prompting gap

```
SAM with GT box (oracle):    mIoU = 0.751
SAM-Auto (no GT):            mIoU = 0.470
                             ─────────────
Prompting gap:               0.281 mIoU
```

This is the most important finding: **deployment SAM is not dramatically better than Mask R-CNN (0.501)**. The impressive SAM scores you see in papers depend on GT-guided prompting.

### Why SAM-Auto falls short

1. **Large objects:** A 400×400 object might not be "triggered" by any of the 32×32 grid points if it has complex internal texture (the model might segment just part of it). SAM-Auto large = 0.622 vs Mask R-CNN large = 0.785.

2. **No category information:** SAM-Auto predicts masks but no class labels. Our matcher uses IoU-only matching (no category filter), which can accidentally match predictions to the wrong GT instance.

3. **Over-segmentation:** the grid generates proposals for EVERYTHING — shadows, reflections, texture patterns — adding many false positives.

---

## 9. Model 5 — Cascade (MaskRCNN → SAM) (Improvement 2)

### The big idea

Mask R-CNN's main weakness: **imprecise 28×28 mask head** (boundary quality).  
SAM's main strength: **full-resolution masks given a box prompt** (boundary quality).

**Cascade = use Mask R-CNN for detection + use SAM for masking.**

```
Image
  │
  ├──→ Mask R-CNN ──→ boxes + category IDs + scores
  │                              │
  │         ┌────────────────────┘
  │         │ (for each predicted box)
  │         ▼
  └──→ SAM Encoder (once per image)
              │
              ▼
         SAM Decoder (once per box)
              │
              ▼
         Full-resolution mask
```

### Implementation details

```python
# Stage 1: Mask R-CNN
output = self.mrcnn([tensor])[0]
boxes  = output["boxes"][keep].cpu().numpy()   # (N, 4) xyxy

# Stage 2: SAM (image encoded once, masks decoded per box)
self.sam.set_image(image_rgb)   # expensive — done ONCE

for box_xyxy in boxes:
    sam_masks, scores, _ = self.sam.predict(
        box=box_xyxy.astype(float),   # (4,) numpy array ← exact format required
        multimask_output=False,
    )
    masks.append(sam_masks[0])
```

The image is encoded by SAM once and reused. Only the mask decoder runs once per detected box.

### Results

| | mIoU | Small | Medium | Large | Thin |
|---|---|---|---|---|---|
| Mask R-CNN | 0.501 | 0.337 | 0.648 | 0.785 | 0.364 |
| **Cascade** | **0.492** | **0.343** | **0.639** | **0.716** | **0.368** |
| Δ | −0.009 | **+0.006** | −0.009 | **−0.069** | +0.004 |

### Why it helps small objects

Small objects have the most to gain from better boundary quality. Mask R-CNN's 28×28 mask for a 20×20 object is already pretty coarse. SAM's full-resolution mask can fit the object exactly.

### Why it hurts large objects

For a large object like a bus (400×300 pixels):
- Mask R-CNN's mask head: trained specifically to segment large things, does well (0.785).
- SAM given the bus bounding box: sometimes over-segments (includes nearby pixels) or under-segments (gets confused by the large homogeneous surface). SAM was not trained to "fill in" large boxes.

### The lesson

Cascade is a **selective improvement** — apply it only to small/thin objects. For large objects, Mask R-CNN's own mask head is better.

---

## 10. Model 6 — Mask2Former (and why it failed)

### What Mask2Former is

Mask2Former (Cheng et al., CVPR 2022) is a **universal segmentation model** — trained once for instance, semantic, and panoptic segmentation.

Architecture:
- **Swin-B backbone** (hierarchical ViT with shifted windows)
- **Pixel decoder** (multi-scale deformable attention FPN)
- **Transformer decoder** with 100 masked-attention queries

Each query has two heads:
1. Class prediction (what category?)
2. Mask prediction (where is it?)

Training uses Hungarian matching + mask-classification loss simultaneously.  
Paper reports **51.1 AP** on COCO instance segmentation — the highest of all models we test.

### Why it scored mIoU = 0.039 in our evaluation

This is a **library compatibility issue**, not a model architecture issue.

**What we found (diagnostic results):**
- Input to model: correctly normalised (min=−2.12, max=2.64, mean=−0.26, std=1.10 — correct ImageNet normalisation)
- Input resolution: correct (800×1120 for shortest-edge=800)
- Model output: correct shapes — `class_queries_logits (1,100,81)`, `masks_queries_logits (1,100,200,280)`

**The fundamental problem:** class scores and mask quality are **completely decoupled**.

Scanning ALL 100 queries with zero threshold:
- Best achievable IoU over all GT instances: **0.515** (one small truck, by coincidence)
- That query's foreground class score: **0.0099** — the model thinks it's background
- Mean background score across all 100 queries: **0.863** — model thinks almost everything is background

A properly trained Mask2Former should have high class scores for queries that produce good masks. In our environment (transformers 4.46.3 + Python 3.8), the class scores and mask quality are uncorrelated. This is a known checkpoint conversion issue in this version of the transformers library.

**Mask2Former is therefore excluded from the main analysis.** Results are recorded in `results/mask2former_results.json` but not used in comparisons.

---

## 11. How the Evaluation Pipeline Works

### Data flow

```
data/coco/val2017/*.jpg            ← 5,000 COCO images
data/coco/annotations/             ← 5,000 image annotations
            │
            ▼
src/dataset/coco_loader.py         ← sample 200 balanced images
            │
            ▼
data/subsets/subset_200.json       ← list of 2,754 instances with:
                                      image_path, bbox, segmentation (RLE),
                                      category_id, area, image_height, image_width
            │
            ▼
src/experiments/run_*.py           ← for each image:
                                      1. load image as RGB numpy array
                                      2. run model → (masks, scores, category_ids)
                                      3. match predictions to GT
                                      4. compute IoU for each GT
            │
            ▼
results/*_results.json             ← mIoU, success/failure rates,
                                      by_size, by_category, per_instance
            │
            ▼
src/visualization/plot.py          ← read all result JSONs → 11 PNG figures
```

### The Matcher (src/evaluation/matcher.py)

```python
# Greedy one-to-one IoU matching
# 1. Compute IoU between every GT mask and every prediction mask → matrix
# 2. Find the (GT, pred) pair with the highest IoU → assign them
# 3. Remove both from consideration
# 4. Repeat until no more pairs or IoU < threshold
# 5. Remaining GT instances → IoU = 0
```

This is a greedy algorithm, not optimal (Hungarian matching would be optimal).  
But for evaluation purposes, greedy is standard and fast.

### Results JSON structure

```json
{
  "model": "MaskRCNN",
  "summary": {
    "overall": {
      "miou": 0.501,
      "success_rate": 0.592,
      "failure_rate": 0.350,
      "n": 2754
    },
    "by_size": {
      "small":  {"miou": 0.337, "n": 1470},
      "medium": {"miou": 0.648, "n": 900},
      "large":  {"miou": 0.785, "n": 384}
    },
    "by_category": {
      "person": {"miou": 0.683, "n": 826},
      "sandwich": {"miou": 0.076, "n": 20}
    },
    "thin_only": {"miou": 0.364, "n": 502}
  },
  "per_instance": [
    {
      "ann_id": 12345,
      "image_id": 457884,
      "category_name": "person",
      "size": "medium",
      "area": 4521,
      "is_thin": false,
      "iou": 0.72
    }
  ]
}
```

---

## 12. Results Deep Dive

### Full comparison table

| Model | mIoU | Succ | Fail | Small | Med | Large | Thin | Non-thin |
|---|---|---|---|---|---|---|---|---|
| SAM (box) | 0.751 | 0.905 | 0.026 | 0.707 | 0.786 | 0.833 | 0.673 | 0.768 |
| Mask R-CNN | 0.501 | 0.592 | 0.350 | 0.337 | 0.648 | 0.785 | 0.364 | 0.532 |
| Cascade | 0.492 | 0.574 | 0.359 | 0.343 | 0.639 | 0.716 | 0.368 | 0.519 |
| SAM-Auto | 0.470 | 0.516 | 0.398 | 0.347 | 0.607 | 0.622 | 0.365 | 0.494 |
| DETR | 0.446 | 0.517 | 0.399 | 0.275 | 0.597 | 0.747 | 0.350 | 0.467 |

### Interesting patterns to understand

#### Pattern 1: Box prompt quality dominates SAM performance
```
box (0.751) vs center_point (0.494) → gap of 0.257
```
The spatial constraint is everything. A tight box tells SAM exactly where the object is; a single interior point is ambiguous.

#### Pattern 2: Cascade hurts what Mask R-CNN already does well
Large objects mIoU: Mask R-CNN (0.785) → Cascade (0.716) = **−0.069**  
SAM with a loose large-object box often over-segments (grabs background) or under-segments (misses parts of a large homogeneous surface like the side of a bus).

#### Pattern 3: DETR vs Mask R-CNN on small vs large
- Small: Mask R-CNN (0.337) > DETR (0.275) — FPN helps for small objects
- Large: Mask R-CNN (0.785) > DETR (0.747) — FPN also helps for large; global attention helps slightly
- Both fail at small objects but Mask R-CNN fails less

#### Pattern 4: The thin-object penalty
```
Model         Thin    Non-thin    Gap
Mask R-CNN    0.364   0.532      −0.168
DETR          0.350   0.467      −0.117
Cascade       0.368   0.519      −0.151
SAM (box)     0.673   0.768      −0.095
```
SAM has the smallest thin-object gap — full-resolution decoder preserves thin structures. Mask R-CNN has the largest gap — 28×28 head rounds away narrow widths.

#### Pattern 5: Worst categories reveal failure modes
```
sandwich (0.076) donut (0.117) orange (0.118)  ← texture-similar boundaries
spoon (0.141) fork (0.181)                     ← thin + small
carrot (0.102) knife (0.220)                   ← elongated thin
book (0.062 for DETR)                          ← stacked identical instances
```

#### Pattern 6: Best categories = large, distinctive, common objects
```
train (0.927) toilet (0.910) bed (0.889) keyboard (0.886)
```
Large, distinctive objects that rarely overlap with other instances of the same class.

---

## 13. Failure Modes Explained

### Under-segmentation

The predicted mask is **smaller** than the GT — part of the object is classified as background.

Common causes:
- Low-confidence pixels at the object boundary (model is uncertain)
- Thin object parts (fork tines) below the mask head resolution
- Occluded parts of an object (the model stops at the occlusion)

### Over-segmentation

The predicted mask is **larger** than the GT — background or adjacent objects are included.

Common causes:
- Poor box prompt (especially large loose boxes fed to SAM)
- Similar texture between object and background (e.g. sandwich on a plate — both are "food-coloured")
- Merging of adjacent instances of the same class (DETR, books)

### Boundary confusion

The mask is roughly the right shape but the edges are blurry or offset.

Common causes:
- 28×28 mask upsampling in Mask R-CNN (blurry boundaries)
- Low-resolution features for small objects

### Complete miss (IoU = 0)

The model produces no prediction for a GT instance.

Common causes:
- Object is too small (below RPN threshold)
- Object is heavily occluded
- Object category was not represented in training (rare in COCO)
- Cascade: if Mask R-CNN misses the object in Stage 1, SAM never gets to see it

---

## 14. Likely Exam / Viva Questions and Answers

### Q: What is the difference between semantic and instance segmentation?

**A:** Semantic segmentation assigns a class label to every pixel but does not distinguish between different instances of the same class — two cats get the same colour. Instance segmentation assigns a separate mask to each individual object, so cat #1 and cat #2 have different masks even though they are both "cat". Instance segmentation is harder because it requires counting and separating individual objects.

---

### Q: What is IoU and why is it used as the evaluation metric?

**A:** IoU (Intersection over Union) measures how much two masks overlap:  
IoU = |A ∩ B| / |A ∪ B|  
It ranges from 0 (no overlap) to 1 (perfect match). It naturally penalises both under-segmentation (small predicted mask → small intersection) and over-segmentation (large predicted mask → large union). We use mIoU (mean IoU over all GT instances) as our primary metric, treating unmatched GT instances as IoU=0.

---

### Q: How does Mask R-CNN work?

**A:** Mask R-CNN is a two-stage detector. In Stage 1, a Region Proposal Network (RPN) generates ~2000 candidate bounding boxes. In Stage 2, each candidate region is cropped from the FPN feature map using RoI Align (precise sub-pixel crop), then classified (what class?) and regressed (refine the box). Simultaneously, a mask head generates a 28×28 binary mask for each detection. This 28×28 mask is then upsampled to the detection box size.

---

### Q: What is FPN and why does Mask R-CNN use it?

**A:** FPN (Feature Pyramid Network) combines feature maps at multiple scales — from coarse (high semantic, low resolution) to fine (low semantic, high resolution) — into a single multi-scale representation. Each scale is good for detecting objects of a specific size: small objects are detected in high-resolution maps, large objects in low-resolution maps. Without FPN, Mask R-CNN would miss many small objects because there are too few feature-map activations for them.

---

### Q: What is RoI Align and why is it better than RoI Pool?

**A:** RoI Pool rounds the proposed region boundaries to the nearest integer pixel, which introduces spatial misalignment — the extracted features don't precisely correspond to the proposed region. RoI Align uses bilinear interpolation to sample feature values at exact sub-pixel locations, preserving precise spatial correspondence. This is critical for mask prediction because even 1 pixel of misalignment makes the mask wrong.

---

### Q: How does DETR differ from Mask R-CNN?

**A:** DETR replaces the region proposal + NMS pipeline with a transformer encoder-decoder. The encoder builds global image representations (each patch attends to every other patch). The decoder takes 100 learnable "object queries" and produces 100 predictions in one shot. Hungarian bipartite matching then assigns predictions to GT objects during training, ensuring each object is predicted exactly once. DETR needs no NMS, no anchors, no FPN. However, it trains slowly, struggles with small objects (no FPN), and the panoptic version we use can merge adjacent identical instances.

---

### Q: What is the SAM architecture and what makes it unique?

**A:** SAM (Segment Anything) has three components: (1) a ViT image encoder that processes the image once into a compact embedding, (2) a lightweight prompt encoder that converts user prompts (points, boxes) into embeddings, and (3) a mask decoder transformer that attends to both and produces 3 candidate masks. What makes SAM unique: it was trained on 11 billion masks from SA-1B (a dataset they created) with a prompt-to-mask training objective. It generalises to any object category and produces full-resolution masks — there is no 28×28 bottleneck. However, it does not classify what it segments, and performance depends heavily on prompt quality.

---

### Q: What is the "prompting gap" and why does it matter?

**A:** The prompting gap is the difference between oracle SAM performance (where we give it the exact ground-truth bounding box) and realistic deployment SAM (SAM-Auto, where we give it a grid of automatic prompts). In our evaluation:  
- SAM with GT box: mIoU = 0.751  
- SAM-Auto (no GT): mIoU = 0.470  
- Gap: 0.281 mIoU  

This matters because many papers report SAM results using GT prompts and claim SAM is dramatically better than Mask R-CNN. But in real deployment, SAM-Auto (0.470) is only slightly below Mask R-CNN (0.501). The improvements reported in papers are largely due to the evaluation protocol, not architectural superiority.

---

### Q: What is the Cascade model and what does it achieve?

**A:** Cascade is a two-stage pipeline where Mask R-CNN handles detection and classification (Stage 1), and SAM generates the actual mask from each predicted bounding box (Stage 2). The motivation: Mask R-CNN's main failure is imprecise boundaries (28×28 mask head). SAM produces full-resolution masks given a box prompt. Results: Cascade helps small objects (+0.006 mIoU) and thin objects (+0.004 mIoU) where boundary precision matters most, but hurts large objects (−0.069 mIoU) where Mask R-CNN's trained mask head already performs well. Overall: −0.009 vs standalone Mask R-CNN.

---

### Q: Why does Mask2Former score mIoU = 0.039 despite being the strongest model on paper?

**A:** This is an implementation compatibility issue. We confirmed via diagnostics that the model receives correctly normalised input at the correct resolution. However, scanning all 100 queries with no threshold, the maximum achievable IoU over all GT instances is only 0.515 (a coincidental overlap with a small truck), and that query has a foreground score of 0.0099 — the model thinks it's background. This means the class scores and mask quality are completely decoupled: queries with high class confidence produce wrong masks, and queries with good spatial masks have near-zero class scores. This is a known issue with the `facebook/mask2former-swin-base-coco-instance` checkpoint in transformers 4.46.3 + Python 3.8.

---

### Q: Why do all models fail on small objects?

**A:** Small objects (< 1024 px²) have very few pixels. After ResNet's stride-32 downsampling, a 32×32 object becomes a single 1×1 feature map location. Even FPN's highest resolution (stride-4) gives only 8×8 features. The models have almost no information to work with. Additionally, any small spatial error in the mask (a few pixels off) represents a large fraction of the object's area, dramatically reducing IoU. SAM handles small objects better because its full-resolution decoder avoids the 28×28 bottleneck, but even SAM drops from 0.833 (large) to 0.707 (small).

---

### Q: Why are food items (sandwich, donut, orange) the worst-performing categories?

**A:** Two reasons: (1) **Texture-similar boundaries** — a sandwich on a plate and the plate itself have similar textures and colours; the boundary between sandwich and background is not a sharp edge but a gradual transition. The model is uncertain where the sandwich ends. (2) **Irregular, non-convex shapes** — sandwiches, donuts, and oranges are irregular blobs without consistent geometric shape. Models trained on all 80 categories with rectangular anchor proposals struggle with highly irregular shapes. The same applies to "donut with a hole" — the hole confuses the mask head (should the hole be part of the donut mask or not?).

---

### Q: What is Hungarian matching and why does DETR need it?

**A:** Hungarian matching (also called the assignment problem) finds the optimal one-to-one matching between two sets that minimises total cost. DETR generates 100 predictions and needs to match them to, say, 5 GT objects. The cost for matching prediction P to GT G is: class classification loss + box regression loss. Hungarian matching finds the assignment of 100 predictions to (5 GT + 95 "no object") that minimises total cost. This is necessary because without it, multiple predictions would learn to predict the same object, and no unique assignment would be forced. The Hungarian algorithm runs in O(n³) but since n=100 predictions this is fast.

---

### Q: What is the thin-object failure mode and which model handles it best?

**A:** Thin objects (bicycle, fork, spoon, tie, etc.) have high aspect ratios or narrow widths. Mask R-CNN's 28×28 mask head fails because a 3-pixel-wide fork tine might map to 0.5 pixels in the 28×28 grid, which rounds to either 0 (miss) or 1 (over-thick). DETR has a similar issue from panoptic pixel assignment. SAM with a box prompt handles thin objects best (thin mIoU = 0.673 vs Mask R-CNN's 0.364) because its full-resolution mask decoder can represent sub-pixel thin structures. Even SAM drops 0.095 from non-thin to thin (0.768 vs 0.673), showing thin objects are universally challenging.

---

### Q: In the evaluation, what happens if the model predicts nothing for an image?

**A:** All GT instances in that image get IoU = 0. In our greedy matching scheme, each GT is matched to the best available prediction. If there are no predictions, no matching occurs, and every GT instance counts as a complete failure. This is why high failure rate and low mIoU are correlated — both are driven by the fraction of GT instances that the model completely misses.

---

### Q: What is the difference between mIoU in this project and COCO AP?

**A:** COCO AP (Average Precision) is more complex: it varies the IoU threshold from 0.5 to 0.95 in steps of 0.05, computes precision-recall curves at each threshold, averages them, then averages over all 80 categories. Our mIoU simply averages IoU directly over all GT instances with one-to-one greedy matching. COCO AP is the standard benchmark metric; our mIoU is simpler and more directly interpretable. The ranking of models is similar under both metrics, but the absolute numbers differ. A model with COCO AP=50 typically has per-instance mIoU around 45–55% in our evaluation.

---

*End of Study Guide. Good luck!*
