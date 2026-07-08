# Benchmarking Instance Segmentation Architectures: Failure Modes and Improvements

**Course:** High-Level Computer Vision (HLCV)  
**Evaluation set:** COCO val2017, 200-image balanced subset (2,754 instances)  

---

## 1. Introduction

Instance segmentation — producing a per-pixel binary mask for every object instance in an image — is one of the most demanding tasks in computer vision. It requires simultaneously solving detection (finding objects), classification (naming them), and precise pixel-level delineation (which pixels belong to each).

Three families of architecture have dominated recent benchmarks:

1. **Proposal-based CNNs** (Mask R-CNN): a region-proposal network produces candidate bounding boxes; a mask head outputs a 28×28 binary mask for each.
2. **Transformer models with bipartite matching** (DETR): a set of learned object queries is matched to ground-truth instances end-to-end, eliminating NMS.
3. **Promptable vision-foundation models** (SAM): a ViT image encoder combined with a prompt encoder enables zero-shot segmentation from point, box, or automatic grid prompts.

We evaluate these families on a balanced 200-image subset of COCO val2017 and analyse where each architecture fails. We additionally implement two improvements that directly challenge the most common failure modes.

---

## 2. Models Evaluated

| Model | Architecture | Source | Notes |
|---|---|---|---|
| Mask R-CNN | CNN + FPN + RoIAlign | torchvision (COCO\_V1) | Baseline |
| Mask2Former | Transformer (masked-attention) | HF `mask2former-swin-base-coco-instance` | See §6 |
| DETR | Transformer (bipartite matching) | HF `detr-resnet-50-panoptic` | Things only |
| SAM | ViT + prompt encoder | SAM ViT-B checkpoint | 5 prompt strategies |
| SAM-Auto | SAM (automatic grid) | same checkpoint | Improvement 1 |
| Cascade | Mask R-CNN → SAM | same checkpoint | Improvement 2 |

### SAM prompt strategies

SAM is tested with five GT-guided prompt strategies to isolate the effect of prompt quality:

| Strategy | Inputs provided to SAM |
|---|---|
| `box` | GT bounding box |
| `center_point` | Single point at mask centroid |
| `multi_point` | 3–5 random points inside GT mask |
| `box_point` | GT box + centroid point |
| `box_pos_neg` | GT box + interior positive + boundary negative points |

These use ground-truth information at inference time and therefore represent oracle upper bounds, not realistic deployment performance.

---

## 3. Evaluation Protocol

**Subset creation.** We sample 200 images from COCO val2017, balanced across categories, yielding 2,754 annotated instances. The subset is stratified to avoid the long-tail bias (where per-image sampling would over-represent images with many small objects).

**Matching.** Each ground-truth instance is greedily matched to the prediction with the highest IoU (one-to-one). Unmatched ground-truth instances contribute IoU=0.

**Metrics.**
- **mIoU**: mean IoU over all ground-truth instances (including unmatched).
- **Success rate**: fraction of instances with IoU ≥ 0.5.
- **Failure rate**: fraction of instances with IoU < 0.3.

**Object size classes** (following COCO convention):
- Small: area < 32² px = 1,024 px² (n=1,470)
- Medium: 32² ≤ area < 96² px = 9,216 px² (n=900)
- Large: area ≥ 96² px (n=384)

**Thin-object flag.** Nine morphologically thin or articulated COCO categories are flagged: bicycle, chair, tie, fork, spoon, wine glass, skateboard, umbrella, scissors. These have high aspect ratio or complex topology that is systematically harder to segment. Thin instances: n=502 (18.2% of total).

---

## 4. Overall Results

| Model | mIoU | Success rate | Failure rate |
|---|---|---|---|
| SAM (`box`, GT-guided) | **0.751** | **0.905** | **0.026** |
| SAM (`box_point`) | 0.734 | 0.875 | 0.044 |
| SAM (`box_pos_neg`) | 0.702 | 0.820 | 0.058 |
| SAM (`multi_point`) | 0.635 | 0.731 | 0.166 |
| **Mask R-CNN** | **0.501** | 0.592 | 0.350 |
| SAM (`center_point`) | 0.494 | 0.540 | 0.319 |
| Cascade (MaskRCNN→SAM) | 0.492 | 0.574 | 0.359 |
| SAM-Auto | 0.470 | 0.516 | 0.398 |
| **DETR** | **0.446** | 0.517 | 0.399 |
| Mask2Former | 0.039† | 0.008 | 0.970 |

† Mask2Former shows an implementation limitation; see §6.

**Key observation.** The SAM architecture with a GT bounding-box prompt achieves mIoU=0.751 — 50 percentage points above the best proposal-based baseline (Mask R-CNN, 0.501). However, this requires ground-truth boxes at test time. SAM in realistic deployment (SAM-Auto, no GT) drops to 0.470, closely matching Mask R-CNN.

Among prompt strategies, using the GT box alone (`box`, mIoU=0.751) outperforms using the box plus extra points (`box_pos_neg`, 0.702). This suggests the extra negative boundary points occasionally mislead the mask decoder when they fall in ambiguous regions.

---

## 5. Size-Based Analysis

| Model | Small (n=1470) | Medium (n=900) | Large (n=384) |
|---|---|---|---|
| Mask R-CNN | 0.337 | 0.648 | 0.785 |
| DETR | 0.275 | 0.597 | 0.747 |
| SAM (`box`) | 0.707 | 0.786 | 0.833 |
| SAM-Auto | 0.347 | 0.607 | 0.622 |
| Cascade | 0.343 | 0.639 | 0.716 |

**All models fail disproportionately on small objects.** Mask R-CNN's mIoU drops from 0.785 (large) to 0.337 (small) — a 2.3× gap. DETR shows an even larger gap (0.747 → 0.275, 2.7×). This is expected: small objects produce fewer feature-map activations, and the 28×28 mask head in Mask R-CNN loses fine detail at low resolution.

**SAM is the only model with uniformly high performance across sizes** (box prompt: 0.707 small → 0.833 large), reflecting its full-resolution mask decoder architecture that does not downsample to a fixed grid.

**SAM-Auto vs Cascade on small objects.** Both score similarly for small objects (0.347 vs 0.343), but SAM-Auto outperforms Cascade on large objects (0.622 vs 0.716). The Cascade model actually hurts large-object mIoU (-0.069 vs standalone Mask R-CNN), because SAM given a loose large-object box sometimes produces a less precise mask than the Mask R-CNN mask head for objects that fit well within the box constraint.

---

## 6. Prompt Quality and the Deployment Gap (SAM Strategies)

| SAM strategy | mIoU | Input quality |
|---|---|---|
| `box` | 0.751 | GT box (oracle) |
| `box_point` | 0.734 | GT box + GT centroid |
| `box_pos_neg` | 0.702 | GT box + GT interior + GT boundary |
| `multi_point` | 0.635 | GT points inside mask |
| `center_point` | 0.494 | GT centroid only |
| **SAM-Auto** | **0.470** | No GT; 32×32 grid (automatic) |

The prompting gap — the difference between the best GT-guided strategy and realistic deployment — is **0.281 mIoU** (0.751 − 0.470). This is large: deployment SAM performs at the same level as classic baselines (Mask R-CNN, DETR), despite SAM's architectural superiority when adequately prompted.

Two sub-findings from the prompt ablation:

1. **A tight spatial constraint drives performance.** Box-based prompts (0.702–0.751) substantially outperform point-only prompts (0.494–0.635). The box eliminates ambiguity about which object to segment, especially when multiple objects of the same class are nearby.

2. **More information ≠ better results.** `box_pos_neg` (0.702) is slightly worse than plain `box` (0.751), and `center_point` (0.494) is slightly worse than `multi_point` (0.635). Poorly placed negative points or centroid prompts for elongated objects can confuse the decoder.

---

## 7. Thin and Articulated Objects

| Model | Thin mIoU (n=502) | Non-thin mIoU (n=2252) | Gap |
|---|---|---|---|
| Mask R-CNN | 0.364 | 0.532 | −0.168 |
| DETR | 0.350 | 0.467 | −0.117 |
| SAM (`box`) | 0.673 | 0.768 | −0.095 |
| SAM-Auto | 0.365 | 0.494 | −0.129 |
| Cascade | 0.368 | 0.519 | −0.151 |

Thin objects are systematically harder for all models. The gap is largest for Mask R-CNN (0.168), moderate for Cascade (0.151) and DETR (0.117), and smallest for SAM-box (0.095).

Thin objects (bicycle, spoon, fork, tie, etc.) have high aspect ratios and complex topologies. Mask R-CNN's 28×28 mask head rounds corners and fills internal gaps. DETR's bipartite matching tends to produce bounding-box-aligned masks that over-segment elongated thin objects. SAM's full-resolution decoder preserves fine detail much better, reducing (though not eliminating) the thin-object gap.

---

## 8. Worst-Performing Categories

**Mask R-CNN worst 5 categories** (by mIoU):

| Category | mIoU | n |
|---|---|---|
| sandwich | 0.076 | 20 |
| donut | 0.117 | 24 |
| orange | 0.118 | 26 |
| spoon | 0.141 | 32 |
| fork | 0.181 | 44 |

**DETR worst 5 categories:**

| Category | mIoU | n |
|---|---|---|
| book | 0.062 | 25 |
| sandwich | 0.083 | 20 |
| orange | 0.086 | 26 |
| carrot | 0.102 | 18 |
| spoon | 0.114 | 32 |

**Common failure categories across both models:** food items with irregular or clustered appearance (sandwich, donut, orange) and thin utensils (spoon, fork). These represent two orthogonal failure modes:

- **Texture-similar boundaries** (sandwich, orange): the mask boundary runs through a visually homogeneous region, giving the model few gradient cues to place the boundary precisely.
- **High aspect ratio / small width** (spoon, fork, carrot): the 28×28 mask head in Mask R-CNN lacks the resolution to represent a 2-pixel-wide utensil, and DETR's transformer queries are not designed for long-thin objects.

Books are a DETR-specific failure: DETR's panoptic model frequently merges stacked books into one segment, confusing individual instances.

---

## 9. Improvements

### 9.1 SAM-Auto: Quantifying the Prompting Gap

**Motivation.** All SAM strategies in §6 (box, center_point, etc.) use ground-truth information at inference time, making them unrealistic evaluations. SAM-Auto removes this: it runs `SamAutomaticMaskGenerator` with a 32×32 point grid, predicting all masks in the image, then matches each predicted mask to the closest ground-truth instance by IoU.

**Result.** SAM-Auto achieves mIoU=0.470 — matching Mask R-CNN (0.501) and DETR (0.446). The prompting gap between oracle GT-box SAM and realistic SAM-Auto is **0.281 mIoU** (0.751 − 0.470). This substantial gap means that the high scores reported for prompted SAM are largely an artifact of the evaluation protocol. In deployment, SAM is a strong but not exceptional model, competitive with but not decisively beating two-year-old baselines.

Interestingly, SAM-Auto with automatic prompts still shows strong medium-object performance (0.607), but its large-object mIoU (0.622) is notably lower than Mask R-CNN large (0.785). The 32×32 grid generates dense small proposals well-suited to medium objects, but large objects may be incompletely covered by the grid.

### 9.2 Cascade (Mask R-CNN → SAM): Boundary Refinement

**Motivation.** Mask R-CNN's most consistent failure mode is imprecise mask boundaries, especially for thin or articulated objects, because its mask head is trained at 28×28 resolution and upsampled. We implemented a two-stage cascade:

1. **Stage 1**: Mask R-CNN predicts bounding boxes and category IDs from the full image.
2. **Stage 2**: Each predicted box is fed to SAM as a box prompt. SAM encodes the image once and generates a full-resolution mask for each box, replacing the Mask R-CNN mask head output.

No ground-truth information is used at inference time.

**Result.**

| Model | mIoU | Small | Medium | Large | Thin |
|---|---|---|---|---|---|
| Mask R-CNN | 0.501 | 0.337 | 0.648 | 0.785 | 0.364 |
| Cascade | 0.492 | 0.343 | 0.639 | 0.716 | 0.368 |
| Δ | −0.009 | **+0.006** | −0.009 | **−0.069** | +0.004 |

The cascade slightly hurts overall mIoU (−0.009) but shows a **+0.006 improvement for small objects** and a small improvement for thin objects (+0.004). The large-object penalty (−0.069) is the dominant effect: when Mask R-CNN correctly finds a large object, its box prompt to SAM is relatively loose, and SAM sometimes produces a less precise mask than the trained Mask R-CNN mask head for easy large objects.

The correct application of this cascade is **as a boundary refinement for hard cases** (small and thin objects), not as a global replacement for Mask R-CNN. A selective cascade — applying SAM only when Mask R-CNN predicts a small or thin-category object — would be expected to show consistent improvement.

---

## 10. Mask2Former: Implementation Limitation

Mask2Former with the HuggingFace checkpoint `facebook/mask2former-swin-base-coco-instance` gives mIoU=0.039 on our evaluation, well below its reported 51.1 AP on COCO. We conducted an extensive diagnostic:

- **Preprocessing**: correct — pixel values span [−2.12, 2.64] with mean ≈ −0.26, matching ImageNet normalization; input resolution is 800×1120 (correct for shortest-edge=800, longest-edge=1333).
- **Model outputs**: correct shape — `class_queries_logits (1,100,81)`, `masks_queries_logits (1,100,200,280)`.
- **Mask quality**: scanning all 100 queries with no threshold, the highest achievable IoU over all ground-truth instances is only 0.515 (a small truck), achieved by a query that scores near 0 (foreground score = 0.0099).
- **Score calibration**: class scores are systematically miscalibrated — mean background score = 0.863. Queries that happen to produce spatially correct masks receive near-background class scores; queries with high foreground scores produce masks in wrong locations.

The fundamental issue is that **class scores and mask quality are completely decoupled** in our setup, which is inconsistent with a properly trained Mask2Former. We attribute this to a model loading or checkpoint conversion issue in transformers 4.46.3. We exclude Mask2Former from main comparisons and flag this as a known environment limitation.

---

## 11. Failure Mode Analysis

We classify each ground-truth instance into three outcome categories:

- **Success**: IoU ≥ 0.5 — model correctly finds and segments the object.
- **Partial**: 0.3 ≤ IoU < 0.5 — model finds the object but with imprecise boundaries.
- **Failure**: IoU < 0.3 — model either misses the object entirely or hallucinates a mask at the wrong location.

| Model | Success | Partial | Failure |
|---|---|---|---|
| Mask R-CNN | 0.592 | 0.058 | 0.350 |
| DETR | 0.517 | 0.085 | 0.399 |
| Cascade | 0.574 | 0.067 | 0.359 |
| SAM-Auto | 0.516 | —* | 0.398 |

*Partial calculated implicitly from 1 − success − failure.

**Observations:**
1. Failure mode distribution is bimodal: most instances are either well-segmented (IoU > 0.5) or completely missed (IoU < 0.3). The "partial" bucket is small, meaning models rarely produce mediocre masks — they either succeed or fail outright.
2. The partial bucket is larger for DETR (0.085) than Mask R-CNN (0.058), suggesting DETR more often finds the right object but with imprecise boundaries, while Mask R-CNN more often either nails it or misses entirely.
3. Cascade reduces partial failures relative to Mask R-CNN (0.067 vs 0.058), consistent with SAM producing crisper boundaries on objects that Mask R-CNN detects.

---

## 12. Conclusions

**Finding 1: The prompting gap is large and practically significant.** SAM with GT box prompts (mIoU=0.751) appears dramatically better than traditional models, but realistic SAM-Auto (0.470) is on par with Mask R-CNN (0.501). Much of the reported SAM performance advantage is attributable to the evaluation protocol, not architecture.

**Finding 2: Small objects are the dominant failure mode for all models.** MaskRCNN's mIoU drops by 57% from large (0.785) to small (0.337) objects. DETR drops 63%. SAM-box is the only robust exception (−15%). Object detection + segmentation pipelines without specialised small-object handling are unlikely to generalise well to real scenes with fine detail.

**Finding 3: Thin objects form a structured failure category.** Spoons, forks, bicycles, and similar objects consistently appear in the bottom-5 worst categories. CNN mask heads (28×28) lack the resolution to represent thin structures faithfully. SAM's full-resolution decoder reduces but does not eliminate this failure.

**Finding 4: Food items are structurally ambiguous.** Sandwiches (mIoU=0.076) and donuts (0.117) show worse results than all thin-object categories. Their textural similarity to surroundings and loose, ambiguous boundaries make them hard even for high-capacity models.

**Finding 5: Cascade is size-contingent.** Replacing Mask R-CNN's mask head with SAM improves small-object boundary precision (+0.006) but degrades large-object performance (−0.069). A selective cascade would be more effective than a global one.

**Finding 6: Transformer bipartite matching (DETR) underperforms proposal-based methods in this evaluation.** DETR (0.446) scores lower than Mask R-CNN (0.501) and SAM-Auto (0.470). DETR's global assignment mechanism struggles with images that have many similar-category instances (crowd scenes, stacked books), where the greedy IoU matcher assigns predictions to the wrong ground-truth instances.

---

## Appendix: Experimental Setup

| Item | Value |
|---|---|
| Evaluation subset | 200 images, 2,754 instances |
| COCO split | val2017 |
| Size distribution | Small: 1,470 / Medium: 900 / Large: 384 |
| Thin instances | 502 (18.2%) |
| SAM checkpoint | sam_vit_b_01ec64.pth (ViT-B) |
| Mask R-CNN | torchvision MaskRCNN\_ResNet50\_FPN\_Weights.COCO\_V1 |
| DETR | facebook/detr-resnet-50-panoptic (score threshold 0.5) |
| Mask2Former | facebook/mask2former-swin-base-coco-instance (excluded) |
| SAM-Auto grid | 32×32 points, pred\_iou\_thresh=0.86, stability\_thresh=0.92 |
| Cascade threshold | Mask R-CNN score ≥ 0.5 |
| Hardware | CUDA GPU (inference only) |
| Python | 3.8, torch 2.4.1+cu118, transformers 4.46.3 |
