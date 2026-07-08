"""
Cascade: Mask R-CNN detect → SAM segment.

Improvement over standalone models:
  - Mask R-CNN localises objects and classifies categories (no GT needed)
  - SAM refines the mask from each predicted bounding box (better boundaries
    than Mask R-CNN's mask head, which is trained at 28×28 resolution)

This is a concrete answer to the "propose improvements that challenge common
failure cases" requirement: Mask R-CNN's main failure mode is imprecise mask
boundaries (especially for thin / articulated objects). SAM operates at full
image resolution and is architecturally superior at boundary delineation given
a tight box prompt.

No ground-truth information is used at inference time.
"""

from typing import List, Tuple
import numpy as np
import torch


TORCHVISION_TO_COCO_ID = {i: i for i in range(1, 91)}


class CascadePredictor:
    """
    Two-stage pipeline:
      1. Mask R-CNN (torchvision) → predicted bounding boxes + category IDs
      2. SAM (segment-anything) → high-fidelity mask for each predicted box
    """

    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        device: str = "cpu",
        score_threshold: float = 0.5,
    ):
        import torchvision
        from segment_anything import sam_model_registry, SamPredictor

        # Stage 1 — Mask R-CNN for object detection & classification
        self.mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        )
        self.mrcnn.eval().to(device)

        # Stage 2 — SAM for mask refinement from box prompts
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.sam = SamPredictor(sam)

        self.device = device
        self.score_threshold = score_threshold

    @torch.no_grad()
    def predict(
        self, image_rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Run cascade on a single RGB image (H×W×3, uint8).

        Returns:
            masks:        list of binary H×W bool arrays (SAM-refined)
            scores:       Mask R-CNN confidence scores
            category_ids: COCO category IDs from Mask R-CNN
        """
        import torchvision.transforms.functional as TF

        # ── Stage 1: Mask R-CNN detection ───────────────────────────────────
        tensor = TF.to_tensor(image_rgb).to(self.device)
        output = self.mrcnn([tensor])[0]

        keep_idx = (output["scores"] >= self.score_threshold).nonzero(as_tuple=True)[0]
        if len(keep_idx) == 0:
            return [], [], []

        boxes  = output["boxes"][keep_idx].cpu().numpy()   # (N, 4) xyxy
        scores = output["scores"][keep_idx].cpu().tolist()
        labels = output["labels"][keep_idx].cpu().tolist()

        category_ids = [TORCHVISION_TO_COCO_ID.get(int(l), int(l)) for l in labels]

        # ── Stage 2: SAM mask refinement ────────────────────────────────────
        self.sam.set_image(image_rgb)   # encodes image once for all boxes

        masks = []
        for box_xyxy in boxes:
            # SamPredictor.predict() expects box as (4,) numpy float array [x1,y1,x2,y2]
            sam_masks, sam_iou_preds, _ = self.sam.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy.astype(float),   # (4,) numpy array
                multimask_output=False,
            )
            # sam_masks: (1, H, W) bool
            masks.append(sam_masks[0].astype(bool))

        return masks, scores, category_ids
