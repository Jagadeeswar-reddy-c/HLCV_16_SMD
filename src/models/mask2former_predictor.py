"""
Mask2Former wrapper using HuggingFace Transformers.

Uses facebook/mask2former-swin-base-coco-instance (instance segmentation).
"""

from typing import List, Tuple
import numpy as np
import torch


HF_MODEL_ID = "facebook/mask2former-swin-base-coco-instance"

# HuggingFace returns 0-indexed label ids; this maps them to COCO category ids.
# COCO has 80 classes but non-contiguous ids (12, 26, 29, 30 etc. are skipped).
_COCO_LABEL_TO_CAT_ID = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
    18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28,
    26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38,
    34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47,
    42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63,
    58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75,
    66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84,
    74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90,
}


class Mask2FormerPredictor:
    def __init__(self, device: str = "cpu", score_threshold: float = 0.1):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        # transformers ≥4.40 dropped _max_size; the model was trained with
        # shortest_edge=800 / longest_edge=1333 (COCO standard). Without this
        # override the processor defaults to a fixed 384×384 square, which
        # produces degenerate full-image masks.
        self.processor = AutoImageProcessor.from_pretrained(
            HF_MODEL_ID,
            size={"shortest_edge": 800, "longest_edge": 1333},
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(HF_MODEL_ID)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.score_threshold = score_threshold

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Run Mask2Former on a single RGB image (H×W×3, uint8).

        Returns:
            masks:        list of binary H×W bool arrays (original image resolution)
            scores:       confidence score per mask
            category_ids: COCO category id per mask

        Decoding strategy: we bypass post_process_instance_segmentation, which
        uses panoptic pixel-assignment (each pixel → one segment only).  That
        collapses overlapping instances into non-overlapping fragments and gives
        near-zero IoU for occluded / stacked objects.  Instead we:
          1. Take class softmax scores (excluding the background/no-object class)
          2. Upsample mask logits with bilinear interpolation to the original size
          3. Threshold each query's mask independently (sigmoid > 0.5)
        This yields true independent binary masks that can overlap, matching how
        COCO instance masks are defined.
        """
        import torch.nn.functional as F
        from PIL import Image

        H, W = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # class_queries_logits: (1, Q, C+1)  — last class is no-object/background
        # masks_queries_logits: (1, Q, H/4, W/4)
        class_logits = outputs.class_queries_logits[0]          # (Q, C+1)
        class_probs  = class_logits.softmax(dim=-1)             # (Q, C+1)
        # Exclude background (last index) to get per-class scores
        fg_scores, label_ids = class_probs[:, :-1].max(dim=-1)  # (Q,), (Q,)

        # Upsample mask logits to original image size
        mask_logits = outputs.masks_queries_logits[0]           # (Q, h, w)
        mask_logits_up = F.interpolate(
            mask_logits.unsqueeze(0).float(),   # (1, Q, h, w)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                            # (Q, H, W)
        mask_probs = mask_logits_up.sigmoid()   # (Q, H, W)

        masks, scores, category_ids = [], [], []
        for q in range(fg_scores.shape[0]):
            score = float(fg_scores[q])
            if score < self.score_threshold:
                continue
            label_id  = int(label_ids[q])
            coco_cat_id = _COCO_LABEL_TO_CAT_ID.get(label_id, label_id)
            binary_mask = (mask_probs[q] > 0.5).cpu().numpy()
            if binary_mask.sum() < 64:   # skip dust-size masks
                continue
            masks.append(binary_mask)
            scores.append(score)
            category_ids.append(coco_cat_id)

        return masks, scores, category_ids
