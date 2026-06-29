"""
Mask2Former wrapper using HuggingFace Transformers.

Uses facebook/mask2former-swin-base-coco-instance (instance segmentation).
The model outputs COCO category ids directly via the processor's post-processing.
"""

from typing import List, Tuple
import numpy as np
import torch


HF_MODEL_ID = "facebook/mask2former-swin-base-coco-instance"


class Mask2FormerPredictor:
    def __init__(self, device: str = "cpu", score_threshold: float = 0.5):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
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
        """
        from PIL import Image

        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # post_process_instance_segmentation returns per-image results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            target_sizes=[pil_img.size[::-1]],  # (H, W)
        )[0]

        masks, scores, category_ids = [], [], []
        for seg_info in results["segments_info"]:
            seg_id = seg_info["id"]
            score = seg_info["score"]
            label_id = seg_info["label_id"]  # COCO category id

            mask = (results["segmentation"] == seg_id).cpu().numpy().astype(bool)
            masks.append(mask)
            scores.append(float(score))
            category_ids.append(int(label_id))

        return masks, scores, category_ids
