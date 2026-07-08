"""
DETR (panoptic) wrapper using HuggingFace Transformers.

Uses facebook/detr-resnet-50-panoptic. Extracts only the "thing" (instance)
segments from the panoptic output, discarding "stuff" segments (sky, road, …).

Architecture family: DETR — end-to-end transformer with bipartite matching,
no anchor boxes or NMS during inference.
"""

from typing import List, Tuple
import numpy as np
import torch

HF_MODEL_ID = "facebook/detr-resnet-50-panoptic"

# COCO category name → official COCO instance category ID.
# Used to convert DETR's id2label names to the same ID space as GT.
_COCO_NAME_TO_CAT_ID = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15,
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21,
    "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27,
    "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34,
    "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38,
    "baseball bat": 39, "baseball glove": 40, "skateboard": 41,
    "surfboard": 42, "tennis racket": 43, "bottle": 44, "wine glass": 46,
    "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51,
    "banana": 52, "apple": 53, "sandwich": 54, "orange": 55, "broccoli": 56,
    "carrot": 57, "hot dog": 58, "pizza": 59, "donut": 60, "cake": 61,
    "chair": 62, "couch": 63, "potted plant": 64, "bed": 65,
    "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73, "mouse": 74,
    "remote": 75, "keyboard": 76, "cell phone": 77, "microwave": 78,
    "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82, "book": 84,
    "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88,
    "hair drier": 89, "hair dryer": 89, "toothbrush": 90,
}


class DETRPredictor:
    def __init__(self, device: str = "cpu", score_threshold: float = 0.5):
        from transformers import AutoImageProcessor, DetrForSegmentation

        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        self.model = DetrForSegmentation.from_pretrained(HF_MODEL_ID)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.score_threshold = score_threshold

        # Build label_id → COCO cat ID lookup from the model's own id2label
        self._label_to_coco: dict = {}
        for lid, name in self.model.config.id2label.items():
            coco_id = _COCO_NAME_TO_CAT_ID.get(name.lower())
            if coco_id is not None:
                self._label_to_coco[int(lid)] = coco_id

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Run DETR panoptic on a single RGB image (H×W×3, uint8).

        Returns:
            masks:        list of binary H×W bool arrays
            scores:       confidence score per mask
            category_ids: COCO category id per mask
        """
        from PIL import Image

        H, W = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        results = self.processor.post_process_panoptic_segmentation(
            outputs,
            threshold=self.score_threshold,
            target_sizes=[(H, W)],
        )[0]

        panoptic_map = results["segmentation"]  # (H, W) int tensor

        masks, scores, category_ids = [], [], []
        for seg in results["segments_info"]:
            # Keep only "thing" segments (instance-level categories)
            if not seg.get("is_thing", True):
                continue

            label_id = seg["label_id"]
            score = seg.get("score", 1.0)

            coco_id = self._label_to_coco.get(int(label_id), int(label_id))
            mask = (panoptic_map == seg["id"]).cpu().numpy().astype(bool)

            if mask.sum() < 64:
                continue

            masks.append(mask)
            scores.append(float(score))
            category_ids.append(coco_id)

        return masks, scores, category_ids
