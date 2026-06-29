"""
SAM wrapper supporting all five prompt strategies.

Prompt strategies:
    box          — GT bounding box only
    center_point — single point at mask/bbox centroid
    multi_point  — n random points sampled inside GT mask
    box_point    — GT box + center point
    box_pos_neg  — GT box + positive inside + negative near boundary
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.utils import sample_points_inside_mask, sample_negative_points_near_boundary

PROMPT_STRATEGIES = [
    "box",
    "center_point",
    "multi_point",
    "box_point",
    "box_pos_neg",
]


class SAMPredictor:
    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cpu"):
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self._predictor = SamPredictor(sam)
        self.device = device

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._predictor.set_image(image_rgb)

    def predict_with_strategy(
        self,
        gt_mask: np.ndarray,
        bbox_xywh: List[float],
        strategy: str,
        n_pos_points: int = 5,
        n_neg_points: int = 3,
        multimask_output: bool = True,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Run SAM with the given prompt strategy and return the best binary mask (H×W, bool).
        """
        if strategy not in PROMPT_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {PROMPT_STRATEGIES}")

        x, y, w, h = bbox_xywh
        box_xyxy = np.array([x, y, x + w, y + h])
        center = np.array([[x + w / 2, y + h / 2]])

        # Build prompt inputs
        input_box = None
        input_points = None
        input_labels = None

        if strategy == "box":
            input_box = box_xyxy

        elif strategy == "center_point":
            input_points = center
            input_labels = np.array([1])

        elif strategy == "multi_point":
            pts = sample_points_inside_mask(gt_mask, n=n_pos_points, seed=seed)
            input_points = pts.astype(float)
            input_labels = np.ones(len(pts), dtype=int)

        elif strategy == "box_point":
            input_box = box_xyxy
            input_points = center
            input_labels = np.array([1])

        elif strategy == "box_pos_neg":
            pos_pts = sample_points_inside_mask(gt_mask, n=n_pos_points, seed=seed)
            neg_pts = sample_negative_points_near_boundary(gt_mask, n=n_neg_points, seed=seed)
            input_box = box_xyxy
            input_points = np.vstack([pos_pts, neg_pts]).astype(float)
            input_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts))

        masks, scores, _ = self._predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=multimask_output,
        )

        # Pick the highest-confidence mask
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)

    def predict_all_strategies(
        self,
        gt_mask: np.ndarray,
        bbox_xywh: List[float],
        n_pos_points: int = 5,
        n_neg_points: int = 3,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Run all 5 strategies and return a dict strategy → binary mask."""
        return {
            s: self.predict_with_strategy(
                gt_mask, bbox_xywh, s,
                n_pos_points=n_pos_points,
                n_neg_points=n_neg_points,
                seed=seed,
            )
            for s in PROMPT_STRATEGIES
        }
