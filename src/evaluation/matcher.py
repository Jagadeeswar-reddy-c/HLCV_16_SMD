"""
Match model predictions to ground-truth instances.

Used for auto-segmentation models (Mask2Former, Mask R-CNN) that output
many predictions per image without knowing which GT object they correspond to.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.evaluation.metrics import compute_iou


def match_predictions_to_gt(
    gt_mask: np.ndarray,
    gt_category_id: int,
    pred_masks: List[np.ndarray],
    pred_scores: List[float],
    pred_category_ids: List[int],
    score_threshold: float = 0.5,
    require_category_match: bool = True,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Given a single GT mask, find the best matching predicted mask.

    Args:
        gt_mask:               H×W binary GT mask
        gt_category_id:        COCO category id of the GT object
        pred_masks:            list of H×W binary predicted masks
        pred_scores:           confidence score per prediction
        pred_category_ids:     predicted COCO category id per prediction
        score_threshold:       discard predictions with score < this
        require_category_match: if True, only consider predictions with
                                matching category (falls back to all if none match)

    Returns:
        (best_pred_mask, iou)  —  None mask and 0.0 IoU if no prediction found
    """
    if not pred_masks:
        return None, 0.0

    # Filter by score
    candidates = [
        (pmask, pscore, pcat)
        for pmask, pscore, pcat in zip(pred_masks, pred_scores, pred_category_ids)
        if pscore >= score_threshold
    ]

    if not candidates:
        return None, 0.0

    # Prefer same-category predictions
    if require_category_match:
        cat_candidates = [(m, s, c) for m, s, c in candidates if c == gt_category_id]
        if cat_candidates:
            candidates = cat_candidates

    # Pick highest-IoU prediction
    best_mask, best_iou = None, -1.0
    for pmask, _, _ in candidates:
        iou = compute_iou(pmask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_mask = pmask

    return best_mask, max(best_iou, 0.0)


def match_all_gt_instances(
    gt_instances: List[Dict],
    pred_masks: List[np.ndarray],
    pred_scores: List[float],
    pred_category_ids: List[int],
    score_threshold: float = 0.5,
) -> List[Dict]:
    """
    Match every GT instance in an image to the best prediction.

    Args:
        gt_instances: list of dicts, each with keys:
            {ann_id, category_id, gt_mask (H×W), size, category_name, is_thin, area}
        pred_masks / pred_scores / pred_category_ids:
            parallel lists from the model output

    Returns:
        list of dicts with original fields + {pred_mask, iou}
    """
    results = []
    for gt in gt_instances:
        pred_mask, iou = match_predictions_to_gt(
            gt_mask=gt["gt_mask"],
            gt_category_id=gt["category_id"],
            pred_masks=pred_masks,
            pred_scores=pred_scores,
            pred_category_ids=pred_category_ids,
            score_threshold=score_threshold,
        )
        results.append({**gt, "pred_mask": pred_mask, "iou": iou})
    return results
