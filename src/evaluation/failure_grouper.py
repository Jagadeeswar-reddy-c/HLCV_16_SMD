"""
Group evaluation results into failure categories for analysis.

Operates on the flat per-instance result lists produced by the experiment runners.
"""

from typing import Dict, List, Tuple

from src.dataset.instance import THIN_CATEGORIES
from src.evaluation.metrics import metrics_by_group, mean_iou, success_rate, failure_rate

# IoU thresholds
SUCCESS_THRESHOLD = 0.5
FAILURE_THRESHOLD = 0.3

# Categories that represent crowded-scene proxies (dense, overlapping)
CROWDED_PROXY_CATEGORIES = {"person", "chair", "bottle", "cup", "banana", "apple"}


def classify_failure_mode(iou: float, pred_mask, gt_mask) -> str:
    """
    Heuristic label for the failure mode of a single prediction.
    Requires numpy masks; returns 'success' if IoU >= SUCCESS_THRESHOLD.
    """
    import numpy as np

    if iou >= SUCCESS_THRESHOLD:
        return "success"

    if pred_mask is None:
        return "missing"

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    pred_area = pred.sum()
    gt_area = gt.sum()

    if gt_area == 0:
        return "empty_gt"

    ratio = pred_area / gt_area if gt_area > 0 else 0.0

    if ratio > 1.5:
        return "over_segmentation"
    if ratio < 0.5:
        return "under_segmentation"

    # Check object merging: many prediction pixels outside GT but still large coverage
    outside_gt = (pred & ~gt).sum()
    if outside_gt / pred_area > 0.4:
        return "boundary_confusion"

    return "other"


def group_by_failure_mode(results: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate per-instance results by failure mode.
    Each dict must have at least: {iou, size, category_name, is_thin}
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for r in results:
        iou = r["iou"]
        if iou >= SUCCESS_THRESHOLD:
            mode = "success"
        elif iou < FAILURE_THRESHOLD:
            mode = "hard_failure"
        else:
            mode = "partial"
        buckets[mode].append(iou)

    return {
        mode: {
            "miou": mean_iou(vals),
            "n": len(vals),
            "success_rate": success_rate(vals),
            "failure_rate": failure_rate(vals),
        }
        for mode, vals in buckets.items()
    }


def worst_categories(results: List[Dict], n: int = 10) -> List[Tuple[str, float]]:
    """Return the n categories with the lowest mIoU."""
    by_cat = metrics_by_group(
        [r["iou"] for r in results],
        [r["category_name"] for r in results],
    )
    ranked = sorted(by_cat.items(), key=lambda kv: kv[1]["miou"])
    return [(cat, info["miou"]) for cat, info in ranked[:n]]


def thin_vs_nonthin_summary(results: List[Dict]) -> Dict:
    thin = [r["iou"] for r in results if r.get("is_thin")]
    nonthin = [r["iou"] for r in results if not r.get("is_thin")]
    return {
        "thin": {"miou": mean_iou(thin), "n": len(thin)},
        "non_thin": {"miou": mean_iou(nonthin), "n": len(nonthin)},
    }


def full_failure_report(results: List[Dict]) -> Dict:
    """
    One-shot report combining all groupings used in the paper.
    """
    ious = [r["iou"] for r in results]
    sizes = [r["size"] for r in results]
    cats = [r["category_name"] for r in results]

    return {
        "by_size": metrics_by_group(ious, sizes),
        "by_category": metrics_by_group(ious, cats),
        "worst_categories": worst_categories(results),
        "thin_vs_nonthin": thin_vs_nonthin_summary(results),
        "failure_modes": group_by_failure_mode(results),
    }
