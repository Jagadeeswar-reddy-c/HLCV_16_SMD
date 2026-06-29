"""
Segmentation evaluation metrics.

All functions operate on binary numpy arrays (uint8 or bool, H×W).
"""

from typing import Dict, List, Optional
import numpy as np


# ------------------------------------------------------------------
# Per-object IoU
# ------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


# ------------------------------------------------------------------
# Aggregate metrics over a list of per-object IoU scores
# ------------------------------------------------------------------

def mean_iou(ious: List[float]) -> float:
    if not ious:
        return float("nan")
    return float(np.mean(ious))


def success_rate(ious: List[float], threshold: float = 0.5) -> float:
    """Fraction of objects with IoU >= threshold."""
    if not ious:
        return float("nan")
    return float(np.mean([iou >= threshold for iou in ious]))


def failure_rate(ious: List[float], threshold: float = 0.3) -> float:
    """Fraction of objects with IoU < threshold."""
    if not ious:
        return float("nan")
    return float(np.mean([iou < threshold for iou in ious]))


# ------------------------------------------------------------------
# Grouped metrics
# ------------------------------------------------------------------

def metrics_by_group(
    ious: List[float],
    groups: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mIoU, success rate, and failure rate for each unique group label.

    Args:
        ious:   per-object IoU list
        groups: same-length list of group labels (e.g., size or category name)

    Returns:
        dict  group_name → {"miou": float, "success": float, "failure": float, "n": int}
    """
    from collections import defaultdict
    bucket: Dict[str, List[float]] = defaultdict(list)
    for iou, g in zip(ious, groups):
        bucket[g].append(iou)

    result = {}
    for g, vals in sorted(bucket.items()):
        result[g] = {
            "miou": mean_iou(vals),
            "success": success_rate(vals),
            "failure": failure_rate(vals),
            "n": len(vals),
        }
    return result


def summarise_results(records_with_iou: List[Dict]) -> Dict:
    """
    Convenience wrapper: given a list of dicts each containing
    {iou, size, category_name, is_thin}, return a full summary dict.
    """
    ious = [r["iou"] for r in records_with_iou]
    sizes = [r["size"] for r in records_with_iou]
    cats = [r["category_name"] for r in records_with_iou]

    return {
        "overall": {
            "miou": mean_iou(ious),
            "success_rate": success_rate(ious),
            "failure_rate": failure_rate(ious),
            "n": len(ious),
        },
        "by_size": metrics_by_group(ious, sizes),
        "by_category": metrics_by_group(ious, cats),
        "thin_only": {
            "miou": mean_iou([r["iou"] for r in records_with_iou if r.get("is_thin")]),
            "n": sum(1 for r in records_with_iou if r.get("is_thin")),
        },
    }
