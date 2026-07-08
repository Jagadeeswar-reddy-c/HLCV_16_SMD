import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
Generate all plots for the report.

CLI:
    python src/visualization/plot.py \
        --sam       results/sam_results.json \
        --maskrcnn  results/maskrcnn_results.json \
        --mask2former results/mask2former_results.json \
        --output    figures/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils import ensure_dir

# Palette — one colour per logical series
COLORS = {
    "SAM (box)":              "#4C72B0",
    "SAM (center_point)":     "#64B5CD",
    "SAM (multi_point)":      "#8172B2",
    "SAM (box_point)":        "#CCB974",
    "SAM (box_pos_neg)":      "#55A868",
    "SAM-Auto":               "#2196F3",
    "MaskRCNN":               "#C44E52",
    "Mask2Former":            "#DD8452",
    "DETR":                   "#937860",
    "Cascade (MaskRCNN→SAM)": "#7B68EE",
}
SIZE_ORDER = ["small", "medium", "large"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _by_size_key(summary_node: Dict, size: str, key: str, default: float = 0.0) -> float:
    return summary_node.get("by_size", {}).get(size, {}).get(key, default)


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ------------------------------------------------------------------
# 1. Overall mIoU — all models + all SAM strategies
# ------------------------------------------------------------------

def plot_overall_miou(model_results: Dict[str, float], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    models = list(model_results.keys())
    mious = [model_results[m] for m in models]
    colors = [COLORS.get(m, "#888888") for m in models]
    bars = ax.bar(models, mious, color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall mIoU by Model / Prompt Strategy")
    plt.xticks(rotation=20, ha="right")
    _save(fig, output_path)


# ------------------------------------------------------------------
# 2. mIoU by object size
# ------------------------------------------------------------------

def plot_miou_by_size(
    model_size_results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    models = list(model_size_results.keys())
    x = np.arange(len(SIZE_ORDER))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [model_size_results[model].get(s, 0.0) for s in SIZE_ORDER]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=model, color=COLORS.get(model, f"C{i}"))

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("mIoU by Object Size")
    ax.legend(fontsize=8)
    _save(fig, output_path)


# ------------------------------------------------------------------
# 3. Failure rate by object size
# ------------------------------------------------------------------

def plot_failure_rate_by_size(
    model_failure_rates: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    models = list(model_failure_rates.keys())
    x = np.arange(len(SIZE_ORDER))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        vals = [model_failure_rates[model].get(s, 0.0) for s in SIZE_ORDER]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=model, color=COLORS.get(model, f"C{i}"))

    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_ylabel("Failure Rate (IoU < 0.3)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Failure Rate by Object Size")
    ax.legend(fontsize=8)
    _save(fig, output_path)


# ------------------------------------------------------------------
# 4. Worst categories (horizontal bar) — one figure per model
# ------------------------------------------------------------------

def plot_worst_categories(
    category_ious: Dict[str, float],
    output_path: str,
    title: str = "Worst Categories",
    n: int = 10,
) -> None:
    ranked = sorted(category_ious.items(), key=lambda kv: kv[1])[:n]
    if not ranked:
        return
    cats, vals = zip(*ranked)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(cats))
    ax.barh(y, vals, color="#C44E52")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("mIoU")
    ax.set_xlim(0, max(max(vals) * 1.15, 0.05))
    ax.set_title(title)
    _save(fig, output_path)


# ------------------------------------------------------------------
# 5. SAM prompt strategy comparison
# ------------------------------------------------------------------

def plot_sam_prompt_comparison(
    strategy_mious: Dict[str, float],
    output_path: str,
) -> None:
    strategies = list(strategy_mious.keys())
    vals = [strategy_mious[s] for s in strategies]
    colors = [COLORS.get(f"SAM ({s})", "#4C72B0") for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(strategies, vals, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("SAM Prompt Strategy Comparison")
    plt.xticks(rotation=15, ha="right")
    _save(fig, output_path)


# ------------------------------------------------------------------
# 6. Thin vs non-thin objects
# ------------------------------------------------------------------

def plot_thin_vs_nonthin(
    model_thin_data: Dict[str, Dict],
    output_path: str,
) -> None:
    """
    model_thin_data: {model_name: {"thin": miou, "non_thin": miou}}
    """
    models = list(model_thin_data.keys())
    x = np.arange(len(models))
    width = 0.35

    thin_vals = [model_thin_data[m].get("thin", 0.0) for m in models]
    nonthin_vals = [model_thin_data[m].get("non_thin", 0.0) for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, thin_vals,   width, label="Thin objects",     color="#C44E52", alpha=0.85)
    ax.bar(x + width / 2, nonthin_vals, width, label="Non-thin objects", color="#4C72B0", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("mIoU: Thin vs Non-Thin Objects")
    ax.legend()
    _save(fig, output_path)


# ------------------------------------------------------------------
# 7. Failure mode distribution (stacked bar)
# ------------------------------------------------------------------

def plot_failure_modes(
    model_mode_data: Dict[str, Dict[str, int]],
    output_path: str,
) -> None:
    """
    model_mode_data: {model_name: {"success": n, "partial": n, "hard_failure": n}}
    """
    modes = ["success", "partial", "hard_failure"]
    mode_colors = {"success": "#55A868", "partial": "#CCB974", "hard_failure": "#C44E52"}
    mode_labels = {"success": "Success (IoU≥0.5)", "partial": "Partial (0.3–0.5)", "hard_failure": "Failure (IoU<0.3)"}

    models = list(model_mode_data.keys())
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(models))
    for mode in modes:
        vals = []
        for m in models:
            total = sum(model_mode_data[m].get(mm, 0) for mm in modes)
            count = model_mode_data[m].get(mode, 0)
            vals.append(count / total if total > 0 else 0.0)
        ax.bar(x, vals, bottom=bottoms, color=mode_colors[mode],
               label=mode_labels[mode], width=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Fraction of instances")
    ax.set_ylim(0, 1.05)
    ax.set_title("Failure Mode Distribution")
    ax.legend(loc="lower right")
    _save(fig, output_path)


# ------------------------------------------------------------------
# 8. Improvement comparison: MaskRCNN → Cascade → SAM (GT oracle)
# ------------------------------------------------------------------

def plot_improvement_comparison(
    model_size_results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """
    Side-by-side grouped bars showing how cascade improves over Mask R-CNN
    across object sizes, benchmarked against the SAM GT-box oracle.

    model_size_results: {model_name: {"small": miou, "medium": miou, "large": miou, "overall": miou}}
    """
    size_labels = ["small", "medium", "large", "overall"]
    models = list(model_size_results.keys())
    x = np.arange(len(size_labels))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [model_size_results[model].get(s, 0.0) for s in size_labels]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model, color=COLORS.get(model, f"C{i}"))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.15)
    ax.set_title("Improvement: Mask R-CNN → Cascade (MaskRCNN→SAM) vs Oracle (SAM+GT-box)")
    ax.legend(fontsize=8)
    _save(fig, output_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def load_json(path: str) -> Optional[Dict]:
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def _thin_miou_from_data(data: Dict, strategy: Optional[str] = None) -> Dict:
    """Extract thin/non_thin mIoU from a result dict."""
    if strategy:
        node = data["summary"].get(strategy, {})
    else:
        node = data["summary"]

    thin_miou = node.get("thin_only", {}).get("miou", float("nan"))
    # non-thin: derive from overall minus thin contribution
    all_ious = data["per_instance"]
    if strategy:
        all_ious = [r for r in all_ious if r.get("strategy") == strategy]
    thin_vals   = [r["iou"] for r in all_ious if r.get("is_thin")]
    nonthin_vals = [r["iou"] for r in all_ious if not r.get("is_thin")]
    return {
        "thin":     float(np.mean(thin_vals))    if thin_vals    else float("nan"),
        "non_thin": float(np.mean(nonthin_vals)) if nonthin_vals else float("nan"),
    }


def _failure_mode_counts(instances: List[Dict]) -> Dict[str, int]:
    counts = {"success": 0, "partial": 0, "hard_failure": 0}
    for r in instances:
        iou = r["iou"]
        if iou >= 0.5:
            counts["success"] += 1
        elif iou >= 0.3:
            counts["partial"] += 1
        else:
            counts["hard_failure"] += 1
    return counts


def _flat_thin_entry(instances: List[Dict]) -> Dict[str, float]:
    thin_vals    = [r["iou"] for r in instances if r.get("is_thin")]
    nonthin_vals = [r["iou"] for r in instances if not r.get("is_thin")]
    return {
        "thin":     float(np.mean(thin_vals))    if thin_vals    else float("nan"),
        "non_thin": float(np.mean(nonthin_vals)) if nonthin_vals else float("nan"),
    }


def run(args):
    ensure_dir(args.output)

    sam_data      = load_json(args.sam)
    mrcnn_data    = load_json(args.maskrcnn)
    m2f_data      = load_json(args.mask2former)
    detr_data     = load_json(args.detr)
    sam_auto_data = load_json(args.sam_auto)
    cascade_data  = load_json(args.cascade)

    # ── 1. Overall mIoU (all SAM strategies + Mask R-CNN + Mask2Former) ──
    overall = {}
    if sam_data:
        for strategy in sam_data["summary"]:
            overall[f"SAM ({strategy})"] = sam_data["summary"][strategy]["overall"]["miou"]
    if mrcnn_data:
        overall["MaskRCNN"] = mrcnn_data["summary"]["overall"]["miou"]
    if m2f_data:
        overall["Mask2Former"] = m2f_data["summary"]["overall"]["miou"]
    if detr_data:
        overall["DETR"] = detr_data["summary"]["overall"]["miou"]
    if sam_auto_data:
        overall["SAM-Auto"] = sam_auto_data["summary"]["overall"]["miou"]
    if cascade_data:
        overall["Cascade (MaskRCNN→SAM)"] = cascade_data["summary"]["overall"]["miou"]
    if overall:
        plot_overall_miou(overall, os.path.join(args.output, "overall_miou.png"))

    # ── 2. mIoU by size ──
    size_data = {}
    if sam_data:
        for strategy in ["box", "box_pos_neg"]:
            if strategy in sam_data["summary"]:
                size_data[f"SAM ({strategy})"] = {
                    sz: _by_size_key(sam_data["summary"][strategy], sz, "miou")
                    for sz in SIZE_ORDER
                }
    if mrcnn_data:
        size_data["MaskRCNN"] = {
            sz: _by_size_key(mrcnn_data["summary"], sz, "miou") for sz in SIZE_ORDER
        }
    if m2f_data:
        size_data["Mask2Former"] = {
            sz: _by_size_key(m2f_data["summary"], sz, "miou") for sz in SIZE_ORDER
        }
    if detr_data:
        size_data["DETR"] = {
            sz: _by_size_key(detr_data["summary"], sz, "miou") for sz in SIZE_ORDER
        }
    if sam_auto_data:
        size_data["SAM-Auto"] = {
            sz: _by_size_key(sam_auto_data["summary"], sz, "miou") for sz in SIZE_ORDER
        }
    if cascade_data:
        size_data["Cascade (MaskRCNN→SAM)"] = {
            sz: _by_size_key(cascade_data["summary"], sz, "miou") for sz in SIZE_ORDER
        }
    if size_data:
        plot_miou_by_size(size_data, os.path.join(args.output, "miou_by_size.png"))

    # ── 3. Failure rate by size ──
    failure_rate_data = {}
    if mrcnn_data:
        failure_rate_data["MaskRCNN"] = {
            sz: _by_size_key(mrcnn_data["summary"], sz, "failure") for sz in SIZE_ORDER
        }
    if m2f_data:
        failure_rate_data["Mask2Former"] = {
            sz: _by_size_key(m2f_data["summary"], sz, "failure") for sz in SIZE_ORDER
        }
    if detr_data:
        failure_rate_data["DETR"] = {
            sz: _by_size_key(detr_data["summary"], sz, "failure") for sz in SIZE_ORDER
        }
    if sam_auto_data:
        failure_rate_data["SAM-Auto"] = {
            sz: _by_size_key(sam_auto_data["summary"], sz, "failure") for sz in SIZE_ORDER
        }
    if cascade_data:
        failure_rate_data["Cascade (MaskRCNN→SAM)"] = {
            sz: _by_size_key(cascade_data["summary"], sz, "failure") for sz in SIZE_ORDER
        }
    if sam_data:
        for strategy in ["box", "box_pos_neg"]:
            if strategy in sam_data["summary"]:
                failure_rate_data[f"SAM ({strategy})"] = {
                    sz: _by_size_key(sam_data["summary"][strategy], sz, "failure")
                    for sz in SIZE_ORDER
                }
                break
    if failure_rate_data:
        plot_failure_rate_by_size(
            failure_rate_data,
            os.path.join(args.output, "failure_rate_by_size.png"),
        )

    # ── 4. Worst categories — one figure per model ──
    for data, label, title in [
        (mrcnn_data,    "maskrcnn",   "Worst Categories — Mask R-CNN"),
        (m2f_data,      "mask2former","Worst Categories — Mask2Former"),
        (detr_data,     "detr",       "Worst Categories — DETR"),
        (cascade_data,  "cascade",    "Worst Categories — Cascade (MaskRCNN→SAM)"),
    ]:
        if data:
            cat_ious = {
                cat: info["miou"]
                for cat, info in data["summary"]["by_category"].items()
            }
            plot_worst_categories(
                cat_ious,
                os.path.join(args.output, f"worst_cats_{label}.png"),
                title=title,
            )

    # ── 5. SAM prompt comparison ──
    if sam_data:
        strategy_mious = {
            s: sam_data["summary"][s]["overall"]["miou"]
            for s in sam_data["summary"]
        }
        plot_sam_prompt_comparison(
            strategy_mious,
            os.path.join(args.output, "sam_prompt_comparison.png"),
        )

    # ── 6. Thin vs non-thin ──
    # SAM per_instance is {strategy: [records]}; others are flat [records]
    thin_data = {}
    if sam_data:
        for strategy in ["box", "box_pos_neg"]:
            instances = sam_data["per_instance"].get(strategy, [])
            if instances:
                thin_data[f"SAM ({strategy})"] = _flat_thin_entry(instances)
                break
    if mrcnn_data:
        thin_data["MaskRCNN"] = _flat_thin_entry(mrcnn_data["per_instance"])
    if m2f_data:
        thin_data["Mask2Former"] = _flat_thin_entry(m2f_data["per_instance"])
    if detr_data:
        thin_data["DETR"] = _flat_thin_entry(detr_data["per_instance"])
    if sam_auto_data:
        thin_data["SAM-Auto"] = _flat_thin_entry(sam_auto_data["per_instance"])
    if cascade_data:
        thin_data["Cascade (MaskRCNN→SAM)"] = _flat_thin_entry(cascade_data["per_instance"])
    if thin_data:
        plot_thin_vs_nonthin(thin_data, os.path.join(args.output, "thin_vs_nonthin.png"))

    # ── 7. Failure mode distribution ──
    mode_data = {}
    if sam_data:
        for strategy in ["box", "box_pos_neg"]:
            instances = sam_data["per_instance"].get(strategy, [])
            if instances:
                mode_data[f"SAM ({strategy})"] = _failure_mode_counts(instances)
                break
    if mrcnn_data:
        mode_data["MaskRCNN"] = _failure_mode_counts(mrcnn_data["per_instance"])
    if m2f_data:
        mode_data["Mask2Former"] = _failure_mode_counts(m2f_data["per_instance"])
    if detr_data:
        mode_data["DETR"] = _failure_mode_counts(detr_data["per_instance"])
    if sam_auto_data:
        mode_data["SAM-Auto"] = _failure_mode_counts(sam_auto_data["per_instance"])
    if cascade_data:
        mode_data["Cascade (MaskRCNN→SAM)"] = _failure_mode_counts(cascade_data["per_instance"])
    if mode_data:
        plot_failure_modes(mode_data, os.path.join(args.output, "failure_modes.png"))

    # ── 8. Improvement comparison: MaskRCNN → Cascade → SAM-Auto → SAM(GT) ──
    improv_models = {}
    if mrcnn_data:
        improv_models["MaskRCNN"] = {
            **{sz: _by_size_key(mrcnn_data["summary"], sz, "miou") for sz in SIZE_ORDER},
            "overall": mrcnn_data["summary"]["overall"]["miou"],
        }
    if cascade_data:
        improv_models["Cascade (MaskRCNN→SAM)"] = {
            **{sz: _by_size_key(cascade_data["summary"], sz, "miou") for sz in SIZE_ORDER},
            "overall": cascade_data["summary"]["overall"]["miou"],
        }
    if sam_auto_data:
        improv_models["SAM-Auto"] = {
            **{sz: _by_size_key(sam_auto_data["summary"], sz, "miou") for sz in SIZE_ORDER},
            "overall": sam_auto_data["summary"]["overall"]["miou"],
        }
    if sam_data and "box" in sam_data["summary"]:
        improv_models["SAM (box)"] = {
            **{sz: _by_size_key(sam_data["summary"]["box"], sz, "miou") for sz in SIZE_ORDER},
            "overall": sam_data["summary"]["box"]["overall"]["miou"],
        }
    if len(improv_models) >= 2:
        plot_improvement_comparison(
            improv_models,
            os.path.join(args.output, "improvement_comparison.png"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam",         default=None)
    parser.add_argument("--maskrcnn",    default=None)
    parser.add_argument("--mask2former", default=None)
    parser.add_argument("--detr",        default=None)
    parser.add_argument("--sam-auto",    default=None, dest="sam_auto")
    parser.add_argument("--cascade",     default=None)
    parser.add_argument("--output",      default="figures/")
    run(parser.parse_args())
