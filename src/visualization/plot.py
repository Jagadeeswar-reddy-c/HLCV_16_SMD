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

COLORS = {
    "SAM (box)":        "#4C72B0",
    "SAM (box_pos_neg)":"#55A868",
    "MaskRCNN":         "#C44E52",
    "Mask2Former":      "#DD8452",
}
SIZE_ORDER = ["small", "medium", "large"]


# ------------------------------------------------------------------
# Bar charts
# ------------------------------------------------------------------

def plot_overall_miou(model_results: Dict[str, float], output_path: str) -> None:
    """Bar chart: overall mIoU per model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    models = list(model_results.keys())
    mious = [model_results[m] for m in models]
    bars = ax.bar(models, mious, color=[COLORS.get(m, "#888888") for m in models], width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall mIoU by Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_miou_by_size(
    model_size_results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Grouped bar chart: mIoU broken down by object size for each model."""
    models = list(model_size_results.keys())
    x = np.arange(len(SIZE_ORDER))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
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
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_worst_categories(
    category_ious: Dict[str, float],
    output_path: str,
    n: int = 10,
) -> None:
    """Horizontal bar chart of the n worst categories."""
    ranked = sorted(category_ious.items(), key=lambda kv: kv[1])[:n]
    cats, vals = zip(*ranked) if ranked else ([], [])

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(cats))
    ax.barh(y, vals, color="#C44E52")
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("mIoU")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"Top {n} Worst Categories")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_failure_rate_by_size(
    model_failure_rates: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Grouped bar chart: failure rate (IoU < 0.3) per size group."""
    models = list(model_failure_rates.keys())
    x = np.arange(len(SIZE_ORDER))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
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
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_sam_prompt_comparison(
    strategy_mious: Dict[str, float],
    output_path: str,
) -> None:
    """Bar chart comparing SAM prompt strategies."""
    strategies = list(strategy_mious.keys())
    vals = [strategy_mious[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(strategies, vals, color="#4C72B0", width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("SAM Prompt Strategy Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Qualitative mask overlay
# ------------------------------------------------------------------

def overlay_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5) -> np.ndarray:
    out = image.copy().astype(float)
    for c, col in enumerate(color):
        out[..., c] = np.where(mask, out[..., c] * (1 - alpha) + col * alpha, out[..., c])
    return np.clip(out, 0, 255).astype(np.uint8)


def save_qualitative_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    predictions: Dict[str, Optional[np.ndarray]],
    ious: Dict[str, float],
    output_path: str,
    title: str = "",
) -> None:
    """
    Show image + GT + one panel per model/strategy.
    predictions: dict label → binary mask (or None for missing)
    """
    n = 1 + 1 + len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    gt_vis = overlay_mask(image, gt_mask, color=(0, 255, 0))
    axes[1].imshow(gt_vis)
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    for ax, (label, mask) in zip(axes[2:], predictions.items()):
        if mask is not None:
            vis = overlay_mask(image, mask, color=(255, 80, 80))
            ax.imshow(vis)
        else:
            ax.imshow(image)
        iou = ious.get(label, float("nan"))
        ax.set_title(f"{label}\nIoU={iou:.3f}")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Main: load results JSONs and generate all figures
# ------------------------------------------------------------------

def load_json(path: str) -> Optional[Dict]:
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def run(args):
    ensure_dir(args.output)

    sam_data = load_json(args.sam)
    mrcnn_data = load_json(args.maskrcnn)
    m2f_data = load_json(args.mask2former)

    # --- Overall mIoU ---
    overall = {}
    if sam_data:
        # Use best SAM strategy (box_pos_neg) as the representative
        for strategy in ["box_pos_neg", "box", "center_point"]:
            if strategy in sam_data["summary"]:
                overall[f"SAM ({strategy})"] = sam_data["summary"][strategy]["overall"]["miou"]
                break
    if mrcnn_data:
        overall["MaskRCNN"] = mrcnn_data["summary"]["overall"]["miou"]
    if m2f_data:
        overall["Mask2Former"] = m2f_data["summary"]["overall"]["miou"]

    if overall:
        plot_overall_miou(overall, os.path.join(args.output, "overall_miou.png"))

    # --- mIoU by size ---
    size_data = {}
    if sam_data:
        for strategy in ["box_pos_neg", "box"]:
            if strategy in sam_data["summary"]:
                size_data[f"SAM ({strategy})"] = {
                    sz: sam_data["summary"][strategy]["by_size"].get(sz, {}).get("miou", 0.0)
                    for sz in SIZE_ORDER
                }
                break
    if mrcnn_data:
        size_data["MaskRCNN"] = {
            sz: mrcnn_data["summary"]["by_size"].get(sz, {}).get("miou", 0.0)
            for sz in SIZE_ORDER
        }
    if m2f_data:
        size_data["Mask2Former"] = {
            sz: m2f_data["summary"]["by_size"].get(sz, {}).get("miou", 0.0)
            for sz in SIZE_ORDER
        }
    if size_data:
        plot_miou_by_size(size_data, os.path.join(args.output, "miou_by_size.png"))

    # --- Failure rate by size (use MaskRCNN as example if available) ---
    failure_rate_data = {}
    if mrcnn_data:
        failure_rate_data["MaskRCNN"] = {
            sz: mrcnn_data["summary"]["by_size"].get(sz, {}).get("failure", 0.0)
            for sz in SIZE_ORDER
        }
    if m2f_data:
        failure_rate_data["Mask2Former"] = {
            sz: m2f_data["summary"]["by_size"].get(sz, {}).get("failure", 0.0)
            for sz in SIZE_ORDER
        }
    if sam_data:
        for strategy in ["box_pos_neg", "box"]:
            if strategy in sam_data["summary"]:
                failure_rate_data[f"SAM ({strategy})"] = {
                    sz: sam_data["summary"][strategy]["by_size"].get(sz, {}).get("failure", 0.0)
                    for sz in SIZE_ORDER
                }
                break
    if failure_rate_data:
        plot_failure_rate_by_size(
            failure_rate_data,
            os.path.join(args.output, "failure_rate_by_size.png"),
        )

    # --- Worst categories (pick first available model) ---
    for data, label in [(mrcnn_data, "maskrcnn"), (m2f_data, "mask2former")]:
        if data:
            cat_ious = {
                cat: info["miou"]
                for cat, info in data["summary"]["by_category"].items()
            }
            plot_worst_categories(cat_ious, os.path.join(args.output, f"worst_cats_{label}.png"))
            break

    # --- SAM prompt comparison ---
    if sam_data:
        strategy_mious = {
            s: sam_data["summary"][s]["overall"]["miou"]
            for s in sam_data["summary"]
        }
        plot_sam_prompt_comparison(
            strategy_mious,
            os.path.join(args.output, "sam_prompt_comparison.png"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam", default=None)
    parser.add_argument("--maskrcnn", default=None)
    parser.add_argument("--mask2former", default=None)
    parser.add_argument("--output", default="figures/")
    run(parser.parse_args())
