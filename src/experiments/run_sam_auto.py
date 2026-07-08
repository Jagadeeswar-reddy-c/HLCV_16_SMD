import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
SAM evaluation using AUTOMATIC mask generation (no GT prompts).

Contrast with run_sam.py which uses GT bounding boxes / masks as prompts.
Here we run SamAutomaticMaskGenerator which produces all masks from a uniform
point grid — no ground-truth information is used.  This represents the true
zero-shot deployment scenario and quantifies the "prompting gap".

CLI:
    python src/experiments/run_sam_auto.py \
        --subset     data/subsets/subset_200.json \
        --checkpoint checkpoints/sam_vit_b_01ec64.pth \
        --output     results/sam_auto_results.json \
        [--device cuda]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.dataset.coco_loader import load_subset, decode_mask
from src.evaluation.matcher import match_all_gt_instances
from src.evaluation.metrics import compute_iou, summarise_results
from src.utils import load_image_rgb, get_device, ensure_dir


def run(args):
    device = args.device or get_device()
    print(f"Device: {device}")

    records = load_subset(args.subset)
    print(f"Loaded {len(records)} instances from {args.subset}")

    # Build SAM automatic mask generator
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,          # dense grid → more recall on small objects
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=64,     # skip dust-size masks
    )

    by_image = defaultdict(list)
    for r in records:
        by_image[r.image_id].append(r)

    all_results = []

    for img_id, img_records in tqdm(by_image.items(), desc="Images"):
        img_path = img_records[0].image_path
        try:
            image = load_image_rgb(img_path)
        except FileNotFoundError:
            print(f"  [skip] missing image: {img_path}")
            continue

        # Generate all masks automatically — no GT used
        auto_masks_raw = generator.generate(image)

        pred_masks  = [m["segmentation"].astype(bool) for m in auto_masks_raw]
        pred_scores = [float(m["predicted_iou"])      for m in auto_masks_raw]
        # SAM auto has no category prediction; pass -1 so matcher uses IoU-only fallback
        pred_cat_ids = [-1] * len(pred_masks)

        gt_instances = [
            {
                "ann_id":        r.ann_id,
                "image_id":      r.image_id,
                "category_id":   r.category_id,
                "category_name": r.category_name,
                "size":          r.size,
                "area":          r.area,
                "is_thin":       r.is_thin,
                "gt_mask":       decode_mask(r),
            }
            for r in img_records
        ]

        matched = match_all_gt_instances(
            gt_instances,
            pred_masks,
            pred_scores,
            pred_cat_ids,
            score_threshold=0.0,   # scores already filtered by generator thresholds
        )

        for m in matched:
            all_results.append({
                "ann_id":        m["ann_id"],
                "image_id":      m["image_id"],
                "category_name": m["category_name"],
                "size":          m["size"],
                "area":          m["area"],
                "is_thin":       m["is_thin"],
                "iou":           m["iou"],
            })

    summary = summarise_results(all_results)
    output = {
        "model":        "SAM-Auto",
        "summary":      summary,
        "per_instance": all_results,
    }

    ensure_dir(str(Path(args.output).parent))
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {args.output}")
    print(f"  mIoU={summary['overall']['miou']:.3f}  "
          f"success={summary['overall']['success_rate']:.3f}  "
          f"failure={summary['overall']['failure_rate']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output",      default="results/sam_auto_results.json")
    parser.add_argument("--model-type",  default="vit_b")
    parser.add_argument("--device",      default=None)
    run(parser.parse_args())
