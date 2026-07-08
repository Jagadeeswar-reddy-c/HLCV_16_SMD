import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
Evaluate the Mask R-CNN → SAM cascade on the COCO subset.

The cascade uses Mask R-CNN bounding box predictions (not GT) as SAM prompts.
This is a concrete improvement over either standalone model:
  - vs Mask R-CNN alone: SAM produces sharper boundaries (full-resolution mask head)
  - vs SAM+GT-box: no ground-truth information used — fully automatic

Compare results/cascade_results.json to results/maskrcnn_results.json to see
the boundary-quality improvement, and to results/sam_results.json (box strategy)
to see how close this comes to the oracle (GT-box) upper bound.

CLI:
    python src/experiments/run_cascade.py \
        --subset     data/subsets/subset_200.json \
        --checkpoint checkpoints/sam_vit_b_01ec64.pth \
        --output     results/cascade_results.json \
        [--score-threshold 0.5] [--device cuda]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.dataset.coco_loader import load_subset, decode_mask
from src.evaluation.matcher import match_all_gt_instances
from src.evaluation.metrics import summarise_results
from src.models.cascade_predictor import CascadePredictor
from src.utils import load_image_rgb, get_device, ensure_dir


def run(args):
    device = args.device or get_device()
    print(f"Device: {device}")

    records = load_subset(args.subset)
    print(f"Loaded {len(records)} instances from {args.subset}")

    model = CascadePredictor(
        sam_checkpoint=args.checkpoint,
        device=device,
        score_threshold=args.score_threshold,
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

        pred_masks, pred_scores, pred_cat_ids = model.predict(image)

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
            score_threshold=args.score_threshold,
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
        "model":        "Cascade (MaskRCNN→SAM)",
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
    parser.add_argument("--subset",          required=True)
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--output",          default="results/cascade_results.json")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device",          default=None)
    run(parser.parse_args())
