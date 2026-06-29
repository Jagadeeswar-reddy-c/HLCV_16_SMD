import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
Mask2Former evaluation over the COCO subset.

CLI:
    python src/experiments/run_mask2former.py \
        --subset data/subsets/subset_200.json \
        --output results/mask2former_results.json \
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
from src.models.mask2former_predictor import Mask2FormerPredictor
from src.utils import load_image_rgb, get_device, ensure_dir


def run(args):
    device = args.device or get_device()
    print(f"Device: {device}")

    records = load_subset(args.subset)
    print(f"Loaded {len(records)} instances from {args.subset}")

    model = Mask2FormerPredictor(device=device, score_threshold=args.score_threshold)

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
                "ann_id": r.ann_id,
                "image_id": r.image_id,
                "category_id": r.category_id,
                "category_name": r.category_name,
                "size": r.size,
                "area": r.area,
                "is_thin": r.is_thin,
                "gt_mask": decode_mask(r),
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
                "ann_id": m["ann_id"],
                "image_id": m["image_id"],
                "category_name": m["category_name"],
                "size": m["size"],
                "area": m["area"],
                "is_thin": m["is_thin"],
                "iou": m["iou"],
            })

    summary = summarise_results(all_results)
    output = {
        "model": "Mask2Former",
        "summary": summary,
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
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output", default="results/mask2former_results.json")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    run(parser.parse_args())
