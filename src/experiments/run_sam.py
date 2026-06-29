import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
SAM evaluation: all 5 prompt strategies on every GT instance in the subset.

CLI:
    python src/experiments/run_sam.py \
        --subset   data/subsets/subset_200.json \
        --checkpoint checkpoints/sam_vit_b_01ec64.pth \
        --output   results/sam_results.json \
        [--model-type vit_b] [--device cuda]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.dataset.coco_loader import load_subset, decode_mask
from src.evaluation.metrics import compute_iou, summarise_results
from src.models.sam_predictor import SAMPredictor, PROMPT_STRATEGIES
from src.utils import load_image_rgb, get_device, ensure_dir


def run(args):
    device = args.device or get_device()
    print(f"Device: {device}")

    records = load_subset(args.subset)
    print(f"Loaded {len(records)} instances from {args.subset}")

    predictor = SAMPredictor(args.checkpoint, model_type=args.model_type, device=device)

    # Group records by image so we call set_image once per image
    by_image = defaultdict(list)
    for r in records:
        by_image[r.image_id].append(r)

    # strategy → list of per-instance result dicts
    strategy_results = {s: [] for s in PROMPT_STRATEGIES}

    for img_id, img_records in tqdm(by_image.items(), desc="Images"):
        img_path = img_records[0].image_path
        try:
            image = load_image_rgb(img_path)
        except FileNotFoundError:
            print(f"  [skip] missing image: {img_path}")
            continue

        predictor.set_image(image)

        for rec in img_records:
            gt_mask = decode_mask(rec)
            for strategy in PROMPT_STRATEGIES:
                try:
                    pred_mask = predictor.predict_with_strategy(
                        gt_mask=gt_mask,
                        bbox_xywh=rec.bbox,
                        strategy=strategy,
                        seed=rec.ann_id,
                    )
                    iou = compute_iou(pred_mask, gt_mask)
                except Exception as e:
                    print(f"  [error] ann={rec.ann_id} strategy={strategy}: {e}")
                    iou = 0.0

                strategy_results[strategy].append({
                    "ann_id": rec.ann_id,
                    "image_id": rec.image_id,
                    "category_name": rec.category_name,
                    "size": rec.size,
                    "area": rec.area,
                    "is_thin": rec.is_thin,
                    "iou": iou,
                })

    # Aggregate
    summary = {
        strategy: summarise_results(res)
        for strategy, res in strategy_results.items()
    }

    output = {
        "model": "SAM",
        "model_type": args.model_type,
        "summary": summary,
        "per_instance": strategy_results,
    }

    ensure_dir(str(Path(args.output).parent))
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {args.output}")
    for s in PROMPT_STRATEGIES:
        miou = summary[s]["overall"]["miou"]
        sr = summary[s]["overall"]["success_rate"]
        print(f"  {s:20s}  mIoU={miou:.3f}  success={sr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/sam_results.json")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--device", default=None)
    run(parser.parse_args())
