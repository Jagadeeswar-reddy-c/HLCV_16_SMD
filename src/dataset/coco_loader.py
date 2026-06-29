import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""
Load COCO val2017 annotations and build InstanceRecord lists.

CLI usage:
    python src/dataset/coco_loader.py \
        --images   data/coco/val2017 \
        --ann      data/coco/annotations/instances_val2017.json \
        --output   data/subsets/subset_200.json \
        --n-images 200
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_util

from src.dataset.instance import InstanceRecord, size_group, THIN_CATEGORIES

# Categories to ensure representation in the subset
PRIORITY_CATEGORIES = {
    "person", "car", "dog", "chair", "bicycle",
    "bottle", "cat", "bird", "horse", "sheep",
    "umbrella", "tie", "skis", "fork", "knife",
}


def load_coco(ann_file: str) -> COCO:
    return COCO(ann_file)


def build_instance_records(
    coco: COCO,
    images_dir: str,
    image_ids: Optional[List[int]] = None,
) -> List[InstanceRecord]:
    """
    Return one InstanceRecord per non-crowd annotation across the given image IDs.
    If image_ids is None, uses all images in the annotation file.
    """
    if image_ids is None:
        image_ids = coco.getImgIds()

    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    records: List[InstanceRecord] = []

    for img_meta in coco.loadImgs(image_ids):
        img_path = os.path.join(images_dir, img_meta["file_name"])
        ann_ids = coco.getAnnIds(imgIds=img_meta["id"], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
            area = float(ann["area"])
            rec = InstanceRecord(
                image_id=img_meta["id"],
                image_path=img_path,
                image_width=img_meta["width"],
                image_height=img_meta["height"],
                ann_id=ann["id"],
                category_id=ann["category_id"],
                category_name=cat_name,
                bbox=ann["bbox"],
                area=area,
                size=size_group(area),
                is_thin=(cat_name in THIN_CATEGORIES),
                segmentation=ann["segmentation"],
                iscrowd=ann.get("iscrowd", 0),
            )
            records.append(rec)

    return records


def decode_mask(record: InstanceRecord) -> np.ndarray:
    """Decode a COCO RLE or polygon segmentation into a binary uint8 mask."""
    seg = record.segmentation
    h, w = record.image_height, record.image_width
    if isinstance(seg, dict):  # RLE
        rle = seg
    else:  # polygon list
        rle = coco_mask_util.frPyObjects(seg, h, w)
        rle = coco_mask_util.merge(rle)
    return coco_mask_util.decode(rle).astype(np.uint8)


def _select_balanced_images(
    coco: COCO,
    n_images: int,
    seed: int = 42,
) -> List[int]:
    """
    Select image IDs that collectively cover:
    - small, medium, large objects
    - priority categories (thin objects, common COCO classes)
    - crowded scenes (many instances per image)
    """
    rng = random.Random(seed)
    all_ids = coco.getImgIds()
    cat_name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    selected = set()

    # 1. Images with small objects
    for cat_name in PRIORITY_CATEGORIES:
        cat_id = cat_name_to_id.get(cat_name)
        if cat_id is None:
            continue
        ann_ids = coco.getAnnIds(catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        small_imgs = list({a["image_id"] for a in anns if a["area"] < 32 ** 2})
        selected.update(rng.sample(small_imgs, min(5, len(small_imgs))))

    # 2. Images with thin-object categories
    for cat_name in THIN_CATEGORIES:
        cat_id = cat_name_to_id.get(cat_name)
        if cat_id is None:
            continue
        imgs = coco.getImgIds(catIds=[cat_id])
        selected.update(rng.sample(imgs, min(5, len(imgs))))

    # 3. Crowded scenes: images with many instances
    img_instance_counts = {}
    for img_id in all_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        img_instance_counts[img_id] = len(ann_ids)
    crowded = sorted(img_instance_counts, key=img_instance_counts.get, reverse=True)
    selected.update(crowded[:20])

    # 4. Fill remainder randomly
    remaining = [i for i in all_ids if i not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, n_images - len(selected))])

    result = list(selected)[:n_images]
    rng.shuffle(result)
    return result


def create_subset(
    ann_file: str,
    images_dir: str,
    output_path: str,
    n_images: int = 200,
    seed: int = 42,
) -> List[InstanceRecord]:
    """
    Build a balanced subset of n_images and save it as a JSON file.
    Returns the list of InstanceRecord objects.
    """
    coco = load_coco(ann_file)
    image_ids = _select_balanced_images(coco, n_images, seed=seed)
    records = build_instance_records(coco, images_dir, image_ids)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = [
        {
            "image_id": r.image_id,
            "image_path": r.image_path,
            "image_width": r.image_width,
            "image_height": r.image_height,
            "ann_id": r.ann_id,
            "category_id": r.category_id,
            "category_name": r.category_name,
            "bbox": r.bbox,
            "area": r.area,
            "size": r.size,
            "is_thin": r.is_thin,
            "segmentation": r.segmentation,
            "iscrowd": r.iscrowd,
        }
        for r in records
    ]

    summary = _subset_summary(records)
    payload = {"meta": {"n_images": len(image_ids), "n_instances": len(records), **summary}, "instances": serialisable}
    with open(output_path, "w") as f:
        json.dump(payload, f)

    print(f"Saved {len(image_ids)} images / {len(records)} instances → {output_path}")
    print(f"  small={summary['small']}  medium={summary['medium']}  large={summary['large']}")
    print(f"  thin instances: {summary['thin']}")
    return records


def load_subset(subset_json: str) -> List[InstanceRecord]:
    """Deserialise a previously saved subset JSON back into InstanceRecord objects."""
    with open(subset_json) as f:
        payload = json.load(f)
    records = []
    for d in payload["instances"]:
        records.append(
            InstanceRecord(
                image_id=d["image_id"],
                image_path=d["image_path"],
                image_width=d["image_width"],
                image_height=d["image_height"],
                ann_id=d["ann_id"],
                category_id=d["category_id"],
                category_name=d["category_name"],
                bbox=d["bbox"],
                area=d["area"],
                size=d["size"],
                is_thin=d["is_thin"],
                segmentation=d["segmentation"],
                iscrowd=d.get("iscrowd", 0),
            )
        )
    return records


def _subset_summary(records: List[InstanceRecord]) -> dict:
    from collections import Counter
    size_counts = Counter(r.size for r in records)
    cat_counts = Counter(r.category_name for r in records)
    return {
        "small": size_counts["small"],
        "medium": size_counts["medium"],
        "large": size_counts["large"],
        "thin": sum(1 for r in records if r.is_thin),
        "top_categories": dict(cat_counts.most_common(10)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/coco/val2017")
    parser.add_argument("--ann", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--output", default="data/subsets/subset_200.json")
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_subset(
        ann_file=args.ann,
        images_dir=args.images,
        output_path=args.output,
        n_images=args.n_images,
        seed=args.seed,
    )
