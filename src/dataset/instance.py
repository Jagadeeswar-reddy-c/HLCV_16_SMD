from dataclasses import dataclass, field
from typing import List, Optional


THIN_CATEGORIES = {
    "bicycle", "chair", "umbrella", "tie", "skis",
    "sports ball", "fork", "knife", "baseball bat",
}

# COCO area thresholds
SMALL_AREA = 32 ** 2   # < 1024
MEDIUM_AREA = 96 ** 2  # < 9216


def size_group(area: float) -> str:
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


@dataclass
class InstanceRecord:
    image_id: int
    image_path: str
    image_width: int
    image_height: int

    ann_id: int
    category_id: int
    category_name: str

    # [x, y, w, h] in pixel coords
    bbox: List[float]
    area: float
    size: str  # "small" | "medium" | "large"
    is_thin: bool

    # RLE or polygon segmentation from COCO (raw, for mask decoding)
    segmentation: object
    iscrowd: int = 0

    # populated lazily by coco_loader.decode_mask()
    gt_mask: Optional[object] = field(default=None, repr=False)

    @property
    def bbox_xyxy(self) -> List[float]:
        x, y, w, h = self.bbox
        return [x, y, x + w, y + h]

    @property
    def center_point(self) -> List[float]:
        x, y, w, h = self.bbox
        return [x + w / 2, y + h / 2]
