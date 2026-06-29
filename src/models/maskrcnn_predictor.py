"""
Mask R-CNN wrapper using torchvision pretrained on COCO.

COCO category IDs from torchvision are 1-indexed and include a background
class at index 0. The mapping here converts torchvision label ids to COCO ids.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


# torchvision Mask R-CNN outputs labels 1..90 corresponding to COCO category ids 1..90
# (background = 0, which we skip). torchvision and COCO share the same id space.
TORCHVISION_TO_COCO_ID = {i: i for i in range(1, 91)}


class MaskRCNNPredictor:
    def __init__(self, device: str = "cpu", score_threshold: float = 0.5):
        import torchvision
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        )
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.score_threshold = score_threshold

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Run Mask R-CNN on a single RGB image (H×W×3, uint8).

        Returns:
            masks:        list of binary H×W bool arrays
            scores:       confidence score per mask
            category_ids: COCO category id per mask
        """
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(image_rgb).to(self.device)
        output = self.model([tensor])[0]

        keep = output["scores"] >= self.score_threshold
        raw_masks = output["masks"][keep].squeeze(1).cpu().numpy()   # (N, H, W)
        scores = output["scores"][keep].cpu().numpy().tolist()
        labels = output["labels"][keep].cpu().numpy().tolist()

        masks = [(m >= 0.5).astype(bool) for m in raw_masks]
        category_ids = [TORCHVISION_TO_COCO_ID.get(int(l), int(l)) for l in labels]

        return masks, scores, category_ids
