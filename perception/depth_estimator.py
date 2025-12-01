from __future__ import annotations

from typing import Union, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
import torch

class DepthEstimator:
    def __init__(self,model_type: str = "MiDaS_small",device: str = "cpu",) -> None:
        self.device = device

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ("DPT_Large", "DPT_Hybrid", "MiDaS_small"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:

        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Cannot read image: {image}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            raise ValueError("DepthEstimator expects HxWx3 image array")

        raise TypeError("Unsupported image type for DepthEstimator")

    def predict(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        img = self._load_image(image)
        h, w = img.shape[:2]

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        depth_map = prediction.cpu().numpy().astype(np.float32)
        return depth_map


def attach_depth_to_detections(
    detections: List[Dict[str, Any]],
    depth_map: np.ndarray,
) -> None:
    H, W = depth_map.shape[:2]

    for det in detections:
        bbox = det.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(x)) for x in bbox]

        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        region = depth_map[y1:y2, x1:x2]

        mask = det.get("mask", None)
        if mask is not None:
            if mask.shape != (H, W):
                pass
            else:
                mask_crop = mask[y1:y2, x1:x2]
                region = region[mask_crop]

        if region.size == 0:
            continue

        det["depth_min"] = float(region.min())
        det["depth_max"] = float(region.max())
        det["depth_mean"] = float(region.mean())
