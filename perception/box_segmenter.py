from __future__ import annotations

from typing import Any, Dict, List, Union
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

class BoxSegmenter:
    def __init__(
        self,
        sam_model_path: str = "FastSAM-s.pt",
        device: str = "cpu",
        imgsz: int = 512,
        conf: float = 0.4,
        iou: float = 0.9,
    ) -> None:
        self.model = FastSAM(sam_model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
            return img
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            raise ValueError("BoxSegmenter expects HxWx3 image array")
        else:
            raise TypeError("Unsupported image type for BoxSegmenter")

    def _pick_mask_index(self, masks: np.ndarray) -> int:
        areas = masks.sum(axis=(1, 2))
        return int(np.argmax(areas))

    def segment(self,image: Union[str, Path, np.ndarray],detections: List[Dict[str, Any]],) -> List[Dict[str, Any]]:
        img = self._load_image(image)
        H, W = img.shape[:2]

        out: List[Dict[str, Any]] = []

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(round(x)) for x in bbox]

            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))

            det_out = dict(det)

            if x2 <= x1 or y2 <= y1:
                det_out["mask"] = None
                det_out["mask_source"] = "none"
                out.append(det_out)
                continue

            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0:
                det_out["mask"] = None
                det_out["mask_source"] = "none"
                out.append(det_out)
                continue

            results = self.model(
                source=crop,
                device=self.device,
                retina_masks=True,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )

            if not results:
                det_out["mask"] = None
                det_out["mask_source"] = "none"
                out.append(det_out)
                continue

            res = results[0]
            if getattr(res, "masks", None) is None:
                det_out["mask"] = None
                det_out["mask_source"] = "none"
                out.append(det_out)
                continue

            masks = res.masks.data.cpu().numpy()
            if masks.ndim != 3 or masks.shape[0] == 0:
                det_out["mask"] = None
                det_out["mask_source"] = "none"
                out.append(det_out)
                continue

            best_idx = self._pick_mask_index(masks)
            crop_mask = masks[best_idx] > 0.5 

            m = (crop_mask.astype(np.uint8) * 255)
            kernel = np.ones((3, 3), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            crop_mask = m > 0

            full_mask = np.zeros((H, W), dtype=bool)
            h_crop, w_crop = crop_mask.shape
            full_mask[y1 : y1 + h_crop, x1 : x1 + w_crop] = crop_mask

            det_out["mask"] = full_mask
            det_out["mask_source"] = "crop_sam"
            out.append(det_out)

        return out
