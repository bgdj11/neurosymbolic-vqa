from typing import Any, Dict, List, Union
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO, FastSAM
from ultralytics.utils import LOGGER 
import logging

LOGGER.setLevel(logging.ERROR)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_YOLO_PATH = str(MODELS_DIR / "yolov8n.pt")
DEFAULT_SAM_PATH = str(MODELS_DIR / "FastSAM-s.pt")   


class Detector:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        if model_path is None:
            model_path = DEFAULT_YOLO_PATH
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)

    def detect(self, image: Union[str, Path, np.ndarray], conf_threshold: float = 0.1) -> List[Dict[str, Any]]:
        
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for result in results:
            for box in result.boxes:
                box_xyxy = box.xyxy[0].cpu().numpy().tolist()
                box_conf = box.conf[0].cpu().numpy().item()
                box_cls = int(box.cls[0].cpu().numpy().item())

                detection = {
                    "bbox": box_xyxy,
                    "confidence": box_conf,
                    "class_id": box_cls,
                    "class_name": self.model.names[box_cls]
                }
                detections.append(detection)

        return detections

class HybridDetector(Detector):

    def __init__(
        self,
        yolo_model_path: str = None,
        sam_model_path: str = None,
        device: str = "cpu",
        min_rel_area: float = 0.001,
        max_rel_area: float = 0.4,
        iou_match_thresh: float = 0.3,
        iou_duplicate_thresh: float = 0.2,
        sam_imgsz: int = 640,
    ):
        if yolo_model_path is None:
            yolo_model_path = DEFAULT_YOLO_PATH
        if sam_model_path is None:
            sam_model_path = DEFAULT_SAM_PATH
            
        super().__init__(model_path=yolo_model_path, device=device)

        self.sam_model = FastSAM(sam_model_path)
        self.min_rel_area = min_rel_area
        self.max_rel_area = max_rel_area
        self.iou_match_thresh = iou_match_thresh
        self.iou_duplicate_thresh = iou_duplicate_thresh
        self.sam_imgsz = sam_imgsz

    @staticmethod
    def _bbox_area(box: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return w * h

    def _filter_by_area(self, box: np.ndarray, img_w: int, img_h: int) -> bool:
        area = self._bbox_area(box)
        img_area = float(img_w * img_h)
        if img_area <= 0:
            return False
        rel = area / img_area
        return self.min_rel_area <= rel <= self.max_rel_area

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        if inter <= 0:
            return 0.0

        area_a = HybridDetector._bbox_area(a)
        area_b = HybridDetector._bbox_area(b)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _get_image_size(self, image: Union[str, Path, np.ndarray]) -> tuple[int, int] | None:
        if isinstance(image, np.ndarray):
            if image.ndim >= 2:
                h, w = image.shape[:2]
                return (w, h)
            return None
        else:
            from PIL import Image 
            p = str(image)
            with Image.open(p) as im:
                w, h = im.size
            return (w, h)

    def _run_yolo(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float,
        img_w: int,
        img_h: int,
    ) -> List[Dict[str, Any]]:
        yolo_dets = super().detect(image=image, conf_threshold=conf_threshold)

        # filter additionaly by size
        filtered: List[Dict[str, Any]] = []
        for det in yolo_dets:
            box = np.array(det["bbox"], dtype=float)
            if not self._filter_by_area(box, img_w, img_h):
                continue
            d = det.copy()
            d["bbox"] = box
            d["source"] = "yolo"
            filtered.append(d)
        return filtered

    def _run_sam(
        self,
        image: Union[str, Path, np.ndarray],
        sam_conf_threshold: float,
        img_w: int,
        img_h: int,
    ) -> List[Dict[str, Any]]:
        results = self.sam_model(
            source=image,
            device=self.device,
            retina_masks=True,
            imgsz=self.sam_imgsz,
            conf=sam_conf_threshold,
            iou=0.9,
            verbose=False,
        )

        sam_dets: List[Dict[str, Any]] = []

        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            for box in res.boxes:
                box_xyxy = box.xyxy[0].cpu().numpy()
                if not self._filter_by_area(box_xyxy, img_w, img_h):
                    continue

                box_conf = float(box.conf[0].cpu().numpy())
                det = {
                    "bbox": box_xyxy,
                    "confidence": box_conf,
                    "class_id": 0,
                    "class_name": "sam_obj",
                    "source": "sam",
                }
                sam_dets.append(det)

        return sam_dets

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.1,
        sam_conf_threshold: float = 0.4,
        sam_conf_high: float = 0.7,
    ) -> List[Dict[str, Any]]:

        size = self._get_image_size(image)
        if size is None:
            return []

        img_w, img_h = size

        yolo_dets = self._run_yolo(
            image=image,
            conf_threshold=conf_threshold,
            img_w=img_w,
            img_h=img_h,
        )

        sam_dets = self._run_sam(
            image=image,
            sam_conf_threshold=sam_conf_threshold,
            img_w=img_w,
            img_h=img_h,
        )

        final: List[Dict[str, Any]] = []
        used_sam: set[int] = set()

        # 3) YOLO + SAM match
        for y in yolo_dets:
            y_box = np.array(y["bbox"], dtype=float)
            y_conf = float(y["confidence"])

            best_iou = 0.0
            best_j: int | None = None

            for j, s in enumerate(sam_dets):
                s_box = np.array(s["bbox"], dtype=float)
                iou = self._iou(y_box, s_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j is not None and best_iou >= self.iou_match_thresh:
                used_sam.add(best_j)
                s = sam_dets[best_j]
                score = 0.7 * y_conf + 0.3 * float(s["confidence"])

                bbox = s["bbox"]
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()

                final.append(
                    {
                        "bbox": bbox,
                        "confidence": score,
                        "class_id": y["class_id"],
                        "class_name": y["class_name"],
                        "source": "yolo+sam",
                    }
                )
            else:
                final.append(
                    {
                        "bbox": y_box.tolist(),
                        "confidence": y_conf,
                        "class_id": y["class_id"],
                        "class_name": y["class_name"],
                        "source": "yolo_only",
                    }
                )

        # 4) SAM-only
        for j, s in enumerate(sam_dets):
            if j in used_sam:
                continue
            s_conf = float(s["confidence"])
            if s_conf < sam_conf_high:
                continue

            s_box = np.array(s["bbox"], dtype=float)

            duplicate = False
            for f in final:
                f_box = np.array(f["bbox"], dtype=float)
                iou = self._iou(s_box, f_box)
                if iou >= self.iou_duplicate_thresh:
                    duplicate = True
                    break
            if duplicate:
                continue

            final.append(
                {
                    "bbox": s_box.tolist(),
                    "confidence": s_conf,
                    "class_id": -1,
                    "class_name": "sam_only",
                    "source": "sam_only",
                }
            )
        final.sort(key=lambda d: float(d["confidence"]), reverse=True)

        return final
