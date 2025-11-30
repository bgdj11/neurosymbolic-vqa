from typing import Any, Dict, List, Union
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
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
