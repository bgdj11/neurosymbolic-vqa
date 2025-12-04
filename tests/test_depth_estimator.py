import sys
import json
import random
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from perception.detector import HybridDetector
from perception.box_segmenter import BoxSegmenter
from perception.depth_estimator import DepthEstimator, attach_depth_to_detections
from perception.postprocess import resolve_mask_overlaps_by_depth


def pick_random_clevr_sample(jsonl_path: Path, max_samples: int = 200, seed: int | None = 42):
    samples_by_id = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            img_id = data["image_id"]

            if img_id in samples_by_id:
                continue

            samples_by_id[img_id] = {
                "image_id": img_id,
                "image_path": data["image_path"],
            }

            if len(samples_by_id) >= max_samples:
                break

    if not samples_by_id:
        raise RuntimeError("No CLEVR samples loaded")

    img_ids = list(samples_by_id.keys())

    if seed is not None:
        random.seed(seed)

    chosen_id = random.choice(img_ids)
    return samples_by_id[chosen_id]


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    d_min = float(d.min())
    d_max = float(d.max())
    if d_max > d_min:
        d_norm = (d - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(d, dtype=np.float32)
    return d_norm.astype(np.float32)


def build_colormap_lut() -> np.ndarray:
    base = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    lut = cv2.applyColorMap(base, cv2.COLORMAP_JET)
    return lut 

def draw_full_depth_heatmap(depth_norm: np.ndarray, detections: list, lut: np.ndarray,) -> np.ndarray:
    H, W = depth_norm.shape[:2]

    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    for det in detections:
        bbox = det.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(x)) for x in bbox]
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H, y2))
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return heatmap


def draw_object_depth_map(depth_norm: np.ndarray, detections: list, lut: np.ndarray,) -> np.ndarray:
    H, W = depth_norm.shape[:2]
    obj_img = np.zeros((H, W, 3), dtype=np.uint8)

    for det in detections:
        mask = det.get("mask", None)
        bbox = det.get("bbox", None)

        if mask is None or bbox is None or len(bbox) != 4:
            continue

        if mask.shape != (H, W):
            continue

        depth_vals = depth_norm[mask]
        if depth_vals.size == 0:
            continue

        depth_mean = float(depth_vals.mean())
        col_idx = int(np.clip(depth_mean * 255.0, 0, 255))
        color = lut[col_idx, 0, :]

        obj_img[mask] = color

    return obj_img


def main():
    clevr_jsonl = project_root / "datasets" / "converted" / "clevr" / "clevr_val.jsonl"
    if not clevr_jsonl.exists():
        print(f"CLEVR JSONL not found at: {clevr_jsonl}")
        return

    sample = pick_random_clevr_sample(clevr_jsonl, max_samples=300, seed=42)
    img_path = project_root / sample["image_path"]

    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    print(f"Using image: {sample['image_id']} at {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Cannot read image: {img_path}")
        return

    detector = HybridDetector(
        yolo_model_path="yolov8n.pt",
        sam_model_path="FastSAM-s.pt",
        device="cpu",
    )

    segmenter = BoxSegmenter(
        sam_model_path="FastSAM-s.pt",
        device="cpu",
    )

    depth_estimator = DepthEstimator(
        model_type="MiDaS_small",
        device="cpu",
    )

    detections = detector.detect(str(img_path), conf_threshold=0.05)
    detections = segmenter.segment(str(img_path), detections)

    depth_map = depth_estimator.predict(str(img_path))
    depth_norm = normalize_depth(depth_map)

    lut = build_colormap_lut()

    full_depth_img = draw_full_depth_heatmap(depth_norm, detections, lut)

    obj_depth_before = draw_object_depth_map(depth_norm, detections, lut)

    attach_depth_to_detections(detections, depth_map)
    resolve_mask_overlaps_by_depth(detections, depth_map)

    obj_depth_after = draw_object_depth_map(depth_norm, detections, lut)

    out_dir = project_root / "tests" / "depth_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / f"{sample['image_id']}_depth_full.png"
    obj_before_path = out_dir / f"{sample['image_id']}_depth_objects_before_zbuf.png"
    obj_after_path = out_dir / f"{sample['image_id']}_depth_objects_after_zbuf.png"

    cv2.imwrite(str(full_path), full_depth_img)
    cv2.imwrite(str(obj_before_path), obj_depth_before)
    cv2.imwrite(str(obj_after_path), obj_depth_after)

    print(f"Saved full depth heatmap to:          {full_path}")
    print(f"Saved per-object depth (before zbuf) to: {obj_before_path}")
    print(f"Saved per-object depth (after zbuf) to:  {obj_after_path}")


if __name__ == "__main__":
    main()
