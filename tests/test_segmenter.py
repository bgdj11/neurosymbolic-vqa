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


def draw_masks(image_path: Path, detections: list, output_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read image {image_path}")
        return

    H, W = img.shape[:2]

    out_img = img.copy()

    for det in detections:
        mask = det.get("mask", None)
        if mask is None:
            continue

        if mask.shape != (H, W):
            print(f"Mask shape mismatch: {mask.shape} vs {(H, W)}")
            continue

        out_img[mask] = (0, 0, 0)

    for det in detections:
        x1, y1, x2, y2 = [int(round(x)) for x in det["bbox"]]
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out_img)
    print(f"Saved mask visualization to: {output_path}")


def main():
    clevr_jsonl = project_root / "datasets" / "converted" / "clevr" / "clevr_val.jsonl"
    if not clevr_jsonl.exists():
        print(f"CLEVR JSONL not found at: {clevr_jsonl}")
        return

    sample = pick_random_clevr_sample(clevr_jsonl, max_samples=200, seed=42)
    img_path = project_root / sample["image_path"]

    if not img_path.exists():
        print(f"Image not found: {img_path}")
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

    detections = detector.detect(str(img_path), conf_threshold=0.05)
    seg_dets = segmenter.segment(str(img_path), detections)

    out_dir = project_root / "tests" / "segmenter_output"
    out_img = out_dir / f"{sample['image_id']}_masks.png"

    draw_masks(img_path, seg_dets, out_img)


if __name__ == "__main__":
    main()
