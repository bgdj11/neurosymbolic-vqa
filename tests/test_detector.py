import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image
from perception.detector import Detector
import random

def load_clevr_samples(jsonl_path: Path, limit: int = 10, seed: int | None = None):
    samples_by_id = {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            img_id = data['image_id']

            if img_id in samples_by_id:
                continue

            num_objects = len(data['scene_graph']['objects'])
            samples_by_id[img_id] = {
                'image_id': img_id,
                'image_path': data['image_path'],
                'num_objects': num_objects,
                'objects': data['scene_graph']['objects'],
            }

    img_ids = list(samples_by_id.keys())

    if not img_ids:
        return []

    if seed is not None:
        random.seed(seed)

    chosen_ids = random.sample(img_ids, k=min(limit, len(img_ids)))
    return [samples_by_id[i] for i in chosen_ids]

def draw_detections(image_path: str, detections: list, gt_objects: list, output_path: str):
    img = cv2.imread(str(image_path))
    
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        
        x1, y1, x2, y2 = [int(x) for x in bbox]
        conf = det['confidence']
        class_name = det['class_name']

        color = colors[i % len(colors)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(str(output_path), img)


def test_detector(num_samples: int = 10, conf_threshold: float = 0.05, 
                   save_images: bool = True, verbose: bool = False):

    clevr_jsonl = project_root / "datasets" / "converted" / "clevr" / "clevr_val.jsonl"
    output_dir = project_root / "tests" / "detector_output"
    
    if save_images:
        output_dir.mkdir(exist_ok=True)
    
    if not clevr_jsonl.exists():
        print(f"[ERROR] CLEVR dataset not found at: {clevr_jsonl}")
        return []
    
    samples = load_clevr_samples(clevr_jsonl, limit=num_samples)

    detector = Detector(model_path="yolov8n.pt", device="cpu")

    results = []
    
    for sample in samples:
        img_path = project_root / sample['image_path']
        
        if not img_path.exists():
            continue
        
        gt_objects = sample['objects']
        gt_count = len(gt_objects)
        
        detections = detector.detect(str(img_path), conf_threshold=conf_threshold)
        det_count = len(detections)

        result = {
            'image_id': sample['image_id'],
            'gt_count': gt_count,
            'detected_count': det_count,
            'gt_objects': gt_objects,  
            'detections': detections,   
            'missed': gt_count - det_count if det_count < gt_count else 0,
            'extra': det_count - gt_count if det_count > gt_count else 0,
        }
        results.append(result)
        
        if save_images:
            output_path = output_dir / f"det_{sample['image_id']}.png"
            draw_detections(img_path, detections, gt_objects, output_path)
        
        if verbose:
            gt_desc = [f"{o['shape']}/{o['color']}" for o in gt_objects]
            det_desc = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
            print(f"{sample['image_id']}: GT={gt_count}, Det={det_count}")
            print(f"  GT: {', '.join(gt_desc)}")
            print(f"  Det: {', '.join(det_desc) if det_desc else 'NONE'}")
    
    return results


def print_stats(results: list):
    if not results:
        print("No results to display")
        return
    
    total_gt = sum(r['gt_count'] for r in results)
    total_det = sum(r['detected_count'] for r in results)
    total_missed = sum(r['missed'] for r in results)
    
    print()
    print("DETECTOR TEST RESULTS")
    print()
    print(f"{'Image':<25} {'GT':<6} {'Det':<6} {'Missed':<8}")
    print()
    
    for r in results:
        status = "OK" if r['missed'] == 0 else f"-{r['missed']}"
        print(f"{r['image_id']:<25} {r['gt_count']:<6} {r['detected_count']:<6} {status:<8}")
    
    print()
    print(f"{'TOTAL':<25} {total_gt:<6} {total_det:<6} {total_missed:<8}")
    print()
    print(f"Detection Rate: {total_det/total_gt:.1%}" if total_gt > 0 else "N/A")
    print(f"Avg GT per image: {total_gt/len(results):.1f}")
    print(f"Avg Det per image: {total_det/len(results):.1f}")
    
    if total_det < total_gt * 0.5:
        print()
        print("[!!] YOLO misses >50% of CLEVR objects!")
        print("[!!] Consider using SAM as primary detector.")


if __name__ == "__main__":
    results = test_detector(num_samples=10, conf_threshold=0.05, 
                            save_images=True, verbose=False)

    print_stats(results)
